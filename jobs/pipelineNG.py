#!/usr/bin/env python3
"""CMIP6 / CESM2 Data Pipeline NG
=================================
Loads historical (1950-2014) and SSP (2015-2100) daily data for CESM2,
extracts the requested surface and pressure-level variables, concatenates
them, and saves the result as Zarr to scratch.

The output layout matches `pipeline.py`, but each Zarr contains 16 variables:
  tas, ta850, ta500, huss, hus850, hus500,
  ua850, va850, ua500, va500, wap500, zg500,
  pr, psl, rsds, sfcWind

These outputs are intended to be consumed directly by
`build_multivariate_samples_optimized.py`.
"""

import argparse
import os
import shutil

import cftime
import intake
import numpy as np
import pandas as pd
import xarray as xr

# ── Configuration ─────────────────────────────────────────────────────────────
# Catalog of local CMIP6 data exposed through intake-esm.
CATALOG = "/glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json"

# Output root directory for generated Zarr stores.
SCRATCH = "/glade/derecho/scratch/tsalin/CMIP"

# Variables written to each output dataset, in the exact order requested.
VARIABLE_SPECS = [
    ("tas", "tas", None),
    ("ta850", "ta", 85000),
    ("ta500", "ta", 50000),
    ("huss", "huss", None),
    ("hus850", "hus", 85000),
    ("hus500", "hus", 50000),
    ("ua850", "ua", 85000),
    ("va850", "va", 85000),
    ("ua500", "ua", 50000),
    ("va500", "va", 50000),
    ("wap500", "wap", 50000),
    ("zg500", "zg", 50000),
    ("pr", "pr", None),
    ("psl", "psl", None),
    ("rsds", "rsds", None),
    ("sfcWind", "sfcWind", None),
]
OUTPUT_VARIABLES = [name for name, _, _ in VARIABLE_SPECS]
SOURCE_VARIABLES = list(dict.fromkeys(source for _, source, _ in VARIABLE_SPECS))

# Future scenarios to process.
SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]

# Fixed CMIP selectors for this workflow.
MEMBER = "r4i1p1f1"
TABLE = "day"
MODEL = "CESM2"
INSTITUTION = "NCAR"

# Time windows used when slicing each experiment.
HIST_YEARS = (1950, 2014)
SSP_YEARS = (2015, 2100)

# Zarr chunk sizes (tune for your ML workflow)
CHUNKS = {"time": 365, "lat": 96, "lon": 144}


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess(ds):
    """Decode CF time per-file before concat (avoids unit mismatch across files)."""
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        return xr.decode_cf(ds, decode_times=time_coder)
    except AttributeError:
        return xr.decode_cf(ds, use_cftime=True)


def get_years(ds):
    """Return integer year array, handling cftime and numpy datetime64."""
    t0 = ds["time"].values[0]
    if isinstance(t0, cftime.datetime):
        return np.array([t.year for t in ds["time"].values])
    return pd.DatetimeIndex(ds["time"].values).year.values


def list_available_variables(col):
    """Print variables available in the catalog for the active CMIP filters."""
    experiments = ["historical", *SSPS]

    print("\n── Catalog variable discovery ─────────────────────────────────────", flush=True)
    print(
        f"Filters: institution={INSTITUTION}, model={MODEL}, table={TABLE}, member={MEMBER}",
        flush=True,
    )

    union_vars = set()
    shared_vars = None

    for exp in experiments:
        cat = col.search(
            institution_id=INSTITUTION,
            source_id=MODEL,
            table_id=TABLE,
            member_id=MEMBER,
            experiment_id=exp,
        )
        vars_exp = sorted(set(cat.df["variable_id"].dropna().tolist()))
        print(f"  {exp:<10}: {len(vars_exp):3d} variables")

        union_vars.update(vars_exp)
        vars_set = set(vars_exp)
        shared_vars = vars_set if shared_vars is None else (shared_vars & vars_set)

    shared_sorted = sorted(shared_vars) if shared_vars is not None else []
    union_sorted = sorted(union_vars)

    print(f"  Shared across all selected experiments: {len(shared_sorted)}")
    print(f"  Union across selected experiments     : {len(union_sorted)}")

    if shared_sorted:
        print("  Shared variable names:")
        print("   " + ", ".join(shared_sorted))
    else:
        print("  Shared variable names: none")

    required_missing = sorted(set(SOURCE_VARIABLES) - union_vars)
    if required_missing:
        print(f"  Required source variables missing from catalog: {required_missing}")

    print("───────────────────────────────────────────────────────────────────", flush=True)


def _detect_vertical_dim(da: xr.DataArray) -> str:
    vertical_dims = [dim for dim in da.dims if dim not in {"time", "lat", "lon"}]
    if len(vertical_dims) != 1:
        raise ValueError(
            f"Expected exactly one vertical dimension for {da.name}, found {vertical_dims}."
        )
    return vertical_dims[0]


def _select_pressure_level(da: xr.DataArray, level_pa: int) -> xr.DataArray:
    level_dim = _detect_vertical_dim(da)
    coord_values = np.asarray(da[level_dim].values)

    if not np.issubdtype(coord_values.dtype, np.number):
        raise ValueError(f"Vertical coordinate for {da.name} is not numeric: {level_dim}")

    if np.any(np.isclose(coord_values, level_pa)):
        idx = int(np.where(np.isclose(coord_values, level_pa))[0][0])
        return da.isel({level_dim: idx}).squeeze(drop=True).reset_coords(level_dim, drop=True)

    if level_pa % 100 == 0:
        level_hpa = level_pa / 100.0
        if np.any(np.isclose(coord_values, level_hpa)):
            idx = int(np.where(np.isclose(coord_values, level_hpa))[0][0])
            print(f"    selecting {level_hpa:g} {level_dim} for {da.name} (converted from {level_pa} Pa)")
            return da.isel({level_dim: idx}).squeeze(drop=True).reset_coords(level_dim, drop=True)

    target = level_pa if float(np.nanmax(np.abs(coord_values))) > 2000 else level_pa / 100.0
    idx = int(np.abs(coord_values - target).argmin())
    chosen = float(coord_values[idx])
    print(
        f"    WARNING: exact level {level_pa} Pa not found for {da.name}; using nearest {chosen:g} on {level_dim}",
        flush=True,
    )
    return da.isel({level_dim: idx}).squeeze(drop=True).reset_coords(level_dim, drop=True)


def select_spatiotemporal_variables(ds: xr.Dataset) -> list[str]:
    valid = []
    for var in ds.data_vars:
        da = ds[var]
        if all(dim in da.dims for dim in ["time", "lat", "lon"]) and np.issubdtype(da.dtype, np.number):
            valid.append(var)
    return valid


def make_patch_slices(n_lat_total: int, n_lon_total: int, p_lat: int, p_lon: int) -> list[tuple[slice, slice]]:
    lat_starts = list(range(0, n_lat_total - p_lat + 1, p_lat))
    lon_starts = list(range(0, n_lon_total - p_lon + 1, p_lon))
    slices = []
    for i0 in lat_starts:
        for j0 in lon_starts:
            slices.append((slice(i0, i0 + p_lat), slice(j0, j0 + p_lon)))
    return slices


def build_patch_catalog(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    patch_slices: list[tuple[slice, slice]],
) -> pd.DataFrame:
    rows = []
    for patch_id, (sl_lat, sl_lon) in enumerate(patch_slices):
        lat_start = int(sl_lat.start)
        lat_stop = int(sl_lat.stop)
        lon_start = int(sl_lon.start)
        lon_stop = int(sl_lon.stop)
        rows.append(
            {
                "patch_id": patch_id,
                "lat_start_idx": lat_start,
                "lat_stop_idx": lat_stop,
                "lon_start_idx": lon_start,
                "lon_stop_idx": lon_stop,
                "lat_start": float(lat_vals[lat_start]),
                "lat_stop": float(lat_vals[lat_stop - 1]),
                "lon_start": float(lon_vals[lon_start]),
                "lon_stop": float(lon_vals[lon_stop - 1]),
            }
        )
    return pd.DataFrame(rows)


def _choose_pair_indices(
    n_times: int,
    n_patches: int,
    max_samples_per_climate: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    n_total = n_times * n_patches
    if max_samples_per_climate is not None and n_total > max_samples_per_climate:
        return rng.choice(n_total, size=max_samples_per_climate, replace=False)
    return np.arange(n_total, dtype=np.int64)


def load_variable(col, output_variable: str, source_variable: str, level_pa: int | None, experiment: str, year_range):
    """Load one variable/experiment, lazy-filtered to year_range (inclusive)."""
    cat = col.search(
        institution_id=INSTITUTION,
        source_id=MODEL,
        experiment_id=experiment,
        variable_id=source_variable,
        table_id=TABLE,
        member_id=MEMBER,
    )
    if len(cat.df) == 0:
        raise ValueError(f"Required variable {source_variable}/{experiment} not found in catalog")

    paths = cat.df.sort_values("path")["path"].tolist()
    print(f"  {output_variable} ← {source_variable}: {len(paths)} files", flush=True)

    ds_peek = xr.open_dataset(paths[0], decode_times=False)
    drop_vars = [n for n, v in ds_peek.variables.items() if v.dtype == object]
    ds_peek.close()

    ds = xr.open_mfdataset(
        paths,
        combine="nested",
        concat_dim="time",
        chunks=CHUNKS,
        drop_variables=drop_vars,
        decode_times=False,
        join="override",
        preprocess=preprocess,
    )

    years = get_years(ds)
    mask = xr.DataArray((years >= year_range[0]) & (years <= year_range[1]), dims="time")
    ds = ds.sel(time=mask)[[source_variable]]

    # Drop duplicate time steps (can occur when files slightly overlap)
    _, unique_idx = np.unique(ds.time.values, return_index=True)
    if len(unique_idx) < ds.sizes["time"]:
        print(f"    dropping {ds.sizes['time'] - len(unique_idx)} duplicate time steps")
        ds = ds.isel(time=unique_idx)

    da = ds[source_variable]
    if level_pa is not None:
        da = _select_pressure_level(da, level_pa)

    da = da.rename(output_variable)
    if set(da.dims) != {"time", "lat", "lon"}:
        raise ValueError(
            f"Variable {output_variable} does not have expected dims time/lat/lon after selection: {da.dims}"
        )
    return da.to_dataset()


def align_experiment_datasets(datasets: list[xr.Dataset]) -> list[xr.Dataset]:
    """Align all variables on the common time/lat/lon coordinates.

    The SSP inputs should already share the same daily grid, but in practice
    some variables can end up with slightly different time indexes after file
    concatenation and duplicate removal. Use the strict coordinate intersection
    so that only samples present for every variable survive.
    """
    if not datasets:
        raise ValueError("No datasets were provided for alignment")

    aligned_datasets = list(xr.align(*datasets, join="inner", copy=False))
    if aligned_datasets[0].sizes.get("time", 0) == 0:
        raise ValueError("No common time coordinates remained after alignment")

    return [ds.sortby("time") for ds in aligned_datasets]


def load_experiment(col, experiment, year_range):
    """Load and merge all requested variables for one experiment."""
    print(f"\n[{experiment}] Loading {year_range[0]}–{year_range[1]} ...", flush=True)
    datasets = [
        load_variable(col, output_variable, source_variable, level_pa, experiment, year_range)
        for output_variable, source_variable, level_pa in VARIABLE_SPECS
    ]
    aligned_datasets = align_experiment_datasets(datasets)
    return xr.merge(aligned_datasets, compat="override", join="exact")


def build_multivariate_samples(
    ds: xr.Dataset,
    scenario: str,
    selected_variables: list[str],
    patch_slices: list[tuple[slice, slice]],
    n_lat: int,
    n_lon: int,
    time_stride: int,
    max_samples_per_climate: int | None,
    random_seed: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds_small = ds[selected_variables].isel({"time": slice(0, None, time_stride)})
    time_coord = ds_small["time"]
    time_values = time_coord.values
    months = np.asarray(time_coord.dt.month.values, dtype=np.int16)
    seasons = np.asarray([season_from_month(int(m)) for m in months], dtype=object)
    time_strings = np.asarray(time_values.astype(str), dtype=object)

    n_times = len(time_values)
    n_patches = len(patch_slices)
    rng = np.random.default_rng(random_seed + scenario_seed_offset(scenario))
    pair_indices = _choose_pair_indices(n_times, n_patches, max_samples_per_climate, rng)

    # Preload requested variables once and switch to direct NumPy indexing inside the hot loop.
    arrays = {
        var: np.asarray(ds_small[var].transpose("time", "lat", "lon").values, dtype=np.float32)
        for var in selected_variables
    }

    patch_lat_starts = np.fromiter((sl_lat.start for sl_lat, _ in patch_slices), dtype=np.int32, count=n_patches)
    patch_lon_starts = np.fromiter((sl_lon.start for _, sl_lon in patch_slices), dtype=np.int32, count=n_patches)

    expected_size = len(selected_variables) * n_lat * n_lon
    features = np.empty((len(pair_indices), expected_size), dtype=np.float32)

    meta_rows: list[dict] = []
    pair_rows: list[dict] = []

    invalid_shape_count = 0
    all_nan_count = 0
    size_mismatch_count = 0
    samples_with_any_nan = 0
    total_nan_values_imputed = 0

    nan_values_by_variable = {var: 0 for var in selected_variables}
    samples_with_nan_by_variable = {var: 0 for var in selected_variables}

    keep_idx = 0
    for sample_index, pair_idx in enumerate(pair_indices):
        ti = int(pair_idx // n_patches)
        pi = int(pair_idx % n_patches)
        lat0 = int(patch_lat_starts[pi])
        lon0 = int(patch_lon_starts[pi])
        lat1 = lat0 + n_lat
        lon1 = lon0 + n_lon

        valid_patch = True
        sample_had_nan = False
        vec = np.empty(expected_size, dtype=np.float32)
        write_pos = 0

        for var in selected_variables:
            arr = arrays[var][ti, lat0:lat1, lon0:lon1]
            if arr.shape != (n_lat, n_lon):
                valid_patch = False
                invalid_shape_count += 1
                break

            flat = arr.ravel().copy()
            finite_mask = np.isfinite(flat)
            n_finite = int(finite_mask.sum())
            if n_finite == 0:
                valid_patch = False
                all_nan_count += 1
                break

            if n_finite != flat.size:
                nan_count = int(flat.size - n_finite)
                sample_had_nan = True
                total_nan_values_imputed += nan_count
                nan_values_by_variable[var] += nan_count
                samples_with_nan_by_variable[var] += 1
                fill_value = float(flat[finite_mask].mean())
                flat[~finite_mask] = fill_value

            next_pos = write_pos + flat.size
            vec[write_pos:next_pos] = flat
            write_pos = next_pos

        if not valid_patch:
            continue

        if write_pos != expected_size:
            size_mismatch_count += 1
            continue

        month = int(months[ti])
        season = seasons[ti]
        time_str = time_strings[ti]

        features[keep_idx] = vec
        meta_rows.append(
            {
                "scenario": scenario,
                "time": time_str,
                "month": month,
                "season": season,
                "patch_id": pi,
            }
        )
        pair_rows.append(
            {
                "scenario": scenario,
                "sample_index": int(sample_index),
                "time_index": ti,
                "patch_id": pi,
                "time": time_str,
                "month": month,
                "season": season,
            }
        )
        if sample_had_nan:
            samples_with_any_nan += 1
        keep_idx += 1

    if keep_idx == 0:
        raise ValueError(f"No valid samples built for scenario={scenario}.")

    features = features[:keep_idx]

    diagnostics_df = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "n_pairs_considered": int(n_times * n_patches),
                "n_pairs_drawn": int(len(pair_indices)),
                "n_samples_kept": int(keep_idx),
                "n_samples_rejected_shape": int(invalid_shape_count),
                "n_samples_rejected_all_nan": int(all_nan_count),
                "n_samples_rejected_size_mismatch": int(size_mismatch_count),
                "n_samples_with_any_nan": int(samples_with_any_nan),
                "n_nan_values_imputed": int(total_nan_values_imputed),
            }
        ]
    )

    nan_summary_rows = [
        {
            "scenario": scenario,
            "variable": var,
            "n_nan_values_imputed": int(nan_values_by_variable[var]),
            "n_samples_with_nan": int(samples_with_nan_by_variable[var]),
        }
        for var in selected_variables
    ]

    return features, pd.DataFrame(meta_rows), pd.DataFrame(pair_rows), diagnostics_df, pd.DataFrame(nan_summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='CMIP6/CESM2 data pipeline NG')
    parser.add_argument('--base-path', default=SCRATCH, help='Root directory containing historical/sspXXX Zarr stores')
    args = parser.parse_args()

    climate_paths = {
        "historical": f"{args.base_path}/historical/CESM2_historical_1950_2014.zarr",
        "ssp126": f"{args.base_path}/ssp126/CESM2_ssp126_2015_2100.zarr",
        "ssp245": f"{args.base_path}/ssp245/CESM2_ssp245_2015_2100.zarr",
        "ssp370": f"{args.base_path}/ssp370/CESM2_ssp370_2015_2100.zarr",
        "ssp585": f"{args.base_path}/ssp585/CESM2_ssp585_2015_2100.zarr",
    }

    print("Opening intake catalog ...", flush=True)
    col = intake.open_esm_datastore(CATALOG)
    list_available_variables(col)

    for step in ("historical", "ssp"):
        for exp in (["historical"] if step == "historical" else SSPS):
            ds = load_experiment(col, exp, HIST_YEARS if exp == "historical" else SSP_YEARS)
            out_path = f"{args.base_path}/{exp}/CESM2_{exp}_2015_2100.zarr" if exp != "historical" else f"{args.base_path}/historical/CESM2_historical_1950_2014.zarr"
            ds = _prepare_for_zarr(ds)
            if os.path.exists(out_path):
                shutil.rmtree(out_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            ds.to_zarr(out_path, mode='w', consolidated=True)
            print(f"Saved NG zarr: {out_path}")

    print('Pipeline NG complete.', flush=True)


def _prepare_for_zarr(ds):
    ds = ds.unify_chunks().chunk(CHUNKS)
    for name in ds.variables:
        ds[name].encoding.pop('chunks', None)
    return ds


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def scenario_seed_offset(scenario: str) -> int:
    import hashlib
    digest = hashlib.md5(scenario.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1000


if __name__ == "__main__":
    main()