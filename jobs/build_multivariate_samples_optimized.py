#!/usr/bin/env python3
"""Build multivariate CMIP samples and save them for notebook reuse.

This script externalizes the expensive sample-construction step from the
notebook so it can be run in a PBS job.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


LAT_NAME = "lat"
LON_NAME = "lon"
TIME_NAME = "time"
DEFAULT_CLIMATES = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multivariate CMIP samples")
    parser.add_argument(
        "--base-path",
        default="/glade/derecho/scratch/tsalin/CMIP",
        help="Root directory containing historical/sspXXX Zarr stores",
    )
    parser.add_argument(
        "--output-dir",
        default="/glade/derecho/scratch/tsalin/CMIP/derived/multivariate_samples_v1",
        help="Directory where features and metadata are written",
    )
    parser.add_argument("--max-abs-lat", type=float, default=30.0)
    parser.add_argument("--patch-size-km", type=float, default=1000.0)
    parser.add_argument("--time-stride", type=int, default=24)
    parser.add_argument("--max-samples-per-climate", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Optional explicit variable list (e.g. --variables tas huss pr psl rsds sfcWind)",
    )
    return parser.parse_args()


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def scenario_seed_offset(scenario: str) -> int:
    digest = hashlib.md5(scenario.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1000


def subset_lat_band(ds: xr.Dataset, lat_limit: float) -> xr.Dataset:
    latitudes = ds[LAT_NAME]
    return ds.where(np.abs(latitudes) <= lat_limit, drop=True)


def select_spatiotemporal_variables(ds: xr.Dataset) -> list[str]:
    valid = []
    for var in ds.data_vars:
        da = ds[var]
        if all(dim in da.dims for dim in [TIME_NAME, LAT_NAME, LON_NAME]) and np.issubdtype(da.dtype, np.number):
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
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds_small = ds[selected_variables].isel({TIME_NAME: slice(0, None, time_stride)})
    time_coord = ds_small[TIME_NAME]
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
        var: np.asarray(ds_small[var].transpose(TIME_NAME, LAT_NAME, LON_NAME).values, dtype=np.float32)
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
    args = parse_args()

    climate_paths = {
        "historical": f"{args.base_path}/historical/CESM2_historical_1950_2014.zarr",
        "ssp126": f"{args.base_path}/ssp126/CESM2_ssp126_2015_2100.zarr",
        "ssp245": f"{args.base_path}/ssp245/CESM2_ssp245_2015_2100.zarr",
        "ssp370": f"{args.base_path}/ssp370/CESM2_ssp370_2015_2100.zarr",
        "ssp585": f"{args.base_path}/ssp585/CESM2_ssp585_2015_2100.zarr",
    }

    print("Opening climate datasets...")
    climate_datasets = {c: xr.open_zarr(climate_paths[c]) for c in DEFAULT_CLIMATES}

    print(f"Subsetting latitude band +/- {args.max_abs_lat} deg...")
    band_datasets = {c: subset_lat_band(climate_datasets[c], args.max_abs_lat) for c in DEFAULT_CLIMATES}
    auto_variables = select_spatiotemporal_variables(band_datasets["historical"])
    if not auto_variables:
        raise ValueError("No numeric variables with time/lat/lon dimensions were found.")

    if args.variables:
        selected_variables = list(args.variables)
        missing_by_climate = {}
        for climate in DEFAULT_CLIMATES:
            available = set(select_spatiotemporal_variables(band_datasets[climate]))
            missing = [v for v in selected_variables if v not in available]
            if missing:
                missing_by_climate[climate] = missing
        if missing_by_climate:
            raise ValueError(
                f"Requested variables are missing in some climates: {missing_by_climate}"
            )
    else:
        selected_variables = auto_variables

    lat_vals = band_datasets["historical"][LAT_NAME].values
    lon_vals = band_datasets["historical"][LON_NAME].values
    lat_res_deg = float(np.median(np.abs(np.diff(lat_vals))))
    lon_res_deg = float(np.median(np.abs(np.diff(lon_vals))))

    km_per_deg = 111.32
    patch_lat_deg = args.patch_size_km / km_per_deg
    patch_lon_deg = args.patch_size_km / km_per_deg

    n_lat = max(2, int(round(patch_lat_deg / lat_res_deg)))
    n_lon = max(2, int(round(patch_lon_deg / lon_res_deg)))
    patch_slices = make_patch_slices(len(lat_vals), len(lon_vals), n_lat, n_lon)
    patch_catalog = build_patch_catalog(lat_vals, lon_vals, patch_slices)
    grid_points_per_patch = int(n_lat * n_lon)

    print(
        f"Patch grid: {n_lat} x {n_lon} | Variables: {len(selected_variables)} | Patches: {len(patch_slices)}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_sampling_rows = []
    for climate in DEFAULT_CLIMATES:
        n_time_before_stride = int(band_datasets[climate].sizes[TIME_NAME])
        n_patches = int(len(patch_slices))
        pre_sampling_rows.append(
            {
                "scenario": climate,
                "n_time_before_stride": n_time_before_stride,
                "n_patches": n_patches,
                "n_samples_before_stride_and_cap": int(n_time_before_stride * n_patches),
            }
        )
    pre_sampling_df = pd.DataFrame(pre_sampling_rows).set_index("scenario")
    pre_sampling_df.to_csv(output_dir / "pre_sampling_df.csv")
    patch_catalog.to_csv(output_dir / "patch_catalog.csv", index=False)

    sample_count_rows = []
    diagnostics_rows = []
    nan_summary_frames = []

    for climate in DEFAULT_CLIMATES:
        print(f"Building samples for {climate}...")
        features, metadata, pair_trace, diagnostics_df, nan_summary_df = build_multivariate_samples(
            ds=band_datasets[climate],
            scenario=climate,
            selected_variables=selected_variables,
            patch_slices=patch_slices,
            n_lat=n_lat,
            n_lon=n_lon,
            time_stride=args.time_stride,
            max_samples_per_climate=args.max_samples_per_climate,
            random_seed=args.random_seed,
        )

        np.save(output_dir / f"features_{climate}.npy", features)
        metadata.to_csv(output_dir / f"metadata_{climate}.csv", index=False)
        pair_trace.to_csv(output_dir / f"sample_pairs_{climate}.csv", index=False)
        diagnostics_rows.append(diagnostics_df.iloc[0].to_dict())
        nan_summary_frames.append(nan_summary_df)
        sample_count_rows.append({"scenario": climate, "n_samples": int(features.shape[0])})
        print(f"  saved: {features.shape[0]} samples")

    pd.DataFrame(sample_count_rows).to_csv(output_dir / "sample_count.csv", index=False)
    pd.DataFrame(diagnostics_rows).to_csv(output_dir / "sampling_diagnostics.csv", index=False)
    pd.concat(nan_summary_frames, ignore_index=True).to_csv(output_dir / "nan_summary_by_variable.csv", index=False)

    config = {
        "base_path": args.base_path,
        "climate_order": DEFAULT_CLIMATES,
        "climate_colors": {
            "historical": "#4c72b0",
            "ssp126": "#63cab4",
            "ssp245": "#3f8f51",
            "ssp370": "#dd8452",
            "ssp585": "#c44e52",
        },
        "max_abs_lat": args.max_abs_lat,
        "patch_size_km": args.patch_size_km,
        "time_stride": args.time_stride,
        "max_samples_per_climate": args.max_samples_per_climate,
        "random_seed": args.random_seed,
        "selected_variables": selected_variables,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "grid_points_per_patch": grid_points_per_patch,
        "n_patches": len(patch_slices),
        "pre_sampling_df_path": str(output_dir / "pre_sampling_df.csv"),
        "patch_catalog_path": str(output_dir / "patch_catalog.csv"),
        "sample_count_path": str(output_dir / "sample_count.csv"),
        "sampling_diagnostics_path": str(output_dir / "sampling_diagnostics.csv"),
        "nan_summary_by_variable_path": str(output_dir / "nan_summary_by_variable.csv"),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
