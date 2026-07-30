#!/usr/bin/env python3
"""Run CMIP_variable_exp5_CERA notebook in batch mode and export artifacts."""

from __future__ import annotations

import argparse
import base64
import pickle
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbformat.v4 import new_code_cell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute CMIP_variable_exp5_CERA notebook and export artifacts")
    parser.add_argument("--project-dir", default="/glade/u/home/tsalin/CMIP", help="Project directory containing notebooks/")
    parser.add_argument(
        "--notebook",
        default="notebooks/trainings/variable/CMIP_variable_exp5_CERA.ipynb",
        help="Notebook path relative to project dir",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots/plots_exp_5",
        help="Output folder (relative to project dir) for notebook plots",
    )
    parser.add_argument(
        "--latents-dir",
        default="latent_representations/latent_representations_exp_5",
        help="Output folder (relative to project dir) for latent files",
    )
    parser.add_argument(
        "--evaluation-dir",
        default="model_evaluation/CERA",
        help="Output folder (relative to project dir) for history and quality dataframes",
    )
    parser.add_argument(
        "--executed-notebook",
        default="notebooks/CMIP_variable_exp5_CERA.executed.ipynb",
        help="Where to save the executed notebook (relative to project dir)",
    )
    parser.add_argument("--timeout", type=int, default=0, help="Cell timeout in seconds. 0 means no timeout.")
    parser.add_argument("--kernel-name", default="python3", help="Kernel name used to run the notebook")
    parser.add_argument(
        "--run-tag",
        default="default",
        help="Identifier for this run, used to name the executed notebook output",
    )
    parser.add_argument("--num-sample", type=int, default=None, help="Override num_sample")
    parser.add_argument(
        "--autoencoder-type", choices=["MLP", "CNN"], default=None, help="Override chosen_autoencoder_type"
    )
    parser.add_argument(
        "--alignment-method",
        choices=["swd", "swdn", "adversarial"],
        default=None,
        help="Override inv_alignment_method",
    )
    parser.add_argument("--variable", default=None, help="Override variable")
    parser.add_argument("--val-fraction", type=float, default=None, help="Override val_fraction")
    parser.add_argument("--test-fraction", type=float, default=None, help="Override test_fraction")
    parser.add_argument("--cera-lambda-align", type=float, default=None, help="Override cera_lambda_align")
    parser.add_argument("--cera-lambda-pred", type=float, default=None, help="Override cera_lambda_pred")
    return parser.parse_args()


def _apply_hyperparameter_overrides(nb: nbformat.NotebookNode, overrides: dict) -> None:
    """Overwrite hyperparameter assignments (e.g. `variable = "pr"`) in-place across all code
    cells, so a shared notebook file can be run with different hyperparameters per job without
    editing it by hand."""
    remaining = set(overrides)
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for name, value in overrides.items():
            pattern = re.compile(rf"^{re.escape(name)}\s*=.*$", re.MULTILINE)
            new_source, n = pattern.subn(f"{name} = {value!r}", cell["source"])
            if n:
                cell["source"] = new_source
                remaining.discard(name)
    if remaining:
        raise ValueError(f"Hyperparameters not found in notebook, cannot override: {sorted(remaining)}")


def _append_export_cell(nb: nbformat.NotebookNode, evaluation_dir_abs: Path, latents_dir_abs: Path) -> None:
    code = f"""
from pathlib import Path
import pickle
import re as _re
import numpy as np
import pandas as pd

def _sanitize(val):
    return _re.sub(r'[^A-Za-z0-9._-]+', '_', str(val).strip())

_prefix = "_".join([
    _sanitize(setup_name),
    f"ns{{_sanitize(num_sample)}}",
    _sanitize(chosen_autoencoder_type),
    _sanitize(inv_alignment_method),
    _sanitize(variable),
    _sanitize(val_fraction),
    _sanitize(test_fraction),
    _sanitize(cera_lambda_align),
    _sanitize(cera_lambda_pred),
]) + "_"
print(f"OUTPUT_PREFIX:{{_prefix}}")

_evaluation_dir = Path(r"{evaluation_dir_abs.as_posix()}")
_evaluation_dir.mkdir(parents=True, exist_ok=True)
_latent_dir = Path(r"{latents_dir_abs.as_posix()}")
_latent_dir.mkdir(parents=True, exist_ok=True)

if "cera_history_df" not in globals():
    raise ValueError("cera_history_df is missing. Ensure the training cell was executed.")
if "cera_quality_payload" not in globals():
    raise ValueError("cera_quality_payload is missing. Ensure the evaluation cell was executed.")
if "cera_latent_by_climate" not in globals() or "cera_latent_test_metadata_by_climate" not in globals():
    raise ValueError("CERA latent variables are missing. Ensure latent extraction cells were executed.")

def _pack_latent_dict(latent_by_climate, meta_by_climate):
    packed = {{}}
    for climate, z in latent_by_climate.items():
        meta = meta_by_climate.get(climate, None)
        if hasattr(meta, "to_dict"):
            meta_obj = meta.to_dict(orient="list")
        else:
            meta_obj = meta
        packed[climate] = {{"latent": np.asarray(z), "metadata": meta_obj}}
    return packed

_history_path = _evaluation_dir / f"{{_prefix}}history_df.pkl"
_quality_path = _evaluation_dir / f"{{_prefix}}quality_df.pkl"
_latent_path = _latent_dir / f"{{_prefix}}latent_representations_df.pkl"

cera_history_df.to_pickle(_history_path)
with open(_quality_path, "wb") as _qf:
    pickle.dump(cera_quality_payload, _qf)

cera_payload = _pack_latent_dict(cera_latent_by_climate, cera_latent_test_metadata_by_climate)
with open(_latent_path, "wb") as _f:
    pickle.dump(cera_payload, _f)

print("Saved history:", _history_path)
print("Saved quality:", _quality_path)
print("Saved latent:", _latent_path)
""".strip()

    nb.cells.append(new_code_cell(code))


def _save_png_outputs(executed_nb: nbformat.NotebookNode, plots_dir_abs: Path, prefix: str = "") -> int:
    plots_dir_abs.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for ci, cell in enumerate(executed_nb.cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        for oi, out in enumerate(cell.get("outputs", []), start=1):
            data = out.get("data", {}) if isinstance(out, dict) else {}
            png_b64 = data.get("image/png")
            if png_b64 is None:
                continue
            if isinstance(png_b64, list):
                png_b64 = "".join(png_b64)
            out_name = f"{prefix}cell_{ci:03d}_output_{oi:03d}.png"
            (plots_dir_abs / out_name).write_bytes(base64.b64decode(png_b64))
            n_saved += 1
    return n_saved


def main() -> int:
    args = parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    notebook_path = (project_dir / args.notebook).resolve()
    plots_dir = (project_dir / args.plots_dir).resolve()
    latents_dir = (project_dir / args.latents_dir).resolve()
    evaluation_dir = (project_dir / args.evaluation_dir).resolve()
    executed_notebook_path = (project_dir / args.executed_notebook).resolve()

    plots_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    executed_notebook_path.parent.mkdir(parents=True, exist_ok=True)

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    print(f"[INFO] Loading notebook: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)

    overrides = {
        k: v
        for k, v in {
            "num_sample": args.num_sample,
            "chosen_autoencoder_type": args.autoencoder_type,
            "inv_alignment_method": args.alignment_method,
            "variable": args.variable,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "cera_lambda_align": args.cera_lambda_align,
            "cera_lambda_pred": args.cera_lambda_pred,
        }.items()
        if v is not None
    }
    if overrides:
        _apply_hyperparameter_overrides(nb, overrides)
        for name, value in overrides.items():
            print(f"[INFO] Override applied: {name} = {value!r}")

    _append_export_cell(nb, evaluation_dir, latents_dir)

    timeout = None if args.timeout == 0 else args.timeout
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=args.kernel_name,
        resources={"metadata": {"path": str(project_dir)}},
        allow_errors=False,
    )

    try:
        print("[INFO] Starting notebook execution...")
        executed_nb = client.execute()
        print("[INFO] Notebook execution completed.")
    except CellExecutionError as err:
        nbformat.write(nb, executed_notebook_path)
        print(f"[ERROR] Notebook execution failed. Partial notebook saved to: {executed_notebook_path}")
        print(err)
        return 1

    nbformat.write(executed_nb, executed_notebook_path)
    print(f"[INFO] Executed notebook saved to: {executed_notebook_path}")

    n_plots = _save_png_outputs(executed_nb, plots_dir)
    print(f"[INFO] Saved {n_plots} plot images to: {plots_dir}")

    # Extract the auto-computed prefix from the export cell's stdout output
    prefix = None
    export_cell = executed_nb.cells[-1]
    for output in export_cell.get("outputs", []):
        if output.get("output_type") == "stream" and output.get("name") == "stdout":
            for line in output.get("text", "").splitlines():
                if line.startswith("OUTPUT_PREFIX:"):
                    prefix = line[len("OUTPUT_PREFIX:"):]
                    break
        if prefix:
            break

    if not prefix:
        print("[ERROR] Could not extract OUTPUT_PREFIX from export cell output.")
        return 2

    print(f"[INFO] Auto-computed output prefix: {prefix}")

    history_path = evaluation_dir / f"{prefix}history_df.pkl"
    quality_path = evaluation_dir / f"{prefix}quality_df.pkl"
    latent_path = latents_dir / f"{prefix}latent_representations_df.pkl"

    for path, label in ((history_path, "history dataframe"), (quality_path, "quality dataframe"), (latent_path, "latent file")):
        if not path.exists():
            print(f"[ERROR] {label} was not produced as expected: {path}")
            return 2

    print("[INFO] Exported CERA history, quality, and latent files successfully.")
    print("[DONE] CMIP_variable_exp5_CERA batch run completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())