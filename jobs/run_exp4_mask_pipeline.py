#!/usr/bin/env python3
"""Run Experiment 4 notebook in batch mode and export artifacts.

This script executes the notebook:
CMIP/notebooks/trainings/mask/CMIP_mask_exp4.ipynb

It then:
1) Saves all image/png notebook outputs into CMIP/plots_exp_4/
2) Saves latent representations into
   CMIP/latent_representations_exp_4/<prefix>inv_latent_representations.pkl
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbformat.v4 import new_code_cell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Exp 4 notebook and export plots/latents")
    parser.add_argument(
        "--project-dir",
        default="/glade/u/home/tsalin/CMIP",
        help="Project directory containing notebooks/",
    )
    parser.add_argument(
        "--notebook",
        default="notebooks/trainings/mask/CMIP_mask_exp4.ipynb",
        help="Notebook path relative to project dir",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots_exp_4",
        help="Output folder (relative to project dir) for all plots",
    )
    parser.add_argument(
        "--latents-dir",
        default="latent_representations_exp_4",
        help="Output folder (relative to project dir) for latent files",
    )
    parser.add_argument(
        "--evaluation-dir",
        default="model_evaluation/exp4",
        help="Output folder (relative to project dir) for reconstruction dataframes",
    )
    parser.add_argument(
        "--executed-notebook",
        default="notebooks/CMIP_mask_exp4.executed.ipynb",
        help="Where to save the executed notebook (relative to project dir)",
    )
    parser.add_argument(
        "--num-sample",
        type=int,
        default=None,
        help="Override num_sample in notebook before execution",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Cell timeout in seconds. 0 means no timeout.",
    )
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help="Kernel name used to run the notebook",
    )
    return parser.parse_args()


def _override_num_sample(nb: nbformat.NotebookNode, value: int) -> bool:
    pattern = re.compile(r"^\s*num_sample\s*=\s*\d+\s*$")
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        lines = src.splitlines()
        changed = False
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"num_sample = {value}"
                changed = True
        if changed:
            cell["source"] = "\n".join(lines)
            return True
    return False


def _append_export_cell(
    nb: nbformat.NotebookNode,
    evaluation_dir_abs: Path,
    latents_dir_abs: Path,
) -> None:
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
    _sanitize(mask_strategy),
    _sanitize(val_fraction),
    _sanitize(test_fraction),
    _sanitize(cera_lambda_align),
]) + "_"
print(f"OUTPUT_PREFIX:{{_prefix}}")

_evaluation_dir = Path(r"{evaluation_dir_abs.as_posix()}")
_evaluation_dir.mkdir(parents=True, exist_ok=True)
_latent_dir = Path(r"{latents_dir_abs.as_posix()}")
_latent_dir.mkdir(parents=True, exist_ok=True)

if "inv_reconstruction_payload" not in globals():
    raise ValueError("inv_reconstruction_payload is missing. Ensure the evaluation cell was executed.")
if "inv_latent_by_climate" not in globals() or "inv_latent_test_metadata_by_climate" not in globals():
    raise ValueError("Invariant latent variables are missing. Ensure latent extraction cells were executed.")

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

_recon_path = _evaluation_dir / f"{{_prefix}}reconstruction_df.pkl"
_latent_path = _latent_dir / f"{{_prefix}}latent_representations_df.pkl"

with open(_recon_path, "wb") as _qf:
    pickle.dump(inv_reconstruction_payload, _qf)

inv_payload = _pack_latent_dict(inv_latent_by_climate, inv_latent_test_metadata_by_climate)
with open(_latent_path, "wb") as _f:
    pickle.dump(inv_payload, _f)

print("Saved reconstruction:", _recon_path)
print("Saved latent:", _latent_path)
""".strip()

    nb.cells.append(new_code_cell(code))


def _save_png_outputs(executed_nb: nbformat.NotebookNode, plots_dir_abs: Path, prefix: str = "") -> int:
    plots_dir_abs.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for ci, cell in enumerate(executed_nb.cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        for oi, out in enumerate(outputs, start=1):
            data = out.get("data", {}) if isinstance(out, dict) else {}
            png_b64 = data.get("image/png")
            if png_b64 is None:
                continue

            if isinstance(png_b64, list):
                png_b64 = "".join(png_b64)
            png_bytes = base64.b64decode(png_b64)

            out_name = f"{prefix}cell_{ci:03d}_output_{oi:03d}.png"
            out_path = plots_dir_abs / out_name
            out_path.write_bytes(png_bytes)
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

    if args.num_sample is not None:
        changed = _override_num_sample(nb, args.num_sample)
        if changed:
            print(f"[INFO] Overrode num_sample in notebook to {args.num_sample}")
        else:
            print("[WARN] Could not find a num_sample assignment to override.")

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
        # Save partial notebook to help debugging failed job runs.
        nbformat.write(nb, executed_notebook_path)
        print(f"[ERROR] Notebook execution failed. Partial notebook saved to: {executed_notebook_path}")
        print(err)
        return 1

    nbformat.write(executed_nb, executed_notebook_path)
    print(f"[INFO] Executed notebook saved to: {executed_notebook_path}")

    n_plots = _save_png_outputs(executed_nb, plots_dir)
    print(f"[INFO] Saved {n_plots} plot images to: {plots_dir}")

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

    recon_path = evaluation_dir / f"{prefix}reconstruction_df.pkl"
    latent_path = latents_dir / f"{prefix}latent_representations_df.pkl"
    for path, label in ((recon_path, "reconstruction dataframe"), (latent_path, "latent file")):
        if not path.exists():
            print(f"[ERROR] {label} was not produced as expected: {path}")
            return 2

    print("[INFO] Latent and reconstruction files validated successfully.")
    print("[DONE] Exp 4 mask batch run completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())