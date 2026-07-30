#!/usr/bin/env python3
"""Run CMIP_mask_exp5_baseline_simple notebook in batch mode and export artifacts."""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbformat.v4 import new_code_cell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute CMIP_mask_exp5_baseline_simple notebook and export artifacts")
    parser.add_argument("--project-dir", default="/glade/u/home/tsalin/CMIP", help="Project directory containing notebooks/")
    parser.add_argument(
        "--notebook",
        default="notebooks/trainings/mask/CMIP_mask_exp5_baseline_simple.ipynb",
        help="Notebook path relative to project dir",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots/plots_exp_5",
        help="Output folder (relative to project dir) for notebook plots",
    )
    parser.add_argument(
        "--evaluation-dir",
        default="model_evaluation/Baseline_simple",
        help="Output folder (relative to project dir) for history and quality dataframes",
    )
    parser.add_argument(
        "--executed-notebook",
        default="notebooks/CMIP_mask_exp5_baseline_simple.executed.ipynb",
        help="Where to save the executed notebook (relative to project dir)",
    )
    parser.add_argument("--timeout", type=int, default=0, help="Cell timeout in seconds. 0 means no timeout.")
    parser.add_argument("--kernel-name", default="python3", help="Kernel name used to run the notebook")
    return parser.parse_args()


def _append_export_cell(nb: nbformat.NotebookNode, evaluation_dir_abs: Path) -> None:
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
    _sanitize(mask_strategy),
    _sanitize(val_fraction),
    _sanitize(test_fraction),
]) + "_"
print(f"OUTPUT_PREFIX:{{_prefix}}")

_evaluation_dir = Path(r"{evaluation_dir_abs.as_posix()}")
_evaluation_dir.mkdir(parents=True, exist_ok=True)

if "baseline_history_df" not in globals():
    raise ValueError("baseline_history_df is missing. Ensure the training cell was executed.")
if "baseline_simple_quality_payload" not in globals():
    raise ValueError("baseline_simple_quality_payload is missing. Ensure the evaluation cell was executed.")

_history_path = _evaluation_dir / f"{{_prefix}}history_df.pkl"
_quality_path = _evaluation_dir / f"{{_prefix}}quality_df.pkl"

baseline_history_df.to_pickle(_history_path)
with open(_quality_path, "wb") as _qf:
    pickle.dump(baseline_simple_quality_payload, _qf)

print("Saved history:", _history_path)
print("Saved quality:", _quality_path)
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
    evaluation_dir = (project_dir / args.evaluation_dir).resolve()
    executed_notebook_path = (project_dir / args.executed_notebook).resolve()

    plots_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    executed_notebook_path.parent.mkdir(parents=True, exist_ok=True)

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    print(f"[INFO] Loading notebook: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    _append_export_cell(nb, evaluation_dir)

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
    for path, label in ((history_path, "history dataframe"), (quality_path, "quality dataframe")):
        if not path.exists():
            print(f"[ERROR] {label} was not produced as expected: {path}")
            return 2

    print("[INFO] Exported baseline-simple history and quality files successfully.")
    print("[DONE] CMIP_mask_exp5_baseline_simple batch run completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())