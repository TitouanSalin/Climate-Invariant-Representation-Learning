#!/usr/bin/env python3
"""Run CMIP_analysis_agregate notebook in batch mode.

Unlike the exp5 training notebooks, CMIP_analysis_agregate.ipynb already saves its
own output (the aggregated DataFrame) to reportGraph/ in its last code cells — this
script only executes the notebook, optionally overriding its hyperparameters, and
verifies that the DataFrame file the notebook reports having saved actually exists.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute CMIP_analysis_agregate notebook")
    parser.add_argument("--project-dir", default="/glade/u/home/tsalin/CMIP", help="Project directory containing notebooks/")
    parser.add_argument(
        "--notebook",
        default="notebooks/analysis/CMIP_analysis_agregate.ipynb",
        help="Notebook path relative to project dir",
    )
    parser.add_argument(
        "--executed-notebook",
        default="notebooks/CMIP_analysis_agregate.executed.ipynb",
        help="Where to save the executed notebook (relative to project dir)",
    )
    parser.add_argument("--timeout", type=int, default=0, help="Cell timeout in seconds. 0 means no timeout.")
    parser.add_argument("--kernel-name", default="python3", help="Kernel name used to run the notebook")
    parser.add_argument(
        "--run-tag",
        default="default",
        help="Identifier for this run (only used for logging here; the executed notebook path is fixed)",
    )
    parser.add_argument("--num-sample", type=int, default=None, help="Override num_sample")
    parser.add_argument(
        "--autoencoder-type", choices=["MLP", "CNN"], default=None, help="Override chosen_autoencoder_type"
    )
    parser.add_argument(
        "--alignment-method",
        choices=["swd", "swdn", "adversarial"],
        default=None,
        help="Override inv_alignment_method (base/default value, used for output file naming only)",
    )
    parser.add_argument("--variable", default=None, help="Override variable")
    parser.add_argument("--val-fraction", type=float, default=None, help="Override val_fraction")
    parser.add_argument("--test-fraction", type=float, default=None, help="Override test_fraction")
    parser.add_argument("--cera-lambda-align", type=float, default=None, help="Override cera_lambda_align")
    parser.add_argument("--cera-lambda-pred", type=float, default=None, help="Override cera_lambda_pred")
    return parser.parse_args()


def _apply_hyperparameter_overrides(nb: nbformat.NotebookNode, overrides: dict) -> None:
    """Overwrite hyperparameter assignments (e.g. `variable = "pr"`) in-place across all code
    cells, so the shared notebook file can be run with different hyperparameters per job
    without editing it by hand."""
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


def _find_saved_df_path(executed_nb: nbformat.NotebookNode) -> str | None:
    """Scan stdout outputs for the `Saved: <path>` line the notebook's export cell prints."""
    for cell in executed_nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream" or output.get("name") != "stdout":
                continue
            for line in output.get("text", "").splitlines():
                if line.startswith("Saved: "):
                    return line[len("Saved: "):].strip()
    return None


def main() -> int:
    args = parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    notebook_path = (project_dir / args.notebook).resolve()
    executed_notebook_path = (project_dir / args.executed_notebook).resolve()
    executed_notebook_path.parent.mkdir(parents=True, exist_ok=True)

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    print(f"[INFO] Run tag: {args.run_tag}")
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

    saved_df_path = _find_saved_df_path(executed_nb)
    if not saved_df_path:
        print("[ERROR] Could not find the 'Saved: <path>' line printed by the notebook's Part 8 cell.")
        return 2

    if not Path(saved_df_path).exists():
        print(f"[ERROR] Notebook reported saving the DataFrame to {saved_df_path}, but the file does not exist.")
        return 2

    print(f"[INFO] Aggregated DataFrame confirmed on disk: {saved_df_path}")
    print("[DONE] CMIP_analysis_agregate batch run completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
