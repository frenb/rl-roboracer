"""Wrap ``beni_rl.headless_export`` into the mjlab_core export_command shape.

``mjlab_core.launch.export_command`` invokes::

    python export.py <task_id> --checkpoint <pt> --onnx-file <path>

``headless_export`` writes ``policy.onnx`` (and optionally ``policy.mnn``)
under an output directory; this wrapper copies them next to the checkpoint
path the worker recorded.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", help="Unused; accepted for mjlab_core argv compatibility")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx-file", required=True, type=Path)
    parser.add_argument("--skip-mnn", action="store_true",
                        help="ONNX only (default exports MNN too when MNN is installed)")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    onnx_path = args.onnx_file.expanduser().resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".beni-export-", dir=str(onnx_path.parent)) as tmp:
        out_dir = Path(tmp)
        cmd = [
            sys.executable, "-m", "beni_rl.headless_export",
            "--checkpoint", str(checkpoint),
            "--output-dir", str(out_dir),
        ]
        if args.skip_mnn:
            cmd.append("--skip-mnn")
        subprocess.check_call(cmd)
        produced = out_dir / "policy.onnx"
        if not produced.is_file():
            raise SystemExit(f"headless_export did not write {produced}")
        shutil.copy2(produced, onnx_path)
        print(f"Copied ONNX -> {onnx_path}", flush=True)
        mnn = out_dir / "policy.mnn"
        if mnn.is_file():
            mnn_dest = onnx_path.with_suffix(".mnn")
            shutil.copy2(mnn, mnn_dest)
            print(f"Copied MNN -> {mnn_dest}", flush=True)


if __name__ == "__main__":
    main()
