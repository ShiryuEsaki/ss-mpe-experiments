#!/usr/bin/env python3
"""Select a threshold on validation data and evaluate one target test set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"evaluation failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--target", choices=("urmp", "maps", "musicnet"), default="urmp")
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parent
    source_root = repository / "src"
    upstream = source_root / "ss_nt_mpe_rc"
    run_dir = args.run_dir.resolve(strict=True)
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    if config.get("smoke_test"):
        raise RuntimeError("smoke runs do not retain a checkpoint")
    checkpoint = run_dir / "models/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    source_root_path = Path(config["dataset_root"]).resolve(strict=True)
    target_root = (
        args.target_root.resolve(strict=True)
        if args.target_root is not None else source_root_path
    )
    if args.target != "urmp" and args.target_root is None:
        raise ValueError("--target-root is required for MAPS and MusicNet")

    output = run_dir / "evaluation" / args.target
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), str(upstream), env.get("PYTHONPATH", "")]
    )
    common = [
        sys.executable, "-m", "ss_mpe_experiments.evaluator",
        "--checkpoint", str(checkpoint),
        "--urmp-split", str(run_dir / "split.json"),
        "--validation-manifest", str(run_dir / "manifests/urmp_validation.json"),
        "--seed", str(config["seed"]), "--sample-rate", "22050",
        "--device", "cuda", "--global-seed",
    ]
    validation_json = output / "source_validation_thresholds.json"
    run(common + [
        "--dataset", "urmp", "--role", "source_validation",
        "--data-base", str(source_root_path), "--out-json", str(validation_json),
        "--threshold-start", "0.10", "--threshold-stop", "0.95",
        "--threshold-step", "0.01",
    ], cwd=repository, env=env, log=output / "source_validation.log")
    validation = json.loads(validation_json.read_text(encoding="utf-8"))
    selected = validation["best"]["maximum_f1"]

    test_json = output / "target_test_fixed_threshold.json"
    run(common + [
        "--dataset", args.target, "--role", "target_test",
        "--data-base", str(target_root), "--out-json", str(test_json),
        "--fixed-threshold", str(selected["threshold"]),
    ], cwd=repository, env=env, log=output / "target_test.log")
    (output / "evaluation_complete.json").write_text(json.dumps({
        "checkpoint_selection": "minimum audio-only validation loss",
        "threshold_selection": "maximum macro F1 on URMP validation",
        "selected_threshold": selected,
        "target": args.target,
        "target_policy": "one fixed-threshold test pass",
        "validation_results": str(validation_json),
        "test_results": str(test_json),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"EVALUATION_COMPLETE output={output}")


if __name__ == "__main__":
    main()
