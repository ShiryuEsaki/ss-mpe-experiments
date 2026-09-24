#!/usr/bin/env python3
"""Train the proposed SS-MPE model on an arbitrary URMP 26/9/9 split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys


LOSS_WEIGHTS = {
    "LOSS_WEIGHT_SUPPORT": 1.2,
    "LOSS_WEIGHT_HARMONIC": 1.5,
    "LOSS_WEIGHT_SPARSITY": 1.5,
    "LOSS_WEIGHT_TIMBRE": 1.0,
    "LOSS_WEIGHT_GEOMETRIC": 1.0,
    "LOSS_WEIGHT_RECONSTRUCTION": 1.0,
    "LOSS_WEIGHT_TIME_SIMILARITY": 0.0,
}


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--split-seed", type=int)
    split_group.add_argument("--split-file", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--validation-interval", type=int, default=300)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parent
    source_root = repository / "src"
    upstream = source_root / "ss_nt_mpe_rc"
    if not (upstream / "timbre_drill").is_dir():
        raise RuntimeError("install the pinned upstream source with scripts/install_upstream.sh")

    dataset_root = args.dataset_root.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    sys.path.insert(0, str(source_root))
    from ss_mpe_experiments.split import (
        build_audio_manifests,
        load_split,
        random_split,
        save_split,
    )

    split = (
        random_split(args.split_seed)
        if args.split_seed is not None
        else load_split(args.split_file)
    )
    split_path = output / "split.json"
    save_split(split_path, split)
    train_manifest, validation_manifest = build_audio_manifests(
        dataset_root, split, output / "manifests"
    )

    pseudo = {"H": [1, 2, 3, 4], "q": 0.0, "r": 1.0}
    strict_enabled = True
    strict_lambda = 40.0

    max_steps = 2 if args.smoke else args.max_steps
    interval = 1 if args.smoke else args.validation_interval
    if max_steps <= 0 or interval <= 0 or max_steps % interval:
        raise ValueError("max-steps must be a positive multiple of validation-interval")

    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), str(upstream), env.get("PYTHONPATH", "")]
    )
    env.update({
        "BASELINE_RUN_DIR": str(output),
        "BASELINE_MAX_STEPS": str(max_steps),
        "BASELINE_CHECKPOINT_INTERVAL": str(interval),
        "BASELINE_SEED": str(args.seed),
        "SS_MPE_EXPERIMENTS_SMOKE_TEST": "1" if args.smoke else "0",
        "SS_MPE_EXPERIMENTS_TRAIN_AUDIO_MANIFEST": str(train_manifest),
        "SS_MPE_EXPERIMENTS_VALIDATION_AUDIO_MANIFEST": str(validation_manifest),
        "SS_MPE_EXPERIMENTS_VALIDATION_RNG_SEED": "9173",
        "HARMONIC_MAX_ORDER": str(max(pseudo["H"])),
        "HARMONIC_INCLUDED_ORDERS": ",".join(map(str, pseudo["H"])),
        "HARMONIC_POWER_EXPONENT": str(pseudo["q"]),
        "HARMONIC_AGGREGATION_ORDER": str(pseudo["r"]),
        "HARMONIC_WEIGHT_NORMALIZATION": "baseline_power_sum",
        "ENABLE_STRICT_POSITIVE": "1" if strict_enabled else "0",
        "LAMBDA_STRICT_POSITIVE": str(strict_lambda),
        "STRICT_POSITIVE_MODE": "relative_energy_base",
        "STRICT_CANDIDATE_THRESHOLD": "0.6",
        "STRICT_THRESHOLD_BASE_SNR_DB": "10.0",
        "STRICT_THRESHOLD_DB_2X": "-10.0",
        "STRICT_THRESHOLD_DB_3X": "-12.0",
        "STRICT_THRESHOLD_DB_1P5X": "-15.0",
        "STRICT_HARMONIC_BAND_RADIUS_BINS": "0",
        "STRICT_HARMONIC_TOLERANCE_CENTS": "",
        "STRICT_PEAK_RADIUS_BINS": "2",
        "STRICT_PEAK_PROMINENCE_THRESHOLD_DB": "6.0",
        "STRICT_RELATIVE_ENERGY_EPS": "1e-8",
        "STRICT_HARMONIC_ENERGY_REDUCTION": "sum",
        "ENABLE_PAIR_NEGATIVE": "0",
        "POSITIVE_SUPPORT_MASK_ALPHA": "0.0",
        "WANDB_ENABLED": "0",
        **{key: str(value) for key, value in LOSS_WEIGHTS.items()},
    })
    write_json(output / "resolved_config.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "proposed",
        "seed": args.seed,
        "dataset_root": str(dataset_root),
        "split": str(split_path),
        "pseudo_label": pseudo,
        "strict_positive_enabled": strict_enabled,
        "strict_positive_lambda": strict_lambda,
        "max_steps": max_steps,
        "validation_interval": interval,
        "checkpoint_selection": "minimum audio-only validation loss",
        "persistent_checkpoint_limit": 0 if args.smoke else 1,
        "smoke_test": args.smoke,
    })

    log_path = output / "train.stdout.log"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [sys.executable, "-m", "ss_mpe_experiments.trainer"],
            cwd=repository,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"training failed ({result.returncode}); see {log_path}")

    model_files = sorted(output.rglob("*.pt"))
    expected_models = [] if args.smoke else [output / "models/best.pt"]
    if model_files != expected_models:
        raise RuntimeError(f"checkpoint retention violation: {model_files}")
    history_path = output / "models/validation_history.jsonl"
    history = [
        json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_steps = list(range(interval, max_steps + 1, interval))
    if [int(row["step"]) for row in history] != expected_steps:
        raise RuntimeError("validation history is incomplete")
    if any(not math.isfinite(float(row["loss_total"])) for row in history):
        raise RuntimeError("validation history contains a non-finite loss")
    write_json(output / "completion.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "minimum audio-only validation loss",
        "best_checkpoint": None if args.smoke else "models/best.pt",
        "validation_points": len(history),
    })
    print(f"TRAINING_COMPLETE output={output}")


if __name__ == "__main__":
    main()
