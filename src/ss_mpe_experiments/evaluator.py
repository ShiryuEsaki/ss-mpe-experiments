# SS_MPE_EXPERIMENTS_ONE_PASS_EVALUATOR_V1

import argparse
import json
import math
import os
import warnings
from decimal import Decimal
from pathlib import Path

import librosa
import mir_eval.multipitch
import numpy as np
import torch

mir_eval.multipitch.MAX_FREQ = 8000.0

from timbre_drill.datasets import NoteDataset
from .datasets import make_eval_dataset
from timbre_drill.framework import *
from timbre_drill.utils import *


class Dummy:
    pass


def load_model(checkpoint, device):
    try:
        return SS_NT.load(checkpoint, device=device)
    except Exception:
        return Timbre_Drill.load(checkpoint, device=device)


def to_builtin(value):
    if isinstance(value, dict):
        return {
            str(key): to_builtin(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]

    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        return to_builtin(value.detach().cpu().numpy())

    if isinstance(value, Path):
        return str(value)

    return value


def make_thresholds(start, stop, step):
    start_dec = Decimal(str(start))
    stop_dec = Decimal(str(stop))
    step_dec = Decimal(str(step))

    if step_dec <= 0:
        raise ValueError("threshold-step must be positive")

    if not (
        Decimal("0") <= start_dec <= Decimal("1")
        and Decimal("0") <= stop_dec <= Decimal("1")
    ):
        raise ValueError("threshold range must be within [0, 1]")

    if start_dec > stop_dec:
        raise ValueError("threshold-start must be <= threshold-stop")

    thresholds = []
    value = start_dec

    while value <= stop_dec:
        thresholds.append(float(value))
        value += step_dec

    if not thresholds:
        raise ValueError("empty threshold grid")

    return thresholds


def threshold_key(value):
    return f"{value:.3f}"


def validate_metrics(metrics, label):
    for key, value in metrics.items():
        numeric = float(value)

        if not math.isfinite(numeric):
            raise RuntimeError(
                f"Non-finite metric: {label} / {key} = {value}"
            )


def select_best(results_by_threshold, metric, maximize):
    rows = []

    for key, result in results_by_threshold.items():
        value = float(result["average_results"][metric])
        rows.append((value, float(result["threshold"]), key))

    if maximize:
        value, threshold_value, key = max(
            rows,
            key=lambda row: (row[0], -row[1]),
        )
    else:
        value, threshold_value, key = min(
            rows,
            key=lambda row: (row[0], row[1]),
        )

    return {
        "metric": metric,
        "value": value,
        "threshold": threshold_value,
        "threshold_key": key,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=("urmp", "maps", "musicnet"))
    parser.add_argument("--role", required=True, choices=("source_validation", "target_test"))
    parser.add_argument("--data-base", required=True)
    parser.add_argument("--urmp-split", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda"
    )
    parser.add_argument(
        "--global-seed",
        action="store_true",
        help=(
            "Enable seed_everything(), including "
            "torch.backends.cudnn.deterministic=True. "
            "Disabled by default to match the original evaluator."
        ),
    )
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--fixed-threshold", type=float, default=None)

    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    out_path = Path(args.out_json)

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if out_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing output: {out_path}"
        )

    thresholds = (
        [float(args.fixed_threshold)]
        if args.fixed_threshold is not None
        else make_thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)
    )

    threshold_keys = [
        threshold_key(value)
        for value in thresholds
    ]

    if len(set(threshold_keys)) != len(threshold_keys):
        raise RuntimeError(
            "Threshold keys are not unique; use a larger step"
        )

    if any(value < 0.0 or value > 1.0 for value in thresholds):
        raise RuntimeError("threshold must be in [0, 1]")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluator requested but CUDA is unavailable")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("=" * 72)
    print("One-pass MPE threshold sweep")
    print("=" * 72)
    print("checkpoint:", checkpoint)
    print("dataset:", args.dataset)
    print("role:", args.role)
    print("data_base:", args.data_base)
    print("seed:", args.seed)
    print("thresholds:", threshold_keys)
    print("threshold_count:", len(thresholds))
    print("device:", device)
    print("=" * 72)

    if args.global_seed:
        print("global_rng_seeding: enabled")
        seed_everything(args.seed)
    else:
        print(
            "global_rng_seeding: disabled "
            "(original evaluator compatible)"
        )

    model = load_model(str(checkpoint), device=device)
    model = model.to(device)
    model.eval()

    eval_set = make_eval_dataset(
        dataset_name=args.dataset,
        role=args.role,
        data_base=args.data_base,
        sample_rate=args.sample_rate,
        cqt=model.sliCQ,
        seed=args.seed,
        urmp_split_path=args.urmp_split,
        validation_manifest_path=args.validation_manifest,
    )

    evaluators = {
        key: MultipitchEvaluator()
        for key in threshold_keys
    }

    per_track = {
        key: {}
        for key in threshold_keys
    }

    prediction_rule = (
        "threshold("
        "filter_non_peaks(to_array(pitch_salience)), "
        "threshold"
        ")"
    )

    warnings.filterwarnings(
        "once",
        message=(
            "Estimate times not equal to reference times.*"
        ),
    )

    with torch.no_grad():
        for index, data in enumerate(eval_set):
            track = str(data[constants.KEY_TRACK])

            print(
                f"[{index + 1}/{len(eval_set)}] "
                f"{track}"
            )

            audio = (
                data[constants.KEY_AUDIO]
                .to(device)
                .unsqueeze(0)
            )

            if isinstance(eval_set, NoteDataset):
                times_ref = data[constants.KEY_TIMES]
                pitches, intervals = (
                    eval_set.get_ground_truth(track)
                )
                pitches = librosa.midi_to_hz(pitches)
                multi_pitch_ref = (
                    eval_set.notes_to_multi_pitch(
                        pitches,
                        intervals,
                        times_ref,
                    )
                )
            else:
                times_ref, multi_pitch_ref = (
                    eval_set.get_ground_truth(track)
                )

            features = model.get_all_features(
                audio,
                onset_mode=True,
            )
            input_hcqt = features["hcqt"]
            model_output = model(input_hcqt)
            pitch_salience = model_output["pitch_salience"]

            times_est = model.sliCQ.get_times(
                model.sliCQ.get_expected_frames(
                    audio.size(-1)
                )
            )

            filtered_salience = filter_non_peaks(
                to_array(pitch_salience)
            )

            for threshold_value, key in zip(
                thresholds,
                threshold_keys,
            ):
                activations = threshold(
                    filtered_salience,
                    threshold_value,
                ).squeeze(0)

                multi_pitch_est = (
                    eval_set.activations_to_multi_pitch(
                        activations,
                        model.sliCQ.get_midi_freqs(),
                    )
                )

                track_metrics = evaluators[key].evaluate(
                    times_est,
                    multi_pitch_est,
                    times_ref,
                    multi_pitch_ref,
                )

                validate_metrics(
                    track_metrics,
                    f"{key}/{track}",
                )

                evaluators[key].append_results(
                    track_metrics
                )

                per_track[key][track] = to_builtin(
                    track_metrics
                )

            del audio
            del features
            del input_hcqt
            del model_output
            del pitch_salience

    results_by_threshold = {}

    for threshold_value, key in zip(
        thresholds,
        threshold_keys,
    ):
        average_results, std_results = (
            evaluators[key].average_results()
        )

        validate_metrics(
            average_results,
            f"{key}/average",
        )
        validate_metrics(
            std_results,
            f"{key}/std",
        )

        physical_tracks = set(per_track[key])
        if not physical_tracks:
            raise RuntimeError(f"No evaluation tracks at {key}")
        if len(physical_tracks) != len(eval_set):
            raise RuntimeError(
                f"track count mismatch at {key}: {len(physical_tracks)} != {len(eval_set)}"
            )

        results_by_threshold[key] = {
            "threshold": threshold_value,
            "average_results": to_builtin(
                average_results
            ),
            "std_results": to_builtin(
                std_results
            ),
            "track_results": per_track[key],
        }

    best = {
        "maximum_f1": select_best(
            results_by_threshold,
            "mpe/f1-score",
            maximize=True,
        ),
        "maximum_accuracy": select_best(
            results_by_threshold,
            "mpe/accuracy",
            maximize=True,
        ),
        "minimum_total_error": select_best(
            results_by_threshold,
            "mpe/total error",
            maximize=False,
        ),
    }

    output = {
        "format_version": 1,
        "marker": "SS_MPE_EXPERIMENTS_ONE_PASS_EVALUATOR_V1",
        "checkpoint": str(checkpoint),
        "dataset": args.dataset,
        "role": args.role,
        "data_base": args.data_base,
        "urmp_split": args.urmp_split,
        "validation_manifest": args.validation_manifest,
        "seed": args.seed,
        "global_rng_seeding": bool(args.global_seed),
        "sample_rate": args.sample_rate,
        "prediction_rule": prediction_rule,
        "threshold_start": args.threshold_start,
        "threshold_stop": args.threshold_stop,
        "threshold_step": args.threshold_step,
        "thresholds": thresholds,
        "results_by_threshold": results_by_threshold,
        "best": best,
    }

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(str(out_path) + ".tmp")

    if temporary_path.exists():
        temporary_path.unlink()

    temporary_path.write_text(
        json.dumps(
            to_builtin(output),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    os.replace(temporary_path, out_path)

    print("=" * 72)
    print("Best results")
    print("=" * 72)
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
