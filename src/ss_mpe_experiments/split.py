"""Create and validate recording-disjoint URMP train/validation/test splits."""

from __future__ import annotations

import json
import random
from pathlib import Path


URMP_IDS = tuple(f"{index:02d}" for index in range(1, 45))


def random_split(seed: int) -> dict[str, object]:
    identifiers = list(URMP_IDS)
    random.Random(int(seed)).shuffle(identifiers)
    return validate_split({
        "schema": "urmp_recording_split_v1",
        "split_seed": int(seed),
        "algorithm": (
            "random.Random(seed).shuffle(01..44); "
            "first 26 train, next 9 validation, last 9 test"
        ),
        "train_splits": sorted(identifiers[:26]),
        "validation_splits": sorted(identifiers[26:35]),
        "final_test_splits": sorted(identifiers[35:]),
    })


def load_split(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_split(payload)


def validate_split(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    keys = ("train_splits", "validation_splits", "final_test_splits")
    normalized = {
        key: [f"{int(value):02d}" for value in result.get(key, [])]
        for key in keys
    }
    if [len(normalized[key]) for key in keys] != [26, 9, 9]:
        raise ValueError("URMP split must contain 26 train, 9 validation, and 9 test recordings")
    combined = sum((normalized[key] for key in keys), [])
    if len(set(combined)) != 44 or sorted(combined) != list(URMP_IDS):
        raise ValueError("URMP roles must be disjoint and cover recordings 01--44 exactly once")
    result.update({key: sorted(values) for key, values in normalized.items()})
    return result


def save_split(path: str | Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(validate_split(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_audio_manifests(
    dataset_root: str | Path,
    split: dict[str, object],
    output_dir: str | Path,
    *,
    clip_seconds: float = 4.0,
) -> tuple[Path, Path]:
    """Write annotation-free training and fixed validation audio manifests."""
    import soundfile as sf

    dataset = Path(dataset_root).resolve(strict=True)
    output = Path(output_dir)
    split = validate_split(split)
    directories: dict[str, Path] = {}
    for path in dataset.iterdir():
        if path.is_dir() and len(path.name) >= 3 and path.name[:2].isdigit() \
                and path.name[2] == "_":
            identifier = path.name[:2]
            if identifier in directories:
                raise RuntimeError(f"duplicate URMP recording ID: {identifier}")
            directories[identifier] = path
    if set(directories) != set(URMP_IDS):
        missing = sorted(set(URMP_IDS) - set(directories))
        extra = sorted(set(directories) - set(URMP_IDS))
        raise RuntimeError(f"URMP inventory mismatch; missing={missing}, extra={extra}")

    def audio_path(identifier: str) -> Path:
        directory = directories[identifier]
        path = directory / f"AuMix_{directory.name}.wav"
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"invalid URMP mixture: {path}")
        info = sf.info(str(path))
        if info.frames <= 0 or info.duration < clip_seconds:
            raise RuntimeError(f"URMP mixture is too short: {path}")
        return path.resolve()

    training_paths = [audio_path(identifier) for identifier in split["train_splits"]]
    validation_paths = [
        audio_path(identifier) for identifier in split["validation_splits"]
    ]
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "urmp_train.json"
    validation_path = output / "urmp_validation.json"
    train_path.write_text(json.dumps({
        "schema": "ss_nt_mpe_audio_only_v1",
        "dataset": "urmp",
        "role": "train",
        "labels_read_by_training": False,
        "audio_paths": [str(path) for path in sorted(training_paths)],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    clips = []
    for path in sorted(validation_paths):
        maximum = max(0.0, sf.info(str(path)).duration - clip_seconds)
        for fraction in (0.2, 0.5, 0.8):
            clips.append({
                "audio_path": str(path),
                "offset_sec": round(maximum * fraction, 6),
                "duration_sec": clip_seconds,
            })
    validation_path.write_text(json.dumps({
        "schema": "ss_nt_mpe_audio_only_v1",
        "dataset": "urmp",
        "role": "validation",
        "labels_read_by_training": False,
        "fixed_across_checkpoints": True,
        "clip_fractions": [0.2, 0.5, 0.8],
        "clip_seconds": clip_seconds,
        "clips": clips,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return train_path, validation_path
