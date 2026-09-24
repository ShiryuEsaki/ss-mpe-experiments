"""Annotation-bearing datasets used only after training has completed."""

from __future__ import annotations

import json
from pathlib import Path

from timbre_drill.datasets.Common.MAPS import MAPS
from timbre_drill.datasets.Common.MusicNet import MusicNet
from timbre_drill.datasets.MixedMultiPitch.URMP import URMP


MAPS_SOURCE_SPLITS = [
    "MAPS_AkPnBcht_2", "MAPS_AkPnBsdf_2", "MAPS_AkPnCGdD_2",
    "MAPS_AkPnStgb_2", "MAPS_SptkBGAm_2", "MAPS_SptkBGCl_2",
]
MAPS_TEST_SPLITS = ["MAPS_ENSTDkAm_2", "MAPS_ENSTDkCl_2"]


class DirectMAPS(MAPS):
    """MAPS wrapper for the extracted ``condition/MUS`` layout."""

    def get_tracks(self, split):
        directory = Path(self.base_dir) / split / "MUS"
        return [str(Path(split) / path.name) for path in sorted(directory.glob("*.wav"))]

    def get_audio_path(self, track):
        condition, name = Path(track).parts
        return str(Path(self.base_dir) / condition / "MUS" / name)


def _validation_basenames(manifest_path: Path) -> set[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("role") != "validation":
        raise RuntimeError(f"not a validation manifest: {manifest_path}")
    result = {Path(clip["audio_path"]).name for clip in payload["clips"]}
    if not result:
        raise RuntimeError(f"empty validation manifest: {manifest_path}")
    return result


def make_eval_dataset(
    *, dataset_name: str, role: str, data_base: str, sample_rate: int,
    cqt, seed: int, urmp_split_path: str, validation_manifest_path: str,
):
    name = dataset_name.lower()
    if role not in {"source_validation", "target_test"}:
        raise ValueError(role)

    if name == "urmp":
        split = json.loads(Path(urmp_split_path).read_text(encoding="utf-8"))
        ids = split["validation_splits" if role == "source_validation" else "final_test_splits"]
        return URMP(
            base_dir=data_base, splits=ids, sample_rate=sample_rate,
            cqt=cqt, seed=seed,
        )

    if name == "maps":
        splits = (
            [name.split("_")[1] for name in MAPS_SOURCE_SPLITS]
            if role == "source_validation"
            else [name.split("_")[1] for name in MAPS_TEST_SPLITS]
        )
        dataset = DirectMAPS(
            base_dir=data_base, splits=splits, sample_rate=sample_rate,
            cqt=cqt, seed=seed,
        )
        if role == "source_validation":
            allowed = _validation_basenames(Path(validation_manifest_path))
            dataset.tracks = [track for track in dataset.tracks if Path(track).name in allowed]
            if {Path(track).name for track in dataset.tracks} != allowed:
                missing = allowed - {Path(track).name for track in dataset.tracks}
                raise RuntimeError(f"MAPS validation tracks unavailable: {sorted(missing)}")
        return dataset

    if name == "musicnet":
        split_name = "train" if role == "source_validation" else "test"
        dataset = MusicNet(
            base_dir=data_base, splits=[split_name], sample_rate=sample_rate,
            cqt=cqt, seed=seed,
        )
        if role == "source_validation":
            allowed = {Path(name).stem for name in _validation_basenames(Path(validation_manifest_path))}
            dataset.tracks = [track for track in dataset.tracks if Path(track).name in allowed]
            if {Path(track).name for track in dataset.tracks} != allowed:
                missing = allowed - {Path(track).name for track in dataset.tracks}
                raise RuntimeError(f"MusicNet validation tracks unavailable: {sorted(missing)}")
        return dataset

    raise ValueError(f"unknown dataset: {dataset_name}")
