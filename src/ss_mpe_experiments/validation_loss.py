"""Audio-only datasets for the proposed-method experiments.

Training manifests contain only ``audio_paths``. Validation manifests contain
explicit, immutable ``clips`` entries so every checkpoint, seed, and condition is
compared on exactly the same audio excerpts. No annotation path is accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from timbre_drill.utils.data import constants


def _read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("labels_read_by_training") is not False:
        raise RuntimeError(f"manifest must explicitly forbid label access: {path}")
    forbidden = ("label", "annot", "ground_truth", "midi_path")
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    if any(token in serialized for token in forbidden):
        # Allow only the required policy key itself.
        scrubbed = serialized.replace('"labels_read_by_training": false', "")
        if any(token in scrubbed for token in forbidden):
            raise RuntimeError(f"annotation-like content in audio manifest: {path}")
    return payload


def _load_clip(path: Path, sample_rate: int, offset_sec: float, n_secs: float) -> torch.Tensor:
    if path.suffix.lower() != ".wav":
        raise RuntimeError(f"only WAV input is allowed: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    samples, actual_rate = librosa.load(
        path,
        sr=sample_rate,
        mono=True,
        offset=float(offset_sec),
        duration=float(n_secs),
    )
    if int(actual_rate) != int(sample_rate):
        raise RuntimeError(f"unexpected sample rate {actual_rate}: {path}")
    target = int(round(float(n_secs) * int(sample_rate)))
    values = np.asarray(samples, dtype=np.float32)
    if values.size < target:
        values = np.pad(values, (0, target - values.size))
    elif values.size > target:
        values = values[:target]
    return torch.from_numpy(values).unsqueeze(0)


class RandomAudioClips(Dataset):
    """Random fixed-length crops for optimization, with audio paths only."""

    def __init__(self, audio_paths: list[str], sample_rate: int, n_secs: float):
        self.audio_paths = tuple(Path(path) for path in audio_paths)
        self.sample_rate = int(sample_rate)
        self.n_secs = float(n_secs)
        if not self.audio_paths:
            raise ValueError("training audio manifest is empty")

    @classmethod
    def from_manifest(cls, path: str | Path, sample_rate: int, n_secs: float):
        payload = _read_manifest(path)
        paths = payload.get("audio_paths")
        if not isinstance(paths, list) or not paths:
            raise RuntimeError(f"missing audio_paths: {path}")
        return cls(paths, sample_rate, n_secs)

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.audio_paths[int(index)]
        duration = float(librosa.get_duration(path=path))
        maximum = max(0.0, duration - self.n_secs)
        offset = float(torch.rand(()).item()) * maximum
        return {
            constants.KEY_TRACK: str(path),
            constants.KEY_AUDIO: _load_clip(
                path, self.sample_rate, offset, self.n_secs
            ),
        }


class FixedAudioClips(Dataset):
    """Manifest-defined clips used unchanged at all checkpoint boundaries."""

    def __init__(self, clips: list[dict[str, Any]], sample_rate: int, n_secs: float):
        self.clips = tuple(clips)
        self.sample_rate = int(sample_rate)
        self.n_secs = float(n_secs)
        if not self.clips:
            raise ValueError("validation clip manifest is empty")
        for clip in self.clips:
            if set(clip) != {"audio_path", "offset_sec", "duration_sec"}:
                raise RuntimeError(f"invalid validation clip entry: {clip}")
            if abs(float(clip["duration_sec"]) - self.n_secs) > 1e-9:
                raise RuntimeError(f"validation clip duration mismatch: {clip}")
            if float(clip["offset_sec"]) < 0:
                raise RuntimeError(f"negative validation offset: {clip}")

    @classmethod
    def from_manifest(cls, path: str | Path, sample_rate: int, n_secs: float):
        payload = _read_manifest(path)
        clips = payload.get("clips")
        if not isinstance(clips, list) or not clips:
            raise RuntimeError(f"missing fixed clips: {path}")
        if payload.get("fixed_across_checkpoints") is not True:
            raise RuntimeError(f"validation manifest is not frozen: {path}")
        return cls(clips, sample_rate, n_secs)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self.clips[int(index)]
        path = Path(clip["audio_path"])
        offset = float(clip["offset_sec"])
        return {
            constants.KEY_TRACK: f"{path}@{offset:.6f}",
            constants.KEY_AUDIO: _load_clip(
                path, self.sample_rate, offset, self.n_secs
            ),
        }
