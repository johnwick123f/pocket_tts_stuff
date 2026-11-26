"""Stable return types shared by the CLI and Python API."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


@dataclass
class Profile:
    """Seconds of wall time; nested decoder measurements are not additive to parents."""

    total_seconds: float
    audio_seconds: list[float]
    stages: dict[str, float] = field(default_factory=dict)
    decoder: dict[str, float] = field(default_factory=dict)
    batches: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def aggregate_realtime(self) -> float:
        return sum(self.audio_seconds) / self.total_seconds if self.total_seconds else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "aggregate_realtime": self.aggregate_realtime}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@dataclass
class GenerationResult:
    """Unpadded CPU float32 waveforms, in the same order as input texts."""

    waveforms: list[torch.Tensor]
    sample_rate: int = 24000
    timings: Profile | None = None

    @property
    def audio(self) -> torch.Tensor:
        """Convenience for a single result; use waveforms for a batch."""
        if len(self.waveforms) != 1:
            raise ValueError("This result has multiple waveforms; use result.waveforms.")
        return self.waveforms[0]

    @property
    def durations(self) -> list[float]:
        return [audio.numel() / self.sample_rate for audio in self.waveforms]

    def save(self, path: str | Path) -> list[Path]:
        """Save one WAV to a .wav path, or multiple WAVs to a directory."""
        from scipy.io import wavfile

        path = Path(path)
        if path.suffix.lower() == ".wav":
            if len(self.waveforms) != 1:
                raise ValueError("Use an output directory when saving multiple waveforms.")
            paths = [path]
        else:
            paths = [path / f"audio_{i:03d}.wav" for i in range(len(self.waveforms))]
        for audio, output in zip(self.waveforms, paths):
            output.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(output, self.sample_rate, audio.detach().numpy())
        return paths
