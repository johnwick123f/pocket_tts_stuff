"""Opt-in offline integration checks: POCKET_BUNDLE_DIR=models pytest tests/test_bundle.py."""

import os
import socket
from pathlib import Path

import pytest
import torch

from velvet_tts import VelvetTTS

pytestmark = pytest.mark.skipif(
    not os.environ.get("POCKET_BUNDLE_DIR"), reason="Requires exported bundles."
)


@pytest.mark.parametrize("size", ["7m", "18m"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bundle_is_offline(size, device, monkeypatch, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    def blocked(*args, **kwargs):
        raise AssertionError("Offline inference attempted network access")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    bundle = Path(os.environ["POCKET_BUNDLE_DIR"]) / f"velvet-{size}.pt"
    tts = VelvetTTS(device=device, bundle=bundle, codec="48k")
    assert tts.model_name == size and tts.supports_voice_cloning
    tts.warmup(streaming=True)
    assert tts._enhancer is None
    chunks = list(tts.stream("Hello there.", chunk_frames=4))
    assert chunks and tts.stream_sample_rate == 24000 and tts._enhancer is None
    result = tts.generate(["Hello there.", "This is an offline test."], batch_size=2)
    assert result.sample_rate == 48000
    assert all(torch.isfinite(w).all() and w.numel() > 0 for w in result.waveforms)
    # Exercise offline voice switching and a locally supplied voice reference.
    tts.use_voice(next(iter(tts._bundle_data["voices"])))
    reference = result.save(tmp_path / "voice")[0]
    tts.use_voice(reference)
    assert tts.generate("Hello again.").audio.numel() > 0
    with pytest.raises(ValueError, match="Offline"):
        tts.use_voice("https://example.com/voice.wav")
