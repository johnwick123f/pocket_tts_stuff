import os

import pytest
import torch

from velvet_tts import VelvetTTS


def test_stream_rejects_batches_before_starting():
    tts = VelvetTTS.__new__(VelvetTTS)
    tts.batch_size = 2
    with pytest.raises(ValueError, match="batch_size=1"):
        tts.stream("Hello.")


@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="Opt-in model checks.")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("codec", ["24k", "48k"])
def test_stream_matches_offline_and_can_close_early(device, codec):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    tts = VelvetTTS(device=device, codec=codec)
    text = "The stars look beautiful tonight."
    tts.codec = "24k"
    expected = tts.generate(text).audio
    tts.codec = codec
    assert tts.stream_sample_rate == 24000
    assert tts._enhancer is None
    chunks = list(tts.stream(text, chunk_frames=4))
    assert len(chunks) > 1
    assert all(
        chunk.ndim == 1 and chunk.device.type == "cpu" and torch.isfinite(chunk).all()
        for chunk in chunks
    )
    actual = torch.cat(chunks)
    assert tts._enhancer is None
    assert actual.shape == expected.shape
    torch.testing.assert_close(
        actual, expected, atol=0.003 if device == "cuda" else 2e-5, rtol=1e-3
    )
    iterator = tts.stream(text)
    next(iterator)
    assert torch.is_grad_enabled()  # The iterator must not leak inference context into its caller.
    iterator.close()
    assert tts.generate("Hello again.").audio.numel() > 0
