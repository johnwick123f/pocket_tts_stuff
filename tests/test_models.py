"""Opt-in checks using downloaded weights: RUN_MODEL_TESTS=1 pytest tests/test_models.py."""

import copy
import os

import pytest
import torch

from velvet_tts import VelvetTTS

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1", reason="Opt-in model checks."
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("model", ["7m", "18m"])
def test_single_and_different_text_batch(device, model, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    engine = VelvetTTS(device=device, batch_size=2, model=model)
    engine.warmup([1, 2])
    texts = ["Hello there.", "The quick brown fox jumps over the lazy dog."]
    # Right-aligned heterogeneous prefixes must match individual unpadded attention.
    from velvet_tts._runtime import Runtime

    with torch.no_grad(), engine._device_context():
        prefix_texts = texts * 2
        runtime = engine._runtime(4)
        runtime.prefill(prefix_texts)
        assert runtime.last_prefill["prefill_rows"] == 2
        state = copy.deepcopy(runtime.state)
        lm = engine.model.flow_lm
        sequence = torch.zeros(4, 1, lm.ldim, device=engine.device)
        empty = torch.empty(4, 0, lm.dim, device=engine.device)
        batched = lm.backbone(lm.input_linear(sequence), empty, sequence, state)
        for row, text in enumerate(prefix_texts):
            reference = Runtime(engine.model, engine.voice, 1, engine.capacity)
            reference.prefill([text])
            single = sequence[row : row + 1]
            expected = lm.backbone(lm.input_linear(single), empty[:1], single, reference.state)
            torch.testing.assert_close(batched[row : row + 1], expected, atol=2e-5, rtol=2e-5)
    result = engine.generate(texts, profile=True)
    assert len(result.waveforms) == 2
    assert all(
        audio.device.type == "cpu"
        and audio.ndim == 1
        and torch.isfinite(audio).all()
        and audio.numel() > 0
        for audio in result.waveforms
    )
    assert result.durations[0] < result.durations[1]
    assert result.timings.stages["lm"] > 0 and result.timings.decoder["transformer"] > 0
    assert result.timings.batches[0]["batch_size"] == 2
    assert len(result.save(tmp_path / "batch")) == 2
    repeat = engine.generate(texts, profile=False)
    assert repeat.timings is None
    for expected, actual in zip(result.waveforms, repeat.waveforms):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    single = engine.generate("Hello there.", batch_size=1, profile=True)
    assert single.audio.numel() > 0
    assert engine.generate([texts[0], texts[1], texts[0]], batch_size=2).waveforms[2].numel() > 0
    long = " ".join([texts[1]] * 6)
    long_result = engine.generate(long, profile=True)
    assert sum(batch["batch_size"] for batch in long_result.timings.batches) >= 2
    assert long_result.durations[0] > single.durations[0]
    for invalid in ["", [], ["ok", ""]]:
        with pytest.raises(ValueError):
            engine.generate(invalid)
    if engine.supports_voice_cloning:
        previous_voice = engine.voice
        previous_runtimes = dict(engine._runtimes)
        with pytest.raises(FileNotFoundError):
            engine.use_voice(tmp_path / "missing.wav")
        assert engine.voice is previous_voice and engine._runtimes == previous_runtimes
        if engine.supports_voice_cloning:
            assert engine.use_voice(tmp_path / "batch" / "audio_001.wav") is engine
            assert not engine._runtimes and engine.voice_setup_seconds > 0
            cloned = engine.generate(texts, profile=True)
            assert len(cloned.waveforms) == 2
            assert all(
                torch.isfinite(audio).all() and audio.numel() > 0 for audio in cloned.waveforms
            )
            assert cloned.timings.metadata["voice"] == str(
                (tmp_path / "batch" / "audio_001.wav").resolve()
            )
            engine.use_voice("alba")
            assert not engine._runtimes and engine.voice_source == "alba"
