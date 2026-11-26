import threading
from types import SimpleNamespace

import pytest
import torch

from velvet_tts import VelvetTTS


def test_float_and_pcm_voice_reference(tmp_path):
    import numpy as np
    from scipy.io import wavfile

    from velvet_tts._audio import read_reference

    t = np.arange(8000, dtype=np.float32) / 8000
    signal = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    float_file, pcm_file = tmp_path / "float.wav", tmp_path / "pcm.wav"
    wavfile.write(float_file, 8000, np.stack([signal, signal], axis=1))
    wavfile.write(pcm_file, 8000, (signal * 32767).astype(np.int16))
    first, second = read_reference(float_file), read_reference(pcm_file)
    assert first.shape == second.shape == (1, 24000)
    torch.testing.assert_close(first, second, atol=5e-5, rtol=5e-4)


def test_voice_switch_replaces_state_only_after_success():
    engine = VelvetTTS.__new__(VelvetTTS)
    engine._bundle_data = None
    engine.device = torch.device("cpu")
    engine.capacity = 512
    engine._lock = threading.Lock()
    old_voice = {"layer": {"offset": torch.tensor([3])}}
    engine.voice = old_voice
    engine._runtimes = {2: object()}
    engine.model = SimpleNamespace(
        get_state_for_audio_prompt=lambda *a, **kw: {"layer": {"offset": torch.tensor([510])}}
    )
    with pytest.raises(ValueError, match="capacity"):
        engine.use_voice("alba")
    assert engine.voice is old_voice and 2 in engine._runtimes
    new_voice = {"layer": {"offset": torch.tensor([10])}}
    engine.model.get_state_for_audio_prompt = lambda *a, **kw: new_voice
    assert engine.use_voice("alba") is engine
    assert engine.voice is new_voice and not engine._runtimes


def test_voice_path_resolution(tmp_path):
    file = tmp_path / "reference.wav"
    file.write_bytes(b"placeholder")
    assert VelvetTTS._voice_source(str(file)) == file.resolve()
    assert VelvetTTS._voice_source("alba") == "alba"
    with pytest.raises(FileNotFoundError):
        VelvetTTS._voice_source(tmp_path / "missing.wav")
