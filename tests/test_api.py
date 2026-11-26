import json
import subprocess
import sys

import pytest
import torch

from velvet_tts import GenerationResult, Profile, VelvetTTS


def test_result_and_wav_output(tmp_path):
    result = GenerationResult([torch.zeros(240), torch.ones(480) * 0.1])
    assert result.durations == [0.01, 0.02]
    with pytest.raises(ValueError, match="multiple"):
        _ = result.audio
    with pytest.raises(ValueError, match="directory"):
        result.save(tmp_path / "batch.wav")
    paths = result.save(tmp_path / "batch")
    assert len(paths) == 2 and all(path.exists() for path in paths)


def test_profile_roundtrip(tmp_path):
    profile = Profile(2, [3, 5], {"lm": 1}, {"decoder": 0.5})
    path = profile.save(tmp_path / "profile.json")
    assert json.loads(path.read_text())["aggregate_realtime"] == 4


def test_invalid_codec_before_weight_loading():
    with pytest.raises(ValueError, match="codec must"):
        VelvetTTS(device="cpu", codec="unknown")


def test_invalid_model_before_weight_loading():
    with pytest.raises(ValueError, match="model must"):
        VelvetTTS(device="cpu", model="unknown")


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_batch_size_before_loading(value):
    with pytest.raises(ValueError, match="batch_size"):
        VelvetTTS(batch_size=value)


def test_cpu_import_does_not_load_our_cuda_backend():
    code = "import sys, velvet_tts; assert 'velvet_tts._cuda' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "velvet_tts", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--batch-size" in result.stdout and "--profile-json" in result.stdout


def test_cli_rejects_multiple_outputs_to_one_file():
    from velvet_tts.cli import main

    with pytest.raises(SystemExit) as error:
        main(["--text", "one", "--text", "two", "--output", "batch.wav"])
    assert error.value.code == 2
