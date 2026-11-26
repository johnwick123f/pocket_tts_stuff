"""Create/load self-contained offline model files: python -m velvet_tts.bundle."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def export_bundle(model, path):
    """One-time online export; inference from the resulting file requires no network."""
    from huggingface_hub import HfApi, hf_hub_download
    from pocket_tts import TTSModel

    if model not in ("7m", "18m"):
        raise ValueError("model must be '7m' or '18m'.")
    language = f"english_{model}_2026-10"
    tts = TTSModel.load_model(language=language)
    if not tts.has_voice_cloning:
        raise RuntimeError("Export requires the voice-cloning-enabled checkpoint.")
    kind, tokenizer = tts.flow_lm.conditioner.tokenizer.serialize()
    config = tts.config.model_dump(mode="json")
    config["weights_path"] = config["weights_path_without_voice_cloning"] = None
    config["flow_lm"]["weights_path"] = config["mimi"]["weights_path"] = None
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = "bundled-tokenizer"
    repo = "kyutai/pocket-tts-without-voice-cloning"
    revision = "a7056f505a4a8ca72d8b5fb1440071d6f1a35739"
    prefix = f"languages/{language}/embeddings/"
    files = [
        f
        for f in HfApi().list_repo_files(repo, revision=revision)
        if f.startswith(prefix) and f.endswith(".safetensors")
    ]
    voices = {}
    for filename in files:
        local = hf_hub_download(repo, filename, revision=revision)
        voices[Path(filename).stem] = tts.get_state_for_audio_prompt(Path(local))
    if "alba" not in voices:
        raise RuntimeError("Default voice is missing from the source checkpoint.")
    lava_repo = "YatharthS/LavaSR"
    lava_weights = hf_hub_download(lava_repo, "enhancer_v2/pytorch_model.bin")
    lava_config = hf_hub_download(lava_repo, "enhancer_v2/config.yaml")
    data = {
        "format": "pocket-hybrid",
        "format_version": 1,
        "model": model,
        "config": config,
        "weights": tts.state_dict(),
        "tokenizer_kind": kind,
        "tokenizer": tokenizer,
        "voices": voices,
        "lava": {
            "config": Path(lava_config).read_text(),
            "weights": torch.load(lava_weights, map_location="cpu", weights_only=True),
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return path


def read_bundle(path):
    data = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if data.get("format") != "pocket-hybrid" or data.get("format_version") != 1:
        raise ValueError("Unsupported VelvetTTS bundle.")
    return data


def build_model(data):
    from pocket_tts import TTSModel
    from pocket_tts.models.mimi import build_mimi
    from pocket_tts.models.tts_model import stamp_state_names
    from pocket_tts.utils.config import Config

    with TemporaryDirectory(prefix="pocket-tokenizer-") as folder:
        tokenizer = Path(folder) / (
            "tokenizer.json" if data["tokenizer_kind"] == "tokenizers" else "tokenizer.model"
        )
        tokenizer.write_bytes(data["tokenizer"])
        config = Config.model_validate(data["config"])
        config.flow_lm.lookup_table.tokenizer_path = str(tokenizer)
        model = TTSModel._from_pydantic_config(
            config, config.default_temperature, 1, None, config.default_eos_threshold, origin=None
        )
    model.flow_lm.speaker_proj_weight = torch.nn.Parameter(
        torch.zeros(
            config.flow_lm.transformer.d_model,
            config.mimi.inner_dim or config.mimi.seanet.dimension,
        )
    )
    model.mimi = build_mimi(config.mimi)
    model.load_state_dict(data.pop("weights"), strict=True)
    stamp_state_names(model)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    for size in ("7m", "18m"):
        path = export_bundle(size, args.output_dir / f"velvet-{size}.pt")
        print(f"{path}: {path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
