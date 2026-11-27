import os
from typing import List, Optional, Union
import torch
from huggingface_hub import hf_hub_download

from .engine import VelvetTTS
from .text_normalizer.model_cpp import SpeechTextNormalizer
from . import controls as control

class VelvetTTSPipeline:
    REPO_ID = "YaTharThShaRma999/tts_test"

    HF_FILES = {
        "small": "tts_backbone/velvet_small.pt",
        "mini": "tts_backbone/velvet_mini.pt",
        "references": "voice_generator/references.pt",
        "transforms": "voice_generator/transforms.pt",
        "normalizer": "text_normalizer/text_normalizer.gguf",
    }

    def __init__(
        self,
        load_mode: str = "all",  # Options: "all", "tts", "normalizer", "voice_gen"
        bundle: str = "small",   # Options: "small", "mini"
        codec: str = "48k",      # Options: "48k", "24k" (also accepts "48khz"/"24khz")
        device: Optional[str] = None,
        batch_size: int = 4,
        flow_gain: float = 1.1,
        cpu_threads: int = 4,
        cache_dir: str = "./velvet_models",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        codec_clean = codec.lower().replace("hz", "")
        if codec_clean not in ("48k", "24k"):
            raise ValueError("Codec must be '48k'/'48khz' or '24k'/'24khz'.")

        valid_modes = {"all", "tts", "normalizer", "voice_gen"}
        if load_mode not in valid_modes:
            raise ValueError(f"load_mode must be one of {valid_modes}")

        if bundle not in ("small", "mini"):
            raise ValueError("Bundle must be 'small' or 'mini'.")

        self.normalizer: Optional[SpeechTextNormalizer] = None
        self.voices: Optional[control.VoiceLibrary] = None

        # 1. Load Text Normalizer (optional)
        if load_mode in ("all", "normalizer"):
            norm_path = self._download_asset(self.HF_FILES["normalizer"])
            self.normalizer = SpeechTextNormalizer(norm_path)

        # 2. Load Voice Generator (optional)
        if load_mode in ("all", "voice_gen"):
            ref_path = self._download_asset(self.HF_FILES["references"])
            trans_path = self._download_asset(self.HF_FILES["transforms"])
            self.voices = control.VoiceLibrary(
                device=self.device,
                index_path=ref_path,
                models_path=trans_path,
            )

        # 3. Always Load TTS Backbone
        tts_model_path = self._download_asset(self.HF_FILES[bundle])
        self.tts = VelvetTTS(
            bundle=tts_model_path,
            device=self.device,
            batch_size=batch_size,
            flow_gain=flow_gain,
            codec=codec_clean,
            cpu_threads=cpu_threads,
        )

    def _download_asset(self, repo_filepath: str) -> str:
        """Downloads a specific file from the HuggingFace repo and returns its local path."""
        return hf_hub_download(
            repo_id=self.REPO_ID,
            filename=repo_filepath,
            local_dir=self.cache_dir,
        )

    def normalize(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if self.normalizer is None:
            raise RuntimeError("Normalizer was not loaded. Initialize with load_mode='all' or 'normalizer'.")

        if isinstance(text, str):
            return self.normalizer.normalize(text)

        if isinstance(text, list):
            if "cuda" in self.device:
                return self.normalizer.normalize_batch(text)
            return [self.normalizer.normalize(t) for t in text]

        raise TypeError("Input `text` must be a `str` or `List[str]`.")

    def generate_voice(
        self,
        description: str,
        save_path: str = "text_voice.wav",
        emotion: Optional[str] = None,
        emotion_strength: Optional[float] = None,
        gender: Optional[str] = None,
        gender_strength: Optional[float] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
    ) -> str:
        if self.voices is None:
            raise RuntimeError("VoiceLibrary was not loaded. Initialize with load_mode='all' or 'voice_gen'.")

        # Only forward controls that were explicitly provided
        optional_kwargs = {
            k: v
            for k, v in {
                "emotion": emotion,
                "emotion_strength": emotion_strength,
                "gender": gender,
                "gender_strength": gender_strength,
                "speed": speed,
                "pitch": pitch,
            }.items()
            if v is not None
        }

        with torch.no_grad():
            result = self.voices.generate(description, **optional_kwargs)

        result.save(save_path)
        return save_path

    def gen(
        self,
        text: Union[str, List[str]],
        voice: str,
        output_path: str = "output",
        profile: bool = False,
        auto_normalize: bool = False,
    ):
        if auto_normalize:
            if self.normalizer is None:
                raise RuntimeError("Cannot auto_normalize: Normalizer was not loaded.")
            text = self.normalize(text)

        texts = [text] if isinstance(text, str) else text

        self.tts.use_voice(voice)
        result = self.tts.generate(texts, profile=profile)
        result.save(output_path)
        return result

    @property
    def stream_sample_rate(self) -> int:
        return self.tts.stream_sample_rate

    def stream(
        self,
        text: str,
        voice: Optional[str] = None,
        chunk_frames: int = 4,
        auto_normalize: bool = False,
    ):
        if not isinstance(text, str):
            raise TypeError("Input `text` for streaming must be a `str`.")
        if not isinstance(chunk_frames, int) or isinstance(chunk_frames, bool):
            raise TypeError("`chunk_frames` must be an `int`.")

        if auto_normalize:
            if self.normalizer is None:
                raise RuntimeError("Cannot auto_normalize: Normalizer was not loaded.")
            text = self.normalize(text)

        if voice is not None:
            self.tts.use_voice(voice)

        yield from self.tts.stream(text, chunk_frames=chunk_frames)
