"""Granite reference retrieval -> cached MioCodec global -> optional controls."""

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import os
import math
import torch
from . import CODEC, apply, _conditioning, _resolve_path, _WEIGHTS

MODEL = "ibm-granite/granite-embedding-small-english-r2"
INDEX = Path(__file__).with_name("references.pt")


def change_speed(audio, speed=1.0, *, pitch=0.0, sample_rate=24000):
    """Pedalboard speed ratio and independent pitch shift in semitones."""
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    if not math.isfinite(pitch):
        raise ValueError("pitch must be finite (semitones)")
    if speed == 1.0 and pitch == 0.0:
        return audio
    from pedalboard import time_stretch

    if audio.ndim != 1:
        raise ValueError("Expected mono audio shaped (samples,)")
    output = time_stretch(
        audio.detach().float().cpu().numpy()[None],
        samplerate=sample_rate,
        stretch_factor=speed,
        pitch_shift_in_semitones=pitch,
        high_quality=True,
    )
    return torch.from_numpy(output.reshape(-1).copy()).to(audio)


@lru_cache(maxsize=2)
def _encoder(device):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL, device=device)


@lru_cache(maxsize=2)
def _codec(device):
    from miocodec import MioCodecModel

    return MioCodecModel.from_pretrained(CODEC).eval().to(device)


def format_voice_profile(record):
    """Same compact profile as the reference, using fields instead of fragile positions."""
    name = record.get("name") or record.get("cid", "Voice")
    summary = str(record.get("tagline") or record.get("search_text") or "").strip().rstrip(".")
    register = str(record.get("register") or "unspecified register").strip().rstrip(".").lower()
    tone = str(record.get("emotional_range") or "expressive").strip().rstrip(".").lower()
    casting = record.get("casting") or {}
    roles = [v["role"] for v in casting.values() if isinstance(v, dict) and v.get("role")][:3]
    return (
        f"**{name}** ({record.get('gender') or 'unspecified'}, {record.get('age') or 'unspecified'}): "
        f"{summary.capitalize()}, featuring a {register}. "
        f"Its {tone} delivery works best for roles like {', '.join(roles) or 'voice acting'}."
    )


@dataclass(frozen=True)
class VoiceMatch:
    voice_id: str
    description: str
    score: float
    global_embedding: torch.Tensor
    source_gender: str | None = None


@dataclass(frozen=True)
class VoiceResult:
    audio: torch.Tensor
    sample_rate: int
    match: VoiceMatch

    def save(self, path):
        import soundfile as sf

        sf.write(path, self.audio.numpy(), self.sample_rate)


class VoiceLibrary:
    """Search descriptions, edit the selected global, optionally decode real speech.

    library = VoiceLibrary()
    result = library.generate("a vampire voice", emotion="angry", emotion_strength=1.0)
    result.save("vampire.wav")
    library.add("new_reference.wav", "A low, raspy, threatening voice.")
    """

    def __init__(self, index_path=INDEX, device="cpu", *, models_path=None):
        self.path = _resolve_path(index_path)
        self.models_path = _resolve_path(_WEIGHTS if models_path is None else models_path)
        self.device = str(torch.device(device))
        if self.path.exists():
            self.data = torch.load(self.path, map_location="cpu", weights_only=True)
            if (
                self.data.get("version") != 1
                or self.data.get("model") != MODEL
                or self.data.get("codec") != CODEC
            ):
                raise ValueError("Incompatible reference index")
        else:
            self.data = dict(
                version=1,
                model=MODEL,
                codec=CODEC,
                entries=[],
                vectors=torch.empty(0, 0),
                globals=torch.empty(0, 128),
                content=None,
            )

    @classmethod
    def build_from_cache(
        cls, cache_dir=None, *, index_path=INDEX, device="cpu", models_path=None, overwrite=False
    ):
        """One-time build from existing LAION caches; no audio rereads or training."""
        if Path(index_path).exists() and not overwrite:
            raise FileExistsError(
                "Index exists; use VoiceLibrary() or explicitly set overwrite=True"
            )
        root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[1]
        data = torch.load(root / "paired.pt", map_location="cpu", weights_only=True)
        library = cls(index_path, device, models_path=models_path)
        profiles = [format_voice_profile(r) for r in data["records"]]
        vectors = (
            _encoder(library.device)
            .encode(
                profiles,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=True,
            )
            .float()
            .cpu()
        )
        source = torch.load(root / "audio_00.pt", map_location="cpu", weights_only=True)[0]
        library.data.update(
            entries=[
                dict(voice_id=k, description=p, source_gender=cls._gender(r.get("gender")))
                for k, p, r in zip(data["ids"], profiles, data["records"])
            ],
            vectors=vectors,
            globals=data["g"].float().cpu(),
            content=source["content"],
            source="LAION and TTS-AGI, CC BY 4.0; cached best-version audio globals",
        )
        library.save()
        return library

    @staticmethod
    def _gender(value):
        value = str(value).lower()
        return value if value in ("male", "female") else None

    def _encode(self, text):
        """Use the same normalized representation for queries and references."""
        return (
            _encoder(self.device)
            .encode(text, normalize_embeddings=True, convert_to_tensor=True)
            .float()
            .cpu()
        )

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(self.data, temporary)
        os.replace(temporary, self.path)

    @torch.inference_mode()
    def search(self, prompt, top_k=5):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Give a nonempty voice description")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.data["entries"]:
            raise ValueError("Reference library is empty; add references first")
        query = self._encode(prompt)
        scores, indices = (self.data["vectors"] @ query).topk(min(top_k, len(self.data["entries"])))
        return [
            VoiceMatch(
                **self.data["entries"][i],
                score=float(score),
                global_embedding=self.data["globals"][i].clone(),
            )
            for score, i in zip(scores, indices.tolist())
        ]

    @torch.inference_mode()
    def select(
        self, prompt, *, emotion=None, emotion_strength=1.0, gender=None, gender_strength=1.0
    ):
        edits = {}
        if gender is not None:
            if not isinstance(gender, str):
                raise TypeError("gender must be male or female")
            name, level = _conditioning(gender, gender_strength)[0]
            if name not in ("male", "female"):
                raise ValueError("gender must be male or female")
            edits[name] = level
        if emotion is not None:
            if not isinstance(emotion, str):
                raise TypeError("emotion must be an emotion name")
            name, level = _conditioning(emotion, emotion_strength)[0]
            if name in ("male", "female"):
                raise ValueError("Use gender= for male/female controls")
            edits[name] = level
        match = self.search(prompt, top_k=1)[0]
        g = match.global_embedding.to(self.device)
        if edits:
            g = apply(g, edits, source=match.source_gender, models_path=self.models_path)
        return replace(match, global_embedding=g)

    def embedding(self, prompt, **controls):
        """Return only the selected/edited 128-dimensional global embedding."""
        return self.select(prompt, **controls).global_embedding

    @torch.inference_mode()
    def generate(self, prompt, *, content_tokens=None, speed=1.0, pitch=0.0, **controls):
        """Speed >1 is faster, <1 slower; pitch is an independent semitone shift."""
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        if not math.isfinite(pitch):
            raise ValueError("pitch must be finite (semitones)")
        match = self.select(prompt, **controls)
        tokens = content_tokens if content_tokens is not None else self.data["content"]
        if tokens is None:
            raise ValueError("Supply content_tokens or add an audio reference first")
        model = _codec(self.device)
        audio = (
            model.decode(
                content_token_indices=tokens.to(self.device),
                global_embedding=match.global_embedding,
            )
            .float()
            .cpu()
        )
        if not bool(torch.isfinite(audio).all()):
            raise RuntimeError("Decoder produced nonfinite audio")
        audio = change_speed(audio, speed, pitch=pitch, sample_rate=model.config.sample_rate)
        return VoiceResult(audio, model.config.sample_rate, match)

    @torch.inference_mode()
    def add_embedding(self, voice_id, description, global_embedding, *, gender=None, save=True):
        if not isinstance(voice_id, str) or not voice_id.strip():
            raise ValueError("Give a voice_id")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Give a voice description")
        g = global_embedding.detach().float().cpu().reshape(-1)
        if g.numel() != 128 or not bool(torch.isfinite(g).all()):
            raise ValueError("Expected a finite 128-dimensional global")
        vector = self._encode(description)
        entry = dict(voice_id=voice_id, description=description, source_gender=self._gender(gender))
        index = next(
            (i for i, e in enumerate(self.data["entries"]) if e["voice_id"] == voice_id), None
        )
        if index is None:
            self.data["entries"].append(entry)
            self.data["globals"] = torch.cat([self.data["globals"], g[None]])
            self.data["vectors"] = (
                torch.cat([self.data["vectors"], vector[None]])
                if self.data["vectors"].numel()
                else vector[None]
            )
        else:
            self.data["entries"][index] = entry
            self.data["globals"][index] = g
            self.data["vectors"][index] = vector
        if save:
            self.save()
        return voice_id

    @torch.inference_mode()
    def add(self, audio_path, description, *, voice_id=None, gender=None, save=True):
        """Encode a new reference once and append/update it; no retraining."""
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Give a voice description")
        if not Path(audio_path).is_file():
            raise FileNotFoundError(audio_path)
        from miocodec import load_audio

        model = _codec(self.device)
        features = model.encode(
            load_audio(str(audio_path), model.config.sample_rate).to(self.device)
        )
        if self.data["content"] is None:
            self.data["content"] = features.content_token_indices.cpu()
        return self.add_embedding(
            voice_id or Path(audio_path).stem,
            description,
            features.global_embedding,
            gender=gender,
            save=save,
        )
