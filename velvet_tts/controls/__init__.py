
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
import math
import os
import re
import torch

CODEC = "Aratako/MioCodec-25Hz-24kHz"
AVAILABLE = ("calm", "happy", "sad", "angry", "fearful", "surprise", "disgust", "male", "female")
__all__ = ["apply", "Control", "AVAILABLE", "CODEC", "VoiceLibrary"]
_WEIGHTS = Path(__file__).with_name("models.pt")


def __getattr__(name):
    if name == "VoiceLibrary":
        from .search import VoiceLibrary

        return VoiceLibrary
    raise AttributeError(name)


def _resolve_path(path):
    """Resolve user paths once; defaults are relative to this package, never cwd."""
    return Path(path).expanduser().resolve()


@lru_cache(maxsize=4)
def _store(models_path=_WEIGHTS):
    data = torch.load(models_path, map_location="cpu", weights_only=True)
    if data.get("format_version") != 1 or data.get("codec") != CODEC:
        raise ValueError("Unsupported control model format/codec")
    return data


def _save_section(name, value):
    data = dict(_store())
    data[name] = value
    temporary = _WEIGHTS.with_suffix(".pt.tmp")
    torch.save(data, temporary)
    os.replace(temporary, _WEIGHTS)
    _store.cache_clear()
    _default.cache_clear()


def _to_device(value, device):
    """Move nested checkpoint tensors without changing metadata."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def gender_network():
    return torch.nn.Sequential(
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 1),
    )


class GenderTransform:
    def __init__(self, device="cpu", *, state=None):
        self.d = _to_device(_store()["gender"] if state is None else state, device)
        self.male = (
            GenderTransform(device, state=self.d["male_model"]) if "male_model" in self.d else None
        )
        with torch.random.fork_rng(devices=[]):
            self.classifier = gender_network()
        self.classifier.to(device).eval().requires_grad_(False)
        self.classifier.load_state_dict(self.d["calibrator"])

    def convert(self, u, source, target, strength):
        if target == 0 and self.male is not None:
            return self.male.convert(u, source, target, strength)
        d = self.d
        p = d["protected"]
        f = d["free"]
        feature = u @ p
        condition = torch.cat([feature, torch.ones(*feature.shape[:-1], 1, device=u.device)], -1)
        old = u @ f
        residual = old - condition @ d["means"][source]
        m = d["maps"][source]
        desired = condition @ d["means"][target] + residual + (residual @ m["right"]) @ m["left"].T
        coord = old + strength * d["gains"][source] * (desired - old)
        fixed = feature @ p.T
        radius = (128 - fixed.square().sum(-1, keepdim=True)).clamp_min(0).sqrt()
        coord = coord / coord.norm(dim=-1, keepdim=True).clamp_min(1e-6) * radius
        return fixed + coord @ f.T

    def apply(self, g, target, strength, source=None):
        if not 0 <= strength <= 1:
            raise ValueError("Gender strength must be in [0,1]")
        if target not in ["male", "female"] or source not in [None, "male", "female"]:
            raise ValueError("Use male/female voice labels")
        if target == "male" and self.male is not None:
            return self.male.apply(g, target, strength, source)
        d = self.d
        u = (g - d["beta"]) / d["gamma"]
        u = (u - u.mean()) / u.std(unbiased=False).clamp_min(1e-6)
        before = self.classifier((u - d["um"]) / d["us"]).sigmoid().item()
        source = int(source == "female") if source else int(before > 0.5)
        target = int(target == "female")
        if strength == 0 or source == target:
            return g, dict(female_before=round(before, 3), female_after=round(before, 3))
        out = self.convert(u, source, target, strength)
        after = self.classifier((out - d["um"]) / d["us"]).sigmoid().item()
        return out * d["gamma"] + d["beta"], dict(
            female_before=round(before, 3), female_after=round(after, 3)
        )


class PairedEmotion:
    words = r"\b(crying|cry|sad|sobbing|sob|tearful|sorrowful|grieving|grief|weeping|weep|heartbroken)\b"

    def __init__(self, device, *, state=None):
        store = _store() if state is None else state
        self.d = _to_device(store["pairs"], device)
        self.models = {"sad": self.d}
        if "emotions" in store:
            self.models.update(_to_device(store["emotions"]["models"], device))

    def identity(self, prompt):
        if not re.search(self.words, prompt, re.I):
            return prompt
        out = re.sub(self.words, "", prompt, flags=re.I)
        out = re.sub(
            r"\b(vulnerable|breathy|trembling|strained|choked[- ]up)\b", "", out, flags=re.I
        )
        return re.sub(r"[,\s]+", " ", out).strip() or "voice"

    def apply(self, g, prompt, strength, emotion=None):
        if emotion is None:
            if not re.search(self.words, prompt, re.I):
                return g
            emotion = "sad"
        if emotion not in self.models:
            raise ValueError(f"Emotion {emotion} not fitted; run controls.py --all-emotions")
        d = self.models[emotion]
        x = (g - d["beta"]) / d["gamma"]
        z = torch.cat([(x - d["center"]) @ d["basis"], torch.ones(1, device=g.device)])
        x = x + strength * (z @ d["weights"])
        x = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-5)
        return x * d["gamma"] + d["beta"]


def _conditioning(conditioning, strength):
    aliases = {
        "anger": "angry",
        "sadness": "sad",
        "happiness": "happy",
        "fear": "fearful",
        "surprised": "surprise",
        "disgusted": "disgust",
    }
    if isinstance(conditioning, str):
        items = [(conditioning, strength)]
    elif isinstance(conditioning, Mapping):
        if strength != 1.0:
            raise ValueError("Put individual strengths in the conditioning dict")
        items = list(conditioning.items())
    else:
        raise TypeError("conditioning must be a name or {name: strength} dict")
    result = {}
    for name, level in items:
        if not isinstance(name, str):
            raise TypeError("Control names must be strings")
        name = name.strip().lower()
        name = aliases.get(name, name)
        if name not in AVAILABLE:
            raise ValueError(f"Unknown control {name!r}; choose from {AVAILABLE}")
        if name in result:
            raise ValueError(f"Duplicate control {name!r}")
        level = float(level)
        if not math.isfinite(level) or level < 0 or (name in ("male", "female") and level > 1):
            raise ValueError(
                "Emotion strengths must be finite and >=0; gender strengths must be in [0,1]"
            )
        result[name] = level
    if "male" in result and "female" in result:
        raise ValueError("Choose one target: male or female")
    return sorted(result.items(), key=lambda item: item[0] not in ("male", "female"))


class Control:
    """Reusable inference object; optional alternative to the cached apply() helper."""

    def __init__(self, device="cpu", *, models_path=None):
        self.models_path = _resolve_path(_WEIGHTS if models_path is None else models_path)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        state = _store(self.models_path)
        self.emotions = PairedEmotion(self.device, state=state)
        self.gender = GenderTransform(self.device, state=state["gender"])

    @torch.inference_mode()
    def apply(self, global_embedding, conditioning, strength=1.0, *, source=None):
        if not isinstance(global_embedding, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")
        g = global_embedding
        if not g.is_floating_point() or g.ndim < 1 or g.shape[-1] != 128:
            raise ValueError("Expected a floating tensor shaped (...,128)")
        if g.device != self.device:
            raise ValueError(f"Input is on {g.device}; this Control is on {self.device}")
        if not bool(torch.isfinite(g).all()):
            raise ValueError("Embedding contains NaN/Inf")
        if source is not None:
            if not isinstance(source, str):
                raise TypeError("source must be male, female, or None")
            source = source.strip().lower()
            if source not in ("male", "female"):
                raise ValueError("source must be male or female")
        steps = _conditioning(conditioning, strength)
        if g.numel() == 0 or not any(level for _, level in steps):
            return g.clone()
        rows = []
        with torch.autocast(device_type=g.device.type, enabled=False):
            for row in g.reshape(-1, 128).float():
                out = row
                for name, level in steps:
                    if level == 0:
                        continue
                    if name in ("male", "female"):
                        out = self.gender.apply(out, name, level, source)[0]
                    else:
                        out = self.emotions.apply(out, "", level, name)
                rows.append(out)
        return torch.stack(rows).reshape(g.shape).to(dtype=g.dtype)

    __call__ = apply


@lru_cache(maxsize=4)
def _default(device, models_path):
    return Control(device, models_path=models_path)


def apply(global_embedding, conditioning, strength=1.0, *, source=None, models_path=None):
    """Apply emotion or voice-gender controls; automatically caches per input device."""
    if not isinstance(global_embedding, torch.Tensor):
        raise TypeError("Expected a torch.Tensor")
    path = _resolve_path(_WEIGHTS if models_path is None else models_path)
    return _default(str(global_embedding.device), path).apply(
        global_embedding, conditioning, strength, source=source
    )
