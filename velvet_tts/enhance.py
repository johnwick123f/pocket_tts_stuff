"""LavaSR v2 through Vocos; no LavaSR package or denoiser.

Head and frequency merge adapted from Apache-2.0 ysharma3501/LavaSR.
"""

import types
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn


def _head_forward(self, features):
    magnitude, phase = self.out(features).transpose(1, 2).chunk(2, dim=1)
    spectrum = magnitude.exp().clamp(max=1000) * (phase.cos() + 1j * phase.sin())
    return self.istft(spectrum)


class LavaSR(nn.Module):
    """Enhance mono [T] or batched [B,T] tensors to 48 kHz; load once and reuse.

    Uses the v2 enhancement weights only. Denoising is deliberately omitted.
    Default 12 kHz merge preserves the original 24 kHz signal's lower frequencies.
    """

    sample_rate = 48000

    def __init__(self, device="cpu", cutoff=12000, model_dir=None, bundle=None):
        super().__init__()
        from vocos import Vocos

        if not 0 < cutoff < 24000:
            raise ValueError("cutoff must be between 0 and 24000 Hz.")
        self.cutoff = cutoff

        def path(filename):
            return (
                str(Path(model_dir) / filename)
                if model_dir
                else hf_hub_download("YatharthS/LavaSR", f"enhancer_v2/{filename}")
            )

        if bundle is not None:
            from tempfile import TemporaryDirectory

            with TemporaryDirectory(prefix="pocket-lava-") as folder:
                config = Path(folder) / "config.yaml"
                config.write_text(bundle["config"])
                self.model = Vocos.from_hparams(str(config))
            weights = bundle["weights"]
        else:
            self.model = Vocos.from_hparams(path("config.yaml"))
            weights = torch.load(path("pytorch_model.bin"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.head.forward = types.MethodType(_head_forward, self.model.head)
        self.to(device).eval()
        self._resamplers, self._masks = {}, {}

    @torch.no_grad()
    def forward(self, audio, sample_rate=24000):
        from torchaudio.transforms import Resample

        if audio.ndim not in (1, 2) or audio.shape[-1] < 2:
            raise ValueError("Expected mono [T] or batched [B,T] audio.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        single = audio.ndim == 1
        device = next(self.model.parameters()).device
        x = audio.to(device=device, dtype=torch.float32).reshape(-1, audio.shape[-1])
        if sample_rate != 48000:
            key = (sample_rate, device)
            if key not in self._resamplers:
                self._resamplers[key] = Resample(sample_rate, 48000, dtype=torch.float32).to(device)
            x = self._resamplers[key](x)
        y = self.model.head(self.model.backbone(self.model.feature_extractor(x)))
        length = min(x.shape[-1], y.shape[-1])
        a, b = torch.fft.rfft(y[:, :length]), torch.fft.rfft(x[:, :length])
        key = (length, min(self.cutoff, sample_rate / 2), device)
        if key not in self._masks:
            if len(self._masks) >= 8:
                self._masks.pop(next(iter(self._masks)))
            bins = a.shape[-1]
            center = int(key[1] / 24000 * bins)
            start, end = max(0, center - 512), min(bins, center + 512)
            t = ((torch.linspace(-1, 1, 1024, device=device) + 1) / 2)[: end - start]
            mask = torch.ones(bins, device=device, dtype=torch.complex64)
            mask[:start], mask[start:end] = 0, (3 * t.square() - 2 * t.pow(3)).to(torch.complex64)
            self._masks[key] = mask
        b += (a - b) * self._masks[key]
        result = torch.fft.irfft(b, n=length)
        return result[0] if single else result
