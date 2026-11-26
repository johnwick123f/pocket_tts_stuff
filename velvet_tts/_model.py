"""Base 7M/18M checkpoints, selectable codec, and baked-in flow conditioning gain."""

import math

import torch
from pocket_tts import TTSModel
from pocket_tts.models.tts_model import stamp_state_names


def load_model(codec="24k", flow_gain=1.0, model="18m", bundle_data=None):
    if model not in ("7m", "18m"):
        raise ValueError("model must be '7m' or '18m'.")
    if codec not in ("24k", "48k", "small"):
        raise ValueError("codec must be '24k', '48k' or 'small'.")
    if not math.isfinite(flow_gain) or flow_gain <= 0:
        raise ValueError("flow_gain must be finite and positive.")
    if bundle_data is not None:
        if codec == "small":
            raise ValueError(
                "Offline bundles support codec='24k' or '48k'; the 2M codec is not bundled."
            )
        from .bundle import build_model

        model = build_model(bundle_data)
    else:
        model = TTSModel.load_model(language=f"english_{model}_2026-10")
    if codec == "small":
        donor = TTSModel.load_model(language="english_2m_2026-10")
        for name in ("quantizer", "upsample", "decoder_transformer", "decoder"):
            setattr(model.mimi, name, getattr(donor.mimi, name))
        model.config.mimi.decoder_seanet = donor.config.mimi.decoder_seanet
        model.config.mimi.decoder_transformer = donor.config.mimi.decoder_transformer
    with torch.no_grad():
        model.flow_lm.flow_net.cond_embed.weight.mul_(flow_gain)
    stamp_state_names(model)
    return model.eval()
