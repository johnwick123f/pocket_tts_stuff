"""CPU attention caching and small-kernel tracing. No CUDA dependencies."""

import math
import types

import torch
from pocket_tts.modules.attention import StreamingMultiheadAttention
from pocket_tts.modules.transformer import StreamingTransformerLayer
from torch.nn import functional as F


class FeedForward(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.norm = layer.norm2
        self.linear1 = layer.linear1
        self.linear2 = layer.linear2
        self.scale = layer.layer_scale_2

    def forward(self, x):
        return x + self.scale(self.linear2(F.gelu(self.linear1(self.norm(x)), approximate="tanh")))


def optimize_cpu(model):
    tables, masks = {}, {}
    for module in model.modules():
        if not isinstance(module, StreamingMultiheadAttention):
            continue
        original = module.forward

        def forward(attn, query, model_state, attn_mask=None, _original=original):
            if model_state is None or attn_mask is not None:
                return _original(query, model_state, attn_mask)
            state = attn.get_state(model_state)
            offset = int(state["offset"][0].item())
            b, t, _ = query.shape
            d = attn.dim_per_head
            projected = attn.in_proj(query)
            q = projected[..., : attn.attn_dim].view(b, t, attn.num_heads, d)
            kv = projected[..., attn.attn_dim :].view(b, t, 2, attn.num_kv_heads, d)
            k, v = kv.unbind(2)
            key = (d, attn.rope.max_period)
            if key not in tables or tables[key][0].shape[0] < offset + t:
                n = max(4096, 1 << (offset + t - 1).bit_length())
                frequencies = torch.exp(
                    torch.arange(d // 2, dtype=torch.float32)
                    * (-math.log(attn.rope.max_period) * 2 / d)
                )
                angles = torch.arange(n, dtype=torch.float32)[:, None, None] * frequencies
                tables[key] = angles.cos(), angles.sin()
            cosine, sine = (value[offset : offset + t] for value in tables[key])

            def rotate(x):
                pairs = x.view(b, t, x.shape[2], d // 2, 2)
                real, imaginary = pairs[..., 0], pairs[..., 1]
                return torch.stack(
                    (real * cosine - imaginary * sine, imaginary * cosine + real * sine), dim=-1
                ).reshape_as(x)

            q, k = rotate(q), rotate(k)
            cache = state["cache"]
            cache[0, :, offset : offset + t].copy_(k)
            cache[1, :, offset : offset + t].copy_(v)
            start = 0 if attn.context is None else max(0, offset - attn.context + 1)
            ka = cache[0, :, start : offset + t].transpose(1, 2)
            va = cache[1, :, start : offset + t].transpose(1, 2)
            if attn.num_kv_heads != attn.num_heads:
                repeats = attn.num_heads // attn.num_kv_heads
                ka = ka.repeat_interleave(repeats, dim=1)
                va = va.repeat_interleave(repeats, dim=1)
            if t > 1:
                mask_key = (t, offset - start, attn.context)
                if mask_key not in masks:
                    delta = (offset - start + torch.arange(t))[:, None] - torch.arange(
                        offset - start + t
                    )[None]
                    mask = delta >= 0
                    if attn.context is not None:
                        mask &= delta < attn.context
                    masks[mask_key] = mask
                attn_mask = masks[mask_key]
            if bool((state["pad"] > 0).any()):
                valid = torch.arange(start, offset + t)[None, :] >= state["pad"][:, None]
                valid = valid[:, None, None, :]
                attn_mask = valid if attn_mask is None else valid & attn_mask
            result = F.scaled_dot_product_attention(
                q.transpose(1, 2), ka, va, attn_mask, dropout_p=0.0
            )
            return attn.out_proj(result.transpose(1, 2).reshape(b, t, attn.attn_dim))

        module.forward = types.MethodType(forward, module)

    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, StreamingTransformerLayer):
                kernel = torch.jit.freeze(
                    torch.jit.trace(
                        FeedForward(layer).eval(),
                        torch.zeros(1, 1, layer.norm2.normalized_shape[0]),
                    )
                )

                def feedforward(self, x, _kernel=kernel):
                    return _kernel(x)

                layer._ff_block = types.MethodType(feedforward, layer)
        net = model.flow_lm.flow_net
        condition = torch.zeros(1, net.cond_embed.in_features)
        noise = torch.zeros(1, model.flow_lm.ldim)
        model.flow_lm.flow_net = torch.jit.freeze(torch.jit.trace(net.eval(), (condition, noise)))
