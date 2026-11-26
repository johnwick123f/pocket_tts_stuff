"""Float32 local attention for one-pass CUDA codec decoding; no compilation."""

import types

import torch
from torch.nn import functional as F


def local_attention(q, k, v, context, block=32):
    b, h, t, d = q.shape
    chunks = (t + block - 1) // block
    right = chunks * block - t
    q = F.pad(q, (0, 0, 0, right)).reshape(b * h, chunks, block, d)

    def windows(x):
        x = F.pad(x, (0, 0, context - 1, right))
        return (
            x.unfold(2, block + context - 1, block)
            .transpose(-1, -2)
            .reshape(b * h, chunks, block + context - 1, d)
        )

    qi = torch.arange(block, device=q.device)[:, None] + context - 1
    ki = torch.arange(block + context - 1, device=q.device)[None, :]
    causal = ((qi - ki >= 0) & (qi - ki < context))[None]
    valid = torch.arange(chunks, device=q.device)[:, None, None] * block + ki - context + 1 >= 0
    # Treat windows as independent SDPA heads, avoiding the dense T-by-T matrix.
    output = F.scaled_dot_product_attention(q, windows(k), windows(v), causal & valid)
    return output.reshape(b, h, chunks * block, d)[:, :, :t]


def configure_codec(model):
    for layer in model.mimi.decoder_transformer.transformer.layers:
        attention = layer.self_attn
        if attention.context is None or attention.num_heads != attention.num_kv_heads:
            continue
        original = attention.forward

        def forward(self, query, model_state, attn_mask=None, _original=original):
            if (
                self.training
                or torch.is_grad_enabled()
                or attn_mask is not None
                or not getattr(model, "_offline_codec", False)
            ):
                return _original(query, model_state, attn_mask)
            b, t, _ = query.shape
            h, d = self.num_heads, self.dim_per_head
            projected = self.in_proj(query).view(b, t, 3, h, d)
            q, k, v = projected.unbind(2)
            q, k = self.rope(q, k, offset=0)
            output = local_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), self.context
            )
            return self.out_proj(output.transpose(1, 2).reshape(b, t, h * d))

        attention.forward = types.MethodType(forward, attention)
