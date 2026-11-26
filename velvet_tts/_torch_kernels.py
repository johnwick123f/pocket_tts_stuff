"""Inference-only algebraic optimizations using existing PyTorch CUDA kernels."""

import types

import torch
from pocket_tts.modules.mlp import LayerNorm
from torch.nn import functional as F


def optimize_torch(model):
    """Fuse native layer norm and pack parallel sampler conditioning projections."""
    net = model.flow_lm.flow_net
    if net.num_time_conds != 0:
        return
    for module in net.modules():
        if isinstance(module, LayerNorm):

            def normalized(self, x):
                return F.layer_norm(
                    x,
                    (x.shape[-1],),
                    getattr(self, "weight", None),
                    getattr(self, "bias", None),
                    self.eps,
                )

            module.forward = types.MethodType(normalized, module)
    layers = [block.adaLN_modulation[1] for block in net.res_blocks]
    layers.append(net.final_layer.adaLN_modulation[1])
    net.register_buffer("_packed_weight", torch.cat([layer.weight.detach() for layer in layers]))
    net.register_buffer("_packed_bias", torch.cat([layer.bias.detach() for layer in layers]))

    def forward(self, condition, noise):
        x = self.input_proj(noise)
        y = self.cond_embed(condition)
        modulation = F.linear(F.silu(y), self._packed_weight, self._packed_bias)
        width = self.model_channels
        for index, block in enumerate(self.res_blocks):
            start = index * 3 * width
            shift, scale, gate = modulation[..., start : start + 3 * width].chunk(3, dim=-1)
            hidden = torch.addcmul(shift, block.in_ln(x), 1 + scale)
            x = torch.addcmul(x, gate, block.mlp(hidden))
        shift, scale = modulation[..., len(self.res_blocks) * 3 * width :].chunk(2, dim=-1)
        hidden = torch.addcmul(shift, self.final_layer.norm_final(x), 1 + scale)
        return self.final_layer.linear(hidden)

    net.forward = types.MethodType(forward, net)
