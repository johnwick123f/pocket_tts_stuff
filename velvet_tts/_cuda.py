"""Native PyTorch attention and reusable CUDA graphs."""

import copy
import math
import types

import torch
from pocket_tts.modules.attention import StreamingMultiheadAttention

from ._runtime import Runtime


def configure_attention(model, capacity):
    for module in model.flow_lm.modules():
        if not isinstance(module, StreamingMultiheadAttention):
            continue
        original = module.forward
        d = module.dim_per_head
        frequencies = torch.exp(
            torch.arange(d // 2, device=model.device, dtype=torch.float32)
            * (-math.log(module.rope.max_period) * 2 / d)
        )
        angles = (
            torch.arange(capacity, device=model.device, dtype=torch.float32)[:, None, None]
            * frequencies
        )
        cosine, sine = angles.cos(), angles.sin()

        def forward(
            attn, query, model_state, attn_mask=None, _original=original, _cosine=cosine, _sine=sine
        ):
            if query.shape[1] != 1 or model_state is None or attn_mask is not None:
                return _original(query, model_state, attn_mask)
            state = attn.get_state(model_state)
            position = state["offset"][:1]
            batch = query.shape[0]
            d = attn.dim_per_head
            projected = attn.in_proj(query)
            q = projected[..., : attn.attn_dim].view(batch, 1, attn.num_heads, d)
            kv = projected[..., attn.attn_dim :].view(batch, 1, 2, attn.num_kv_heads, d)
            k, v = kv.unbind(2)
            cosine = _cosine.index_select(0, position)
            sine = _sine.index_select(0, position)

            def rotate(x):
                pairs = x.view(batch, 1, x.shape[2], d // 2, 2)
                real, imaginary = pairs[..., 0], pairs[..., 1]
                return torch.stack(
                    (real * cosine - imaginary * sine, imaginary * cosine + real * sine), dim=-1
                ).reshape_as(x)

            q, k = rotate(q), rotate(k)
            cache = state["cache"]
            cache[0].index_copy_(1, position, k)
            cache[1].index_copy_(1, position, v)
            active = min(getattr(model, "_hybrid_attention_capacity", capacity), cache.shape[2])
            ka = cache[0, :, :active].transpose(1, 2)
            va = cache[1, :, :active].transpose(1, 2)
            groups = attn.num_heads // attn.num_kv_heads
            grouped_q = q.reshape(batch, attn.num_kv_heads, groups, d)
            scores = torch.matmul(grouped_q, ka.transpose(-1, -2)) * (d**-0.5)
            positions = torch.arange(active, device=query.device)
            valid = (positions[None, :] >= state["pad"][:, None]) & (
                positions[None, :] <= state["offset"][:, None]
            )
            scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
            output = torch.matmul(scores.softmax(dim=-1), va).reshape(batch, attn.num_heads, 1, d)
            return attn.out_proj(output.transpose(1, 2).reshape(batch, 1, attn.attn_dim))

        module.forward = types.MethodType(forward, module)


class CUDARuntime(Runtime):
    graph_steps = 4

    def __init__(self, model, voice, batch_size, capacity):
        super().__init__(model, voice, batch_size, capacity, decoder_frames=0)
        self.graph = None
        self.graphs = {}

    @torch.no_grad()
    def prepare(self):
        if self.graph is not None:
            return
        self._reset(self.state, self.voice)
        self.previous.fill_(float("nan"))
        snapshot = copy.deepcopy(self.state)
        previous = self.previous.clone()
        stream = torch.cuda.Stream(device=self.device)
        prefix = int(self.offsets[0][0].item())
        capacities = [
            n
            for n in (128, 256, 512, 1024, 2048, 4096)
            if prefix + 2 + self.graph_steps < n < self.capacity
        ] + [self.capacity]
        for capacity in capacities:
            for name, values in snapshot.items():
                for key, value in values.items():
                    self.state[name][key].copy_(value)
            self.previous.copy_(previous)
            self.model._hybrid_attention_capacity = capacity
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.step()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            for name, values in snapshot.items():
                for key, value in values.items():
                    self.state[name][key].copy_(value)
            self.previous.copy_(previous)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, ends = [], []
                for _ in range(self.graph_steps):
                    latent, eos = self.step()
                    output.append(latent)
                    ends.append(eos)
                result = torch.cat(output, dim=1)
                eos = torch.cat(ends, dim=1)
            self.graphs[capacity] = graph, result, eos
        self.model._hybrid_attention_capacity = self.capacity
        self.graph, self.output, self.eos = self.graphs[self.capacity]
        for name, values in snapshot.items():
            for key, value in values.items():
                self.state[name][key].copy_(value)
        self.previous.copy_(previous)

    def prefill(self, texts):
        delays, limits = super().prefill(texts)
        self.position = int(self.offsets[0][0].item())
        return delays, limits

    def replay(self):
        capacity = min(n for n in self.graphs if n >= self.position + self.graph_steps)
        self.graph, self.output, self.eos = self.graphs[capacity]
        self.graph.replay()
        self.position += self.graph_steps
        return self.output, self.eos
