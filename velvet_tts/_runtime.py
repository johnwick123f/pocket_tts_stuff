"""Shared batch scheduling and per-row EOS handling, independent of the device."""

import math
from collections import defaultdict

import torch
from pocket_tts.models.text_chunking import prepare_text_prompt
from pocket_tts.modules.attention import StreamingMultiheadAttention
from pocket_tts.modules.stateful_module import StatefulModule, init_states


def codec_state(mimi, batch_size, frames):
    """Allocate only decoder state; encoder caches are not used during synthesis."""
    return {
        name: module.init_state(batch_size, frames)
        for name, module in mimi.named_modules()
        if name.startswith(("upsample", "decoder_transformer", "decoder"))
        and isinstance(module, StatefulModule)
    }


class Runtime:
    graph_steps = 1

    def __init__(self, model, voice, batch_size, capacity, decoder_frames=8):
        self.model = model
        self.voice = voice
        self.batch_size = batch_size
        self.capacity = capacity
        self.device = model.device
        self.decoder_frames = decoder_frames
        self.enhancer = None
        self.state = init_states(model.flow_lm, batch_size, capacity)
        self.previous = torch.empty(batch_size, 1, model.flow_lm.ldim, device=self.device)
        self.empty_embeddings = torch.empty(batch_size, 0, model.flow_lm.dim, device=self.device)
        self.offsets = [state["offset"] for state in self.state.values() if "offset" in state]
        self.attention = {
            name: module
            for name, module in model.flow_lm.named_modules()
            if isinstance(module, StreamingMultiheadAttention)
        }

    def _reset(self, state, voice):
        for name, values in state.items():
            for key, value in values.items():
                value.zero_()
                source = voice.get(name, {}).get(key)
                if source is not None:
                    if key == "cache":
                        value[:, :, : source.shape[2]].copy_(torch.nan_to_num(source))
                    else:
                        value.copy_(source)

    def prepare(self):
        """Device-specific lazy initialization. CPU has no graph setup."""

    def _consume_prefill_noise(self, batch_size):
        lm = self.model.flow_lm
        noise = torch.empty(batch_size, lm.ldim, device=self.device)
        std = self.model.temp**0.5
        clamp = self.model.noise_clamp
        if clamp is None:
            torch.nn.init.normal_(noise, mean=0.0, std=std)
        else:
            torch.nn.init.trunc_normal_(noise, mean=0.0, std=std, a=-clamp, b=clamp)

    def _prefill_tokens(self, tokens, state, consume_noise=True):
        b = tokens.shape[0]
        lm = self.model.flow_lm
        embeddings = lm.conditioner(tokens)
        # Only transformer KV state is used; sampling the discarded prefill output is wasted.
        lm.transformer(embeddings, state)
        if consume_noise:
            self._consume_prefill_noise(b)
        for values in state.values():
            if "offset" in values:
                values["offset"].add_(tokens.shape[1])

    @staticmethod
    def _shift_rotary_key(keys, shift, period):
        """Shift cached RoPE keys when right-aligning unequal-length prefixes."""
        if not shift:
            return keys
        d = keys.shape[-1]
        frequency = torch.exp(
            torch.arange(d // 2, device=keys.device, dtype=torch.float32)
            * (-math.log(period) * 2 / d)
        )
        angle = frequency * shift
        pairs = keys.reshape(*keys.shape[:-1], d // 2, 2)
        real, imaginary = pairs[..., 0], pairs[..., 1]
        return torch.stack(
            (
                real * angle.cos() - imaginary * angle.sin(),
                imaginary * angle.cos() + real * angle.sin(),
            ),
            dim=-1,
        ).reshape_as(keys)

    def prefill(self, texts):
        m = self.model
        tokens, normalized, delays, limits = [], [], [], []
        token_cache = {}
        voice_end = int(next(iter(self.voice.values()))["offset"].view(-1)[0].item())
        for text in texts:
            text, delay = prepare_text_prompt(
                text,
                m.pad_with_spaces_for_short_inputs,
                m.remove_semicolons,
                m.append_terminal_punctuation,
                m.capitalize_first_letter,
                m.replace_characters,
            )
            if text not in token_cache:
                token_cache[text] = m.flow_lm.conditioner.prepare(text)
            token = token_cache[text]
            tokens.append(token)
            normalized.append(text)
            delays.append(
                delay + 2
                if m.model_recommended_frames_after_eos is None
                else m.model_recommended_frames_after_eos
            )
            limits.append(m._estimate_max_gen_len(token.shape[1]))
        shared_end = voice_end + max(token.shape[1] for token in tokens)
        if shared_end + max(limits) + self.graph_steps > self.capacity:
            raise ValueError(
                "A text chunk or voice prompt exceeds KV capacity. "
                "Use shorter sentences or increase capacity when constructing VelvetTTS."
            )
        self._reset(self.state, self.voice)
        self.previous.fill_(float("nan"))
        if len(token_cache) == 1 and self.batch_size > 1:
            token = tokens[0]
            scratch = init_states(m.flow_lm, 1, self.capacity)
            self._reset(scratch, self.voice)
            self._prefill_tokens(token, scratch, consume_noise=False)
            for name, values in self.state.items():
                values["cache"][:, :, :shared_end].copy_(scratch[name]["cache"][:, :, :shared_end])
                values["offset"].fill_(shared_end)
            self._consume_prefill_noise(self.batch_size)
            mode = "shared_prefix"
            prefill_rows = 1
        elif len({token.shape[1] for token in tokens}) == 1:
            self._prefill_tokens(torch.cat(tokens), self.state)
            mode = "batched_equal_length"
            prefill_rows = self.batch_size
        else:
            # Prefill each distinct text once, batched by equal token length. Then scatter
            # valid prefixes to their original rows without changing the batch schedule.
            groups = defaultdict(list)
            for text, token in token_cache.items():
                groups[token.shape[1]].append((text, token))
            self._reset(self.state, {})
            for token_count, group in groups.items():
                scratch = init_states(m.flow_lm, len(group), self.capacity)
                self._reset(scratch, self.voice)
                self._prefill_tokens(
                    torch.cat([token for _, token in group]), scratch, consume_noise=False
                )
                end = voice_end + token_count
                shift = shared_end - end
                source_rows = {text: row for row, (text, _) in enumerate(group)}
                rows = [row for row, text in enumerate(normalized) if text in source_rows]
                destination_index = torch.tensor(rows, device=self.device)
                source_index = torch.tensor(
                    [source_rows[normalized[row]] for row in rows], device=self.device
                )
                rotary_groups = defaultdict(list)
                for name, module in self.attention.items():
                    rotary_groups[
                        (module.rope.max_period, module.num_kv_heads, module.dim_per_head)
                    ].append(name)
                rotated = {}
                for (period, _, _), names in rotary_groups.items():
                    keys = torch.stack([scratch[name]["cache"][0, :, :end] for name in names])
                    keys = self._shift_rotary_key(keys, shift, period)
                    rotated.update(zip(names, keys.unbind(0)))
                for name, values in self.state.items():
                    source = scratch[name]["cache"]
                    values["cache"][0, :, shift:shared_end].index_copy_(
                        0, destination_index, rotated[name].index_select(0, source_index)
                    )
                    values["cache"][1, :, shift:shared_end].index_copy_(
                        0, destination_index, source[1, :, :end].index_select(0, source_index)
                    )
                    values["offset"].fill_(shared_end)
                    values["pad"].index_fill_(0, destination_index, shift)
            # Earlier heterogeneous prefill drew one unused noise tensor per input row.
            # Keep that RNG consumption even though duplicate prefix computation is removed.
            for _ in tokens:
                self._consume_prefill_noise(1)
            mode = "grouped_unique_prefixes"
            prefill_rows = len(token_cache)
        self.last_prefill = {
            "mode": mode,
            "token_lengths": [token.shape[1] for token in tokens],
            "unique_prefixes": len(token_cache),
            "prefill_rows": prefill_rows,
        }
        return delays, limits

    def step(self):
        m = self.model
        value, eos = m.flow_lm._sample_next_latent(
            self.previous,
            self.empty_embeddings,
            self.state,
            m.sampler_decode_steps,
            m.temp,
            m.noise_clamp,
            m.eos_threshold,
        )
        for offset in self.offsets:
            offset.add_(1)
        self.previous.copy_(value[:, None])
        return value[:, None], eos[:, None]

    def replay(self):
        return self.step()

    def generate_latents(self, delays, limits):
        m = self.model
        lengths = [0] * self.batch_size
        ends = [None] * self.batch_size
        done = [False] * self.batch_size
        truncated = [False] * self.batch_size
        blocks = []
        i = 0
        while i < max(limits) and not all(done):
            latent, eos = self.replay()
            flags = eos.squeeze(-1).cpu().tolist()
            blocks.append(latent.clone())
            for step in range(self.graph_steps):
                if i >= max(limits):
                    break
                for row in range(self.batch_size):
                    if done[row]:
                        continue
                    if i >= limits[row]:
                        done[row] = truncated[row] = True
                        continue
                    if flags[row][step] and ends[row] is None and i >= m._MIN_FRAMES_BEFORE_EOS:
                        ends[row] = i
                    if ends[row] is not None and i >= ends[row] + delays[row]:
                        done[row] = True
                    else:
                        lengths[row] += 1
                i += 1
        for row in range(self.batch_size):
            if not done[row]:
                truncated[row] = True
        if any(truncated):
            raise RuntimeError(
                "Generation reached its safety limit before EOS for batch rows "
                f"{[i for i, value in enumerate(truncated) if value]}."
            )
        encoded = torch.cat(blocks, dim=1)[:, : max(lengths)]
        return encoded * m.flow_lm.emb_std + m.flow_lm.emb_mean, lengths

    def decode(self, encoded, timer):
        m = self.model
        ratio = int(m.mimi.encoder_frame_rate / m.mimi.frame_rate)
        state = codec_state(m.mimi, self.batch_size, encoded.shape[1] * ratio)
        blocks = []
        # CPU keeps codec calls small; CUDA amortizes dispatch by decoding the whole chunk.
        remaining = encoded
        m._offline_codec = self.device.type == "cuda" and not self.decoder_frames
        try:
            with timer.codec_hooks(m.mimi):
                while remaining.shape[1]:
                    count = min(self.decoder_frames or remaining.shape[1], remaining.shape[1])
                    blocks.append(m.mimi.decode_from_latent(remaining[:, :count], state))
                    for values in state.values():
                        if "offset" in values:
                            values["offset"].add_(count * ratio)
                    remaining = remaining[:, count:]
        finally:
            m._offline_codec = False
        audio = torch.cat(blocks, dim=-1)[:, 0]
        output_rate = m.sample_rate
        n = min(output_rate // 200, audio.shape[-1])
        audio[:, :n] *= torch.linspace(0, 1, output_rate // 200, device=self.device)[:n]
        return audio

    @torch.no_grad()
    def run(self, texts, timer, seed):
        with timer.phase("runtime_setup"):
            self.prepare()
        torch.manual_seed(seed)
        with timer.phase("prefill"):
            delays, limits = self.prefill(texts)
        with timer.phase("lm"):
            encoded, lengths = self.generate_latents(delays, limits)
        with timer.phase("decoder"):
            audio = self.decode(encoded, timer)
        if self.enhancer is not None:
            with timer.phase("enhancement"):
                audio = self.enhancer(audio, sample_rate=self.model.sample_rate)
        with timer.phase("transfer"):
            host = audio.cpu()
        frame_size = round(
            (48000 if self.enhancer is not None else self.model.sample_rate)
            / self.model.mimi.frame_rate
        )
        return [
            host[row, : length * frame_size].clone() for row, length in enumerate(lengths)
        ], lengths
