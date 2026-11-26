"""Batch-one incremental synthesis at the codec's native 24 kHz rate."""

import torch

from ._runtime import codec_state


class AudioDecoder:
    def __init__(self, model, limit, batch_size=1):
        self.mimi = model.mimi
        self.ratio = int(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
        self.state = codec_state(self.mimi, batch_size, (limit + 8) * self.ratio)

    def push(self, latent):
        mimi = self.mimi
        audio = mimi.decode_from_latent(latent, self.state)
        for values in self.state.values():
            if "offset" in values:
                values["offset"].add_(latent.shape[1] * self.ratio)
        return audio[:, 0]


def generate(runtime, text, frames, seed):
    """Yield device audio while the LM is still generating future frames."""
    runtime.prepare()
    torch.manual_seed(seed)
    delays, limits = runtime.prefill([text])
    model = runtime.model
    decoder = AudioDecoder(model, limits[0])
    pending = None
    end, position, done, first = None, 0, False, True
    while position < limits[0] and not done:
        latent, eos = runtime.replay()
        flags = eos[0, :, 0].cpu().tolist()
        count = 0
        for flag in flags:
            if position >= limits[0]:
                break
            if flag and end is None and position >= model._MIN_FRAMES_BEFORE_EOS:
                end = position
            if end is not None and position >= end + delays[0]:
                done = True
                break
            count += 1
            position += 1
        if count:
            block = latent[:, :count].clone() * model.flow_lm.emb_std + model.flow_lm.emb_mean
            pending = block if pending is None else torch.cat((pending, block), dim=1)
        while pending is not None and (pending.shape[1] >= frames or done):
            count = min(frames, pending.shape[1])
            audio = decoder.push(pending[:, :count])[0]
            pending = pending[:, count:]
            if first:
                n = model.sample_rate // 200
                audio[:n] *= torch.linspace(0, 1, n, device=audio.device)[: audio.shape[-1]]
                first = False
            yield audio.float().cpu()
            if not pending.shape[1]:
                pending = None
    if not done:
        raise RuntimeError("Streaming generation reached its safety limit before EOS.")
