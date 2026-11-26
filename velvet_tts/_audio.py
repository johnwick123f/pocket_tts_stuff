"""Local WAV reference decoding, including float WAVs written by GenerationResult.save."""

import math
import time
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly


def read_reference(path, sample_rate=24000, truncate=False):
    rate, audio = wavfile.read(path)
    if rate <= 0 or audio.size == 0 or audio.ndim not in (1, 2):
        raise ValueError("Voice reference must contain mono or multichannel audio.")
    if audio.dtype.kind == "u":
        midpoint = float(2 ** (np.iinfo(audio.dtype).bits - 1))
        audio = (audio.astype(np.float32) - midpoint) / midpoint
    elif audio.dtype.kind == "i":
        audio = audio.astype(np.float32) / float(2 ** (np.iinfo(audio.dtype).bits - 1))
    elif audio.dtype.kind == "f":
        audio = audio.astype(np.float32)
    else:
        raise ValueError("Unsupported WAV sample type.")
    if not np.isfinite(audio).all():
        raise ValueError("Voice reference contains nonfinite samples.")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if truncate:
        audio = audio[: 30 * rate]
    if rate != sample_rate:
        factor = math.gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // factor, rate // factor)
    return torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)


def save_stream(chunks, path, sample_rate):
    """Write incremental PCM16 WAV; measure inference time separately from file IO."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    elapsed, count, first = 0.0, 0, None
    iterator = iter(chunks)
    try:
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
            while True:
                start = time.perf_counter()
                try:
                    chunk = next(iterator)
                except StopIteration:
                    elapsed += time.perf_counter() - start
                    break
                elapsed += time.perf_counter() - start
                first = elapsed if first is None else first
                count += chunk.numel()
                output.writeframes((chunk.numpy().clip(-1, 1) * 32767).astype("<i2").tobytes())
    finally:
        iterator.close()
    seconds = count / sample_rate
    return {
        "streaming": True,
        "total_seconds": elapsed,
        "first_chunk_seconds": first,
        "audio_seconds": seconds,
        "aggregate_realtime": seconds / elapsed if elapsed else 0,
    }
