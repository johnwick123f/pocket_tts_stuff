"""Quick release throughput, streaming latency, and cloning check."""

import json
import statistics
import time
from pathlib import Path

import torch

from tests.benchmark import TEXT
from velvet_tts import VelvetTTS, __version__


def main():
    rows, streaming = [], []
    torch.set_num_threads(4)
    for device, batch in (("cuda", 1), ("cuda", 32), ("cpu", 1)):
        tts = VelvetTTS(device=device, batch_size=batch, codec="48k")
        for codec in ("24k", "48k"):
            tts.codec = codec
            warmup = tts.warmup()
            elapsed = []
            for _ in range(3):
                start = time.perf_counter()
                result = tts.generate([TEXT] * batch)
                elapsed.append(time.perf_counter() - start)
            seconds = statistics.median(elapsed)
            audio = sum(result.durations)
            row = {
                "device": device,
                "batch": batch,
                "codec": codec,
                "seconds": seconds,
                "rtf": seconds / audio,
                "realtime": audio / seconds,
                "audio_seconds": audio,
                "warmup_seconds": warmup,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        if batch == 1:
            tts.warmup(streaming=True)
            first, total = [], []
            for _ in range(3):
                start = time.perf_counter()
                iterator = tts.stream(TEXT, chunk_frames=4)
                initial = next(iterator)
                first.append(time.perf_counter() - start)
                chunks = [initial, *iterator]
                total.append(time.perf_counter() - start)
            audio = sum(x.numel() for x in chunks) / tts.stream_sample_rate
            row = {
                "device": device,
                "chunk_frames": 4,
                "sample_rate": tts.stream_sample_rate,
                "first_chunk_ms": statistics.median(first) * 1000,
                "realtime": audio / statistics.median(total),
            }
            streaming.append(row)
            print(json.dumps(row), flush=True)
        if device == "cuda" and batch == 1:
            tts.use_voice("/mnt/c/Users/Nitin/Downloads/phonon.wav")
            clone = tts.generate("Hello! This is a voice cloning check.", profile=True)
            assert clone.sample_rate == 48000 and torch.isfinite(clone.audio).all()
            assert "enhancement" in clone.timings.stages
            assert tts.supports_voice_cloning
            print("Voice cloning and enhancement profile checked", flush=True)
        del tts
        torch.cuda.empty_cache()
    Path("tests/performance.json").write_text(
        json.dumps(
            {
                "version": __version__,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "model": "18m",
                "voice": "alba",
                "flow_gain": 1.0,
                "cpu_threads": 4,
                "warm_median_runs": 3,
                "measurements": rows,
                "streaming": streaming,
                "streaming_warm": True,
                "voice_cloning_checked": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
