"""Quick warm benchmarks: python -m tests.benchmark --codec 24k."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from velvet_tts import VelvetTTS

TEXT = "This is a performance test of the seven million parameter Pocket text to speech model. We measure the language model, the sampling head, and the audio decoder separately."
UNEVEN = [
    "Haru-4B is a low-latency and multilingual full-duplex continuous spoken dialogue foundation model: it listens and speaks at the same time, the way people do on a phone call. It is a pretrained base model, released for research.",
    "However, a kind John Hopkins Professor sent a proxy PhD student for me who presented the work on my behalf. After the conference, the student also said the paper was well-received and even Professor Shinji Watanabe (a giant in speech AI) said the paper was well-made which was incredibly rewarding.",
]


def measure(tts, name, texts, size):
    tts.generate(texts, batch_size=size)
    speeds = []
    for _ in range(3):
        start = time.perf_counter()
        result = tts.generate(texts, batch_size=size)
        elapsed = time.perf_counter() - start
        speeds.append(sum(result.durations) / elapsed)
    row = {
        "case": name,
        "device": str(tts.device),
        "codec": tts.codec,
        "batch_size": size,
        "realtime": statistics.median(speeds),
        "runs": speeds,
    }
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", choices=["24k", "48k", "small"], default="24k")
    parser.add_argument("--model", choices=["7m", "18m"], default="18m")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("tests/performance.json"))
    args = parser.parse_args()
    rows = []
    cpu = VelvetTTS(device="cpu", codec=args.codec, model=args.model, cpu_threads=args.cpu_threads)
    warmup = cpu.warmup()
    setup = {"cpu": {"load_seconds": cpu.load_seconds, "warmup_seconds": warmup}}
    print(f"CPU load {cpu.load_seconds:.3f}s; warmup {warmup:.3f}s", flush=True)
    rows.append(measure(cpu, "medium", TEXT, 1))
    del cpu
    if torch.cuda.is_available():
        gpu = VelvetTTS(device="cuda", batch_size=32, codec=args.codec, model=args.model)
        warmup = gpu.warmup([1, 16, 32])
        setup["cuda"] = {"load_seconds": gpu.load_seconds, "warmup_seconds": warmup}
        print(f"CUDA load {gpu.load_seconds:.3f}s; warmup {warmup:.3f}s", flush=True)
        rows.append(measure(gpu, "medium", TEXT, 1))
        for size in (16, 32):
            rows.append(measure(gpu, "uniform", [TEXT] * size, size))
            rows.append(measure(gpu, "uneven", UNEVEN * (size // 2), size))
    report = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "flow_gain": 1.0,
        "model": args.model,
        "cpu_threads": args.cpu_threads,
        "voice": "alba",
        "measurements": rows,
        "setup": setup,
    }
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
