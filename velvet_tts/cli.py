"""Command-line entry point; one --text per requested waveform."""

import argparse
import json
import sys
from pathlib import Path

from .engine import VelvetTTS


def main(argv=None):
    parser = argparse.ArgumentParser(description="VelvetTTS: 7M/18M speech synthesis, CPU or CUDA.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="Text to speak; repeat for a batch.")
    source.add_argument("--text-file", type=Path, help="UTF-8 input: one nonempty line per output.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--codec", choices=["24k", "48k", "small"], default="24k")
    parser.add_argument("--flow-gain", type=float, default=1.0)
    parser.add_argument("--model", choices=["7m", "18m"], default="18m")
    parser.add_argument("--bundle", type=Path, help="Self-contained offline model file.")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--stream", action="store_true", help="Write audio incrementally; one text, batch size one."
    )
    parser.add_argument(
        "--chunk-frames", type=int, default=8, help="Streaming group size; one frame is 80 ms."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--voice", default="alba", help="Predefined voice or reference audio path.")
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument(
        "--profile", action="store_true", help="Print generation and nested codec timings."
    )
    parser.add_argument(
        "--profile-json", type=Path, help="Save timings as JSON; enables profiling."
    )
    parser.add_argument(
        "--warmup", action="store_true", help="Warm batch sizes before timing the request."
    )
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument(
        "--decoder-frames", type=int, default=16, help="CPU offline decoder group size."
    )
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args(argv)
    texts = args.text
    if args.text_file:
        try:
            texts = [
                line.strip()
                for line in args.text_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError as error:
            parser.error(str(error))
    if not texts:
        parser.error("No nonempty input text was supplied.")
    output = args.output
    if len(texts) > 1 and output == Path("output.wav"):
        output = Path("output")
    if len(texts) > 1 and output.suffix.lower() == ".wav":
        parser.error("Use an output directory when generating multiple texts.")
    if args.stream and (len(texts) != 1 or args.batch_size != 1 or output.suffix.lower() != ".wav"):
        parser.error("--stream requires one text, --batch-size 1, and a .wav output.")
    try:
        engine = VelvetTTS(
            device=args.device,
            batch_size=args.batch_size,
            voice=args.voice,
            codec=args.codec,
            flow_gain=args.flow_gain,
            model=args.model,
            bundle=args.bundle,
            capacity=args.capacity,
            decoder_frames=args.decoder_frames,
            cpu_threads=args.cpu_threads,
        )
        print(f"Device: {engine.device}; model load: {engine.load_seconds:.2f}s", file=sys.stderr)
        if args.warmup:
            seconds = engine.warmup([1, args.batch_size], streaming=args.stream)
            print(f"Warmup: {seconds:.2f}s", file=sys.stderr)
        if args.stream:
            from ._audio import save_stream

            timings = save_stream(
                engine.stream(texts[0], chunk_frames=args.chunk_frames, seed=args.seed),
                output,
                engine.stream_sample_rate,
            )
            print(output.resolve())
            if args.profile or args.profile_json:
                print(json.dumps(timings, indent=2))
            if args.profile_json:
                args.profile_json.parent.mkdir(parents=True, exist_ok=True)
                args.profile_json.write_text(json.dumps(timings, indent=2))
            return 0
        result = engine.generate(
            texts, profile=args.profile or args.profile_json is not None, seed=args.seed
        )
        for path in result.save(output):
            print(path.resolve())
        if result.timings:
            print(json.dumps(result.timings.to_dict(), indent=2))
            if args.profile_json:
                result.timings.save(args.profile_json)
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0
