# VelvetTTS

A small inference library for Pocket TTS on CPU and CUDA. Default: **18M model,
standard 24 kHz codec, flow gain 1.0**. Supports voice cloning, heterogeneous
batches, optional profiling, and incremental audio streaming at batch size one.

## Install

Python 3.10–3.13. Install the appropriate PyTorch 2.8 CPU/CUDA build, then:

```bash
pip install .
```

First use downloads the upstream checkpoints. Voice cloning may require accepting
Kyutai's model terms and signing in to Hugging Face. For 48 kHz enhancement, install
`pip install '.[enhance]'`. Neither compilation nor custom C/CUDA/Triton kernels are required.

## Offline single-file models

Each bundle contains the model, tokenizer, built-in voices, and LavaSR v2.
Create both files once (downloads may occur during this export):

```bash
python -m velvet_tts.bundle --output-dir models
```

Then inference uses only the selected local file, including 48 kHz enhancement:

```python
from velvet_tts import VelvetTTS

tts = VelvetTTS(bundle="models/velvet-18m.pt", device="cuda", codec="48k")
tts.use_voice("reference.wav")  # Local cloning still works.
tts.generate("Hello there.").save("hello.wav")
```

Use `velvet-7m.pt` for 7M; model size is inferred from the bundle.
The CLI accepts `--bundle models/velvet-18m.pt`. Runtime Python dependencies must
still be installed; the bundles contain model assets, not Python packages.

## Generate

```python
from velvet_tts import VelvetTTS

tts = VelvetTTS(device="cuda", batch_size=32, model="18m", codec="48k")
tts.use_voice("reference.wav")                  # WAV, MP3, FLAC, or a predefined voice
warmup_seconds = tts.warmup()                  # optional; reuse the instance
result = tts.generate(["Hello there.", "Another sentence."], profile=True)
result.save("output")
print(result.timings.to_dict())
```

A single string works too: `tts.generate("Hello.").save("hello.wav")`.
`result.audio` is the single waveform; `result.waveforms` contains batch outputs
in input order. Audio is unpadded CPU float32; its rate is `result.sample_rate`.

| Option | Meaning |
|---|---|
| `model="18m"` / `"7m"` | Base checkpoints; both use the same standard codec |
| `codec="24k"` | Original standard decoder |
| `codec="48k"` | Standard 24 kHz Mimi followed by LavaSR v2, 12 kHz cutoff; no denoiser |
| `codec="small"` | Faster, lower-fidelity decoder from the 2M checkpoint; 24 kHz output |
| `flow_gain=1.0` / `1.10` | Original weights (default) / optional conditioning-weight gain |
| `device="cpu"` | Defaults to four PyTorch threads; `cpu_threads=8` can be faster |

## Optional LavaSR enhancement

Install with `pip install '.[enhance]'`. Uses Vocos directly, downloads only the
LavaSR v2 enhancer weights/config, and has no LavaSR package or denoiser dependency.
`codec="48k"` applies it automatically on-device before transferring the audio to CPU.
Enhancement loads lazily; `warmup()` includes its first-use setup.

```python
from velvet_tts.enhance import LavaSR

enhancer = LavaSR(device="cuda", cutoff=12000)  # Load once; CPU also works.
audio48 = enhancer(result.audio, sample_rate=result.sample_rate)
# audio48 is a tensor at 48000 Hz; [B, T] batches are also accepted.
```

Flow gain is baked into weights and adds no inference work. Constructor
`voice="reference.wav"` and later `use_voice(path)` both prepare voice state once.
Changing voices clears cached runtimes. `load_seconds`, `voice_setup_seconds`,
and the return value of `warmup()` expose setup costs.

## Stream one text

```python
streamer = VelvetTTS(device="cuda", batch_size=1, voice="reference.wav")
for audio_chunk in streamer.stream("Hello. The stars look beautiful tonight.", chunk_frames=4):
    print(audio_chunk.shape, streamer.stream_sample_rate)  # Always 24000 Hz.
```

This yields audio while later frames are still being generated. One frame is
80 ms of audio; smaller groups favor latency over throughput. All codec options and
both devices support streaming. Streaming bypasses LavaSR and emits native 24 kHz
chunks, including when `codec="48k"` is selected. `warmup(streaming=True)` primes
streaming without loading enhancement. If retaining an iterator and stopping
early, call `iterator.close()` before reusing the instance. Calls on one instance
are serialized. CUDA graphs warm lazily; `warmup([1, 16, 32])` prepares known sizes.
A partial final batch can capture its own graph; warm that size too when needed.

## CLI

```bash
velvet-tts --device cuda --model 18m --batch-size 32 --text "Hello." --output hello.wav
velvet-tts --device cpu --codec 48k --voice reference.wav --stream \
  --text "Hello there." --output streamed.wav --profile
```

Repeat `--text` for batches or use `--text-file` (one input per line). Streaming
requires one text and batch size one; its WAV is written incrementally as PCM16.
`--profile-json timing.json` saves timings. Offline profiling includes LM, decoder,
enhancement, transfer, and nested codec stages; streaming reports first-chunk latency and total
inference time. File writing is excluded from inference timing.

## Performance and checks

The runtime uses float32 weights without quantization; native PyTorch/cuDNN math
defaults apply. CUDA chooses
its attention cache size as history grows and uses local windows in the codec.
Floating-point reduction order can differ from upstream. CPU thread settings affect
the process. Long text is split into sentences; uneven batches incur padded work.
Realtime speed is summed useful audio duration divided by inference wall time.

Current hardware, workloads, and three-run medians are recorded in
[tests/performance.json](tests/performance.json). Cached CUDA warmup is around
1–2 seconds; there is no torch.compile path. CPU offline decoding defaults to
16-frame groups; streaming controls its own group size.

18M, flow 1.0, i7-13700KF / RTX 4070 Ti SUPER, WSL2, PyTorch 2.8:

| Workload | Standard 24 kHz | Mimi + LavaSR 48 kHz |
|---|---:|---:|
| CPU batch 1, four threads | 21.4x | 17.7x |
| GPU batch 1 | 93.0x | 86.6x |
| GPU batch 32, uniform | 661.9x | 548.8x |

Warm streaming first-chunk latency with `chunk_frames=4`: GPU 25 ms, CPU 32 ms.
Streaming outputs 24 kHz. These latencies exclude initialization and warmup.

These are warm, hardware/workload-dependent measurements, excluding file IO.

```bash
pip install '.[dev,enhance]'
pytest -p no:cacheprovider
RUN_MODEL_TESTS=1 pytest -p no:cacheprovider  # CPU/CUDA integration, including streaming
python -m tests.benchmark_release           # 24k/48k throughput and streaming latency
```

The library lives in `velvet_tts/`; experiments are excluded from package builds.
The retired native 48 kHz codec and weights are archived outside the repository.
