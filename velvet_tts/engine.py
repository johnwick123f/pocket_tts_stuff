"""CPU/CUDA synthesis with batches, voice cloning, profiling, and batch-one streaming."""

import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
from pocket_tts.models.text_chunking import split_into_best_sentences

from ._model import load_model
from ._runtime import Runtime
from ._timing import Timings
from ._version import __version__
from .results import GenerationResult, Profile


class VelvetTTS:
    """Load once and reuse. Calls on one instance are serialized safely.

    batch_size limits simultaneously generated text chunks, not output padding.
    Different input texts may have different lengths. Returned audio is always CPU float32.
    """

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 1,
        voice: str | Path = "alba",
        capacity: int = 512,
        decoder_frames: int = 16,
        cpu_threads: int = 4,
        codec: str = "24k",
        flow_gain: float = 1.0,
        model: str = "18m",
        bundle: str | Path | None = None,
    ):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(capacity, int) or capacity < 64:
            raise ValueError("capacity must be an integer of at least 64.")
        if decoder_frames < 1 or cpu_threads < 1:
            raise ValueError("decoder_frames and cpu_threads must be positive.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are cpu, cuda, cuda:N, and auto.")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable. Install a CUDA-enabled PyTorch build or use device='cpu'."
                )
            self.device = torch.device(
                "cuda",
                self.device.index if self.device.index is not None else torch.cuda.current_device(),
            )
        else:
            torch.set_num_threads(cpu_threads)
        self.batch_size = batch_size
        self.codec = codec
        self.flow_gain = flow_gain
        self._bundle_data = None
        if bundle is not None:
            from .bundle import read_bundle

            self._bundle_data = read_bundle(bundle)
            model = self._bundle_data["model"]
        self.model_name = model
        self.capacity = capacity
        self.decoder_frames = decoder_frames
        self._lock = threading.Lock()
        self._runtimes = {}
        self._enhancer = None
        start = time.perf_counter()
        with self._device_context():
            self.model = (
                load_model(codec, flow_gain, model, bundle_data=self._bundle_data)
                if self._bundle_data is not None
                else load_model(codec, flow_gain, model)
            ).to(self.device)
            voice = self._voice_source(voice)
            voice_start = time.perf_counter()
            self.voice = self._prepare_voice(voice)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.voice_setup_seconds = time.perf_counter() - voice_start
            self.voice_source = str(voice)
            if self.device.type == "cuda":
                from ._codec_attention import configure_codec
                from ._cuda import configure_attention
                from ._torch_kernels import optimize_torch

                configure_attention(self.model, capacity)
                configure_codec(self.model)
                optimize_torch(self.model)
            else:
                from ._cpu import optimize_cpu

                optimize_cpu(self.model)
        self.load_seconds = time.perf_counter() - start

    @staticmethod
    def _voice_source(source):
        if isinstance(source, Path):
            return source.expanduser().resolve(strict=True)
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "Voice must be a predefined name, reference-audio path or voice-state URL."
            )
        if source.startswith(("hf://", "https://", "http://")):
            return source
        path = Path(source).expanduser()
        if path.is_file():
            return path.resolve()
        if "/" in source or "\\" in source or path.suffix:
            raise FileNotFoundError(f"Voice reference not found: {source}")
        return source

    @property
    def supports_voice_cloning(self) -> bool:
        return bool(self.model.has_voice_cloning)

    def _prepare_voice(self, source, truncate=False):
        if self._bundle_data is not None and isinstance(source, str):
            voices = self._bundle_data["voices"]
            if source not in voices:
                raise ValueError(
                    "Offline mode requires a bundled voice name or a local reference file."
                )
            return {
                name: {key: value.to(self.device) for key, value in state.items()}
                for name, state in voices[source].items()
            }
        if (
            isinstance(source, Path)
            and source.suffix.lower() in (".wav", ".wave")
            and self.supports_voice_cloning
        ):
            from ._audio import read_reference

            reference = read_reference(source, self.model.sample_rate, truncate)
            return self.model.get_state_for_audio_prompt(reference)
        return self.model.get_state_for_audio_prompt(source, truncate=truncate)

    @torch.no_grad()
    def use_voice(self, source: str | Path, *, truncate: bool = False) -> "VelvetTTS":
        """Switch to a reference WAV, saved state or predefined voice; encode only once.

        Existing graph/KV runtimes are discarded after successful preparation. Call warmup()
        again if you want the next generation to exclude graph capture. Failed switches leave
        the current voice and runtimes unchanged. Returns self for chaining.
        """
        source = self._voice_source(source)
        with self._lock, self._device_context():
            start = time.perf_counter()
            voice = self._prepare_voice(source, truncate=truncate)
            prefix = int(next(iter(voice.values()))["offset"].view(-1)[0].item())
            if prefix + 8 >= self.capacity:
                raise ValueError(
                    "Voice reference exceeds available KV capacity. "
                    "Use a shorter reference or construct VelvetTTS with larger capacity."
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.voice = voice
            self.voice_source = str(source)
            self.voice_setup_seconds = time.perf_counter() - start
            self._runtimes.clear()
        return self

    def _device_context(self):
        return torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()

    @property
    def sample_rate(self) -> int:
        return 48000 if self.codec == "48k" else self.model.sample_rate

    @property
    def stream_sample_rate(self) -> int:
        """Streaming bypasses enhancement and emits native 24 kHz audio."""
        return self.model.sample_rate

    def stream(
        self, text: str, *, chunk_frames: int = 8, seed: int = 101
    ) -> Iterator[torch.Tensor]:
        """Yield mono CPU float32 chunks for batch size one. Close an abandoned iterator."""
        if self.batch_size != 1:
            raise ValueError("Streaming requires batch_size=1.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provide a nonempty text string.")
        if not isinstance(chunk_frames, int) or isinstance(chunk_frames, bool) or chunk_frames < 1:
            raise ValueError("chunk_frames must be a positive integer.")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        def iterate():
            from ._stream import generate

            with self._lock:
                with torch.no_grad(), self._device_context():
                    chunks = self._split(text)
                for number, chunk in enumerate(chunks):
                    with torch.no_grad(), self._device_context():
                        worker = generate(self._runtime(1), chunk, chunk_frames, seed + number)
                    try:
                        while True:
                            with torch.no_grad(), self._device_context():
                                try:
                                    audio = next(worker)
                                except StopIteration:
                                    break
                            yield audio
                    finally:
                        worker.close()

        return iterate()

    def _runtime(self, size, enhance=False):
        if size not in self._runtimes:
            if self.device.type == "cuda":
                from ._cuda import CUDARuntime

                runtime = CUDARuntime(self.model, self.voice, size, self.capacity)
            else:
                runtime = Runtime(self.model, self.voice, size, self.capacity, self.decoder_frames)
            self._runtimes[size] = runtime
        runtime = self._runtimes[size]
        if enhance and self.codec == "48k" and self._enhancer is None:
            from .enhance import LavaSR

            self._enhancer = LavaSR(
                device=self.device, bundle=self._bundle_data["lava"] if self._bundle_data else None
            )
        runtime.enhancer = self._enhancer if enhance else None
        return runtime

    def _split(self, text):
        m = self.model
        return split_into_best_sentences(
            m.flow_lm.conditioner.tokenizer,
            text,
            50,
            m.pad_with_spaces_for_short_inputs,
            remove_semicolons=m.remove_semicolons,
            append_terminal_punctuation=m.append_terminal_punctuation,
            capitalize_first_letter=m.capitalize_first_letter,
            replace_characters=m.replace_characters,
        )

    @torch.no_grad()
    def warmup(self, batch_sizes: Sequence[int] | None = None, *, streaming: bool = False) -> float:
        """Initialize requested CUDA graph shapes and prime a short generation. Returns seconds."""
        sizes = list(batch_sizes) if batch_sizes is not None else [self.batch_size]
        if any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in sizes):
            raise ValueError("Warmup batch sizes must be positive integers.")
        start = time.perf_counter()
        with self._lock, self._device_context():
            timer = Timings(False, self.device)
            for size in dict.fromkeys(sizes):
                self._runtime(size, enhance=not streaming).run(
                    ["Hello, this is a warmup."] * size, timer, 101
                )
            timer.sync()
        return time.perf_counter() - start

    @torch.no_grad()
    def generate(
        self,
        texts: str | Sequence[str],
        *,
        batch_size: int | None = None,
        profile: bool = False,
        seed: int = 101,
    ) -> GenerationResult:
        """Generate one text or a sequence, returning unpadded audio in input order.

        Each effective CUDA batch size captures its graphs lazily on first use.
        Profiling synchronizes stage boundaries and therefore adds some overhead.
        """
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, Sequence):
            texts = list(texts)
        else:
            raise TypeError("texts must be a string or sequence of strings.")
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Provide at least one nonempty text string.")
        size = self.batch_size if batch_size is None else batch_size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        with self._lock, self._device_context():
            timer = Timings(profile, self.device)
            timer.sync()
            start = time.perf_counter()
            with timer.phase("text_processing"):
                split = {text: self._split(text) for text in dict.fromkeys(texts)}
                jobs = [
                    (row, part, chunk)
                    for row, text in enumerate(texts)
                    for part, chunk in enumerate(split[text])
                ]
            output = [[] for _ in texts]
            batches = []
            for number, start_index in enumerate(range(0, len(jobs), size)):
                batch = jobs[start_index : start_index + size]
                with timer.phase("runtime_setup"):
                    runtime = self._runtime(len(batch), enhance=True)
                audio, lengths = runtime.run([job[2] for job in batch], timer, seed + number)
                for (row, part, _), waveform in zip(batch, audio):
                    output[row].append((part, waveform))
                if profile:
                    batches.append(
                        {
                            "batch_size": len(batch),
                            "input_rows": [job[0] for job in batch],
                            "latent_frames": lengths,
                            "padding_utilization": sum(lengths) / (len(lengths) * max(lengths)),
                            "prefill": runtime.last_prefill,
                        }
                    )
            with timer.phase("assembly"):
                waveforms = [torch.cat([audio for _, audio in sorted(parts)]) for parts in output]
            timer.sync()
            elapsed = time.perf_counter() - start
            output_rate = self.sample_rate
            timings = (
                Profile(
                    elapsed,
                    [audio.numel() / output_rate for audio in waveforms],
                    dict(timer.stages),
                    dict(timer.decoder),
                    batches,
                    {
                        "package_version": __version__,
                        "model": self.model_name,
                        "codec": self.codec,
                        "flow_gain": self.flow_gain,
                        "torch_version": torch.__version__,
                        "device": str(self.device),
                        "backend": "pytorch",
                        "configured_batch_size": size,
                        "input_count": len(texts),
                        "chunk_count": len(jobs),
                        "voice": self.voice_source,
                    },
                )
                if profile
                else None
            )
            return GenerationResult(waveforms, output_rate, timings)
