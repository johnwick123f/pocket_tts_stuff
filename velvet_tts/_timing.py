import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class Timings:
    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = enabled
        self.device = device
        self.stages = defaultdict(float)
        self.decoder = defaultdict(float)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.stages[name] += time.perf_counter() - start

    @contextmanager
    def codec_hooks(self, mimi):
        if not self.enabled:
            yield
            return
        hooks = []
        events = []
        try:
            for name, attribute in (
                ("projection", "quantizer"),
                ("upsample", "upsample"),
                ("transformer", "decoder_transformer"),
                ("convolutions", "decoder"),
            ):
                module = getattr(mimi, attribute)
                if self.device.type == "cuda":
                    pairs = []

                    def before(mod, args, pairs=pairs):
                        pair = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        pairs.append(pair)
                        pair[0].record()

                    def after(mod, args, output, pairs=pairs):
                        pairs[-1][1].record()

                    events.append((name, pairs))
                else:
                    starts = []

                    def before(mod, args, starts=starts):
                        starts.append(time.perf_counter())

                    def after(mod, args, output, starts=starts, name=name):
                        self.decoder[name] += time.perf_counter() - starts.pop()

                hooks.append(module.register_forward_pre_hook(before))
                hooks.append(module.register_forward_hook(after))
            yield
        finally:
            for hook in hooks:
                hook.remove()
            if events:
                self.sync()
                for name, pairs in events:
                    self.decoder[name] += (
                        sum(start.elapsed_time(end) for start, end in pairs) / 1000
                    )
