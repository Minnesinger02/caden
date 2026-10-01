"""Synchronized timings and CUDA allocator peaks for the local pilot."""
import json
import math
import time

from .common import percentile, sync


class TrainingTelemetry:
    def __init__(self, device, report_every=50):
        import torch
        self.device = device
        self.report_every = report_every
        self.losses = []
        self.step_seconds = []
        sync(device)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        self.started = self.previous = time.perf_counter()

    def step(self, loss):
        if not math.isfinite(loss):
            raise ValueError("Non-finite training loss; refusing to save a failed training run")
        sync(self.device)
        now = time.perf_counter()
        self.losses.append(loss)
        self.step_seconds.append(now - self.previous)
        self.previous = now
        if len(self.losses) % self.report_every == 0:
            print(json.dumps({"microsteps": len(self.losses),
                              "recent_loss": sum(self.losses[-self.report_every:]) / self.report_every,
                              "elapsed_seconds": now - self.started}), flush=True)

    def finish(self):
        import torch
        sync(self.device)
        cuda = str(self.device).startswith("cuda")
        return {"training_seconds": time.perf_counter() - self.started,
                "microsteps": len(self.losses),
                "step_p50_seconds": percentile(self.step_seconds, .5),
                "step_p95_seconds": percentile(self.step_seconds, .95),
                "first_50_loss": sum(self.losses[:50]) / len(self.losses[:50]),
                "last_50_loss": sum(self.losses[-50:]) / len(self.losses[-50:]),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if cuda else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if cuda else None,
                "timing_scope": "training loop including tokenization, forward/backward and optimizer; excludes loading and saving; synchronized each microstep"}
