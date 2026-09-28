"""Accumulating training loggers adapted from the supplied vec2vec example.

Original repository: https://github.com/rjha18/vec2vec
Jha, R., Zhang, C., Shmatikov, V., & Morris, J. X. (2025).
Harnessing the universal geometry of embeddings. arXiv:2505.12540.
https://arxiv.org/abs/2505.12540
"""

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, Protocol, Union

if TYPE_CHECKING:
    import torch

Scalar = Union[int, float, "torch.Tensor"]

__all__ = ["TrainingLogger", "RunningAverages", "AveragingLogger", "ConsoleLogger", "WandbLogger"]


class TrainingLogger(Protocol):
    """Logger interface; the caller owns external run setup and cleanup."""

    def log(self, metrics: Mapping[str, Scalar], *, step: int) -> None:
        """Record scalar metrics at a training step."""
        ...


class RunningAverages:
    """Accumulate scalar means independently per key, without retaining graphs.

    Python numbers and scalar tensors are accepted. Tensor values are detached
    and converted to Python floats. Each update has equal weight; a metric
    absent from a step is not counted as zero.
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def __iter__(self) -> Iterator[str]:
        return iter(self._sums)

    def update(self, key: str, value: Scalar) -> None:
        if hasattr(value, "detach"):
            value = value.detach().item()
        self._sums[key] = self._sums.get(key, 0.0) + float(value)
        self._counts[key] = self._counts.get(key, 0) + 1

    def get(self, key: str) -> float:
        """Return the current mean, or zero if the key has no observations."""
        count = self._counts.get(key, 0)
        return self._sums[key] / count if count else 0.0

    def get_all(self) -> dict[str, float]:
        return {key: self.get(key) for key in self._sums}

    def clear(self, key: str) -> None:
        self._sums.pop(key, None)
        self._counts.pop(key, None)

    def clear_all(self) -> None:
        self._sums.clear()
        self._counts.clear()

    def get_and_clear_all(self) -> dict[str, float]:
        metrics = self.get_all()
        self.clear_all()
        return metrics


class AveragingLogger(ABC):
    """Accumulate updates and emit per-key means every log_frequency steps.

    Use log(metrics, step=step) once per step, or call logkv/logkvs any number
    of times followed by dumpkvs() once per step. flush() emits a partial
    window without advancing the step. finish() flushes and closes the logger.
    """

    def __init__(self, log_frequency: int = 250) -> None:
        if not isinstance(log_frequency, Integral) or log_frequency < 1:
            raise ValueError("log_frequency must be a positive integer.")
        self.log_frequency = log_frequency
        self.log_step = 0
        self.vals = RunningAverages()
        self._pending_steps = 0
        self._finished = False

    def logkv(self, key: str, value: Scalar) -> Scalar:
        """Accumulate a value and return it unchanged."""
        if self._finished:
            raise RuntimeError("This logger has been finished.")
        self.vals.update(key, value)
        return value

    def logkvs(self, metrics: Mapping[str, Scalar]) -> None:
        for key, value in metrics.items():
            self.logkv(key, value)

    def log(self, metrics: Mapping[str, Scalar], *, step: int) -> None:
        self.logkvs(metrics)
        self.dumpkvs(step=step)

    def dumpkvs(self, force: bool = False, *, step: int | None = None) -> None:
        """Complete a step and emit averages if the window is full or forced."""
        if self._finished:
            raise RuntimeError("This logger has been finished.")
        self.log_step = self.log_step + 1 if step is None else step
        self._pending_steps += 1
        if self._pending_steps >= self.log_frequency or force:
            self.flush()

    def flush(self) -> None:
        """Emit pending averages at the current step; do not advance it."""
        metrics = self.vals.get_all()
        if metrics:
            self._write(metrics, step=self.log_step)
            self.vals.clear_all()
        self._pending_steps = 0

    @abstractmethod
    def _write(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Write one window of averaged metrics to the backend."""
        ...

    def _finish(self) -> None:
        pass

    def finish(self) -> None:
        """Flush remaining metrics and close the backend, once."""
        if not self._finished:
            self.flush()
            self._finish()
            self._finished = True


class ConsoleLogger(AveragingLogger):
    """Print averaged metrics; log_frequency=1 prints every step."""

    def __init__(self, log_frequency: int = 1) -> None:
        super().__init__(log_frequency)

    def _write(self, metrics: Mapping[str, float], *, step: int) -> None:
        values = " ".join(f"{name}={value:.5g}" for name, value in metrics.items())
        print(f"step={step} {values}")


class WandbLogger(AveragingLogger):
    """Send averaged metrics to an optional Weights & Biases run.

    mode defaults to disabled: no wandb import, login, or run creation occurs,
    and metrics are sent to Python's logging module. Online and offline modes
    require the optional wandb package (install neural_encoder[wandb]). entity
    and project default to WANDB_ENTITY and WANDB_PROJECT environment values.
    Extra kwargs are forwarded to wandb.init. Call finish() to close the run.
    """

    def __init__(
        self, name: str | None = None, config: dict[str, Any] | None = None,
        entity: str | None = None, project: str | None = None,
        log_frequency: int = 250, mode: Literal["disabled", "online", "offline"] = "disabled",
        **kwargs: Any,
    ) -> None:
        super().__init__(log_frequency)
        if mode not in ("disabled", "online", "offline"):
            raise ValueError("mode must be 'disabled', 'online', or 'offline'.")
        self.enabled = mode != "disabled"
        self.wandb_run = None
        if self.enabled:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError("Install neural_encoder[wandb] to enable W&B logging.") from exc
            self.wandb_run = wandb.init(
                name=name, config=config, entity=entity or os.getenv("WANDB_ENTITY"),
                project=project or os.getenv("WANDB_PROJECT"), mode=mode, **kwargs,
            )

    def _write(self, metrics: Mapping[str, float], *, step: int) -> None:
        if self.enabled:
            self.wandb_run.log(dict(metrics), step=step)
        else:
            logging.info("step=%s metrics=%s", step, dict(metrics))

    def _finish(self) -> None:
        if self.enabled:
            self.wandb_run.finish()
