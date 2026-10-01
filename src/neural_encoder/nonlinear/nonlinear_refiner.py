"""Estimator interface for residual refinement of fixed representations."""

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any, Literal, Self

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from torch import nn

from ..utils.evaluation import evaluate_views
from ..utils.logging import TrainingLogger
from ..utils.split_views import split_views
from .losses import MultiViewContrastiveLoss
from .networks import ResidualMLP, ResidualNetwork

__all__ = ["NonlinearRefiner"]


class NonlinearRefiner(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Train a residual network on aligned views of fixed representations.

    Parameters
    ----------
    network : {"default"}, nn.Module, or callable, default="default"
        Residual branch g in f(z, y) = z + alpha*g(y), with y=z by default.
        The default is an MLP with one 768-unit GELU hidden layer and dropout
        0.1. Its output layer starts at zero. A factory is called with
        n_features (output size) and network_kwargs; with separate inputs it also
        receives network_input_features. A supplied module is trained directly,
        updating its parameters in place.
    network_kwargs : dict or None, default=None
        Constructor arguments for the default MLP or network factory.
    initial_alpha : float, default=0.25
        Initial residual multiplier. Use a nonzero value with a zero-output
        residual branch so training can begin.
    trainable_alpha : bool, default=True
        Whether alpha is optimized along with network parameters.
    loss : {"default"}, nn.Module, or callable, default="default"
        Default: MultiViewContrastiveLoss(). A custom objective must accept
        loss(outputs, inputs=inputs) and return a scalar tensor. Supplied
        loss modules are used directly; their trainable parameters are optimized.
    loss_kwargs : dict or None, default=None
        Constructor arguments for the default loss only.
    optimizer : callable, default=torch.optim.AdamW
        Optimizer factory accepting model and loss parameters.
    optimizer_kwargs : dict or None, default=None
        Overrides for defaults lr=1e-4 and weight_decay=1e-4.
    scheduler : callable or None, default=None
        Optional scheduler factory receiving the optimizer. Its step method
        is called after every optimization step, without a metric argument.
    scheduler_kwargs : dict or None, default=None
        Arguments for the scheduler factory.
    steps : int, default=2000
        Number of optimizer steps per fit.
    batch_size : int, default=512
        Distinct samples per step, capped at the number of available samples.
        The same sample indices are used in every view.
    impute : {"discard", "mean"}, default="discard"
        Missing-view handling for matrix-based fitting. fit_views requires
        complete, aligned views. Imputation introduces synthetic positives.
    grad_clip : float or None, default=1.0
        Maximum gradient norm, or None to disable clipping.
    device : str, default="auto"
        PyTorch device. Auto selects CUDA, then MPS, then CPU.
    dtype : torch.dtype, default=torch.float32
        Float32 or float64 dtype used for model parameters and training data.
    random_state : int or None, default=None
        Seed for initialization, dropout, and sample selection. Reproducibility
        is intended for the same device and software environment.
    eval_every : int, default=200
        Evaluate supplied validation views every this many steps and at the
        final step. Validation is monitoring only; it does not select weights.
    logger : TrainingLogger or None, default=None
        Optional logger with log(metrics, step=step). No external logging
        occurs by default. If provided, flush() is called after training to
        emit any partial averaging window. The caller closes the logger.

    Attributes
    ----------
    model_ : ResidualNetwork
        Trained PyTorch model, left in evaluation mode. Use directly for
        tensor outputs or differentiable inference.
    loss_ : callable or nn.Module
        Resolved training objective. Custom losses receive base representations
        through inputs, irrespective of the residual branch input.
    history_ : list of dict
        Scalar metrics for every step, including optional validation metrics.
    n_features_in_, n_views_, n_samples_, n_steps_ : int
        Input feature count, training view count, aligned sample count, and
        completed optimizer steps.
    network_input_features_ : int
        Number of input features for the residual branch.
    separate_network_input_ : bool
        Whether fitting used separate residual branch inputs.
    device_ : torch.device
        Resolved device used for training and inference.

    Notes
    -----
    Input representations are detached from any upstream graph. No upstream
    encoder is fitted or updated. Each fit resets the optimizer, history, and
    alpha. Supplied modules retain their current parameters; default networks
    and network factories create a new network on each fit.
    transform accepts arrays or tensors and returns NumPy; ``model_`` is the
    direct PyTorch interface. Validation views must be held out by sample
    identity by the caller and contain at least three aligned samples.
    """

    def __init__(
        self, *, network: Any = "default", network_kwargs: Mapping[str, Any] | None = None,
        initial_alpha: float = 0.25, trainable_alpha: bool = True,
        loss: Any = "default", loss_kwargs: Mapping[str, Any] | None = None,
        optimizer: Callable = torch.optim.AdamW, optimizer_kwargs: Mapping[str, Any] | None = None,
        scheduler: Callable | None = None, scheduler_kwargs: Mapping[str, Any] | None = None,
        steps: int = 2000, batch_size: int = 512, impute: Literal["discard", "mean"] = "discard",
        grad_clip: float | None = 1.0, device: str = "auto", dtype: torch.dtype = torch.float32,
        random_state: int | None = None, eval_every: int = 200, logger: TrainingLogger | None = None,
    ) -> None:
        self.network = network
        self.network_kwargs = network_kwargs
        self.initial_alpha = initial_alpha
        self.trainable_alpha = trainable_alpha
        self.loss = loss
        self.loss_kwargs = loss_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.steps = steps
        self.batch_size = batch_size
        self.impute = impute
        self.grad_clip = grad_clip
        self.device = device
        self.dtype = dtype
        self.random_state = random_state
        self.eval_every = eval_every
        self.logger = logger

    def fit(
        self, X: ArrayLike, y: Any = None, *, sample_ids: ArrayLike,
        view_ids: ArrayLike | None = None, network_input: ArrayLike | None = None,
        validation_views: Sequence[ArrayLike] | None = None,
        validation_network_input: Sequence[ArrayLike] | None = None,
    ) -> Self:
        """Fit from measurements and sample identities; optional y is ignored.

        network_input supplies row-aligned inputs for the residual branch,
        defaulting to X. Validation inputs are supplied as aligned view lists.
        """
        self._validate_parameters()
        X = validate_data(self, _as_array(X), dtype=[np.float64, np.float32], ensure_min_samples=2)
        if network_input is None:
            views = split_views(X, sample_ids, view_ids, impute=self.impute)
            network_views = None
        else:
            network_input = _network_array(network_input, X)
            combined = split_views(np.concatenate([X, network_input], axis=1),
                                   sample_ids, view_ids, impute=self.impute)
            views = [view[:, :self.n_features_in_] for view in combined]
            network_views = [view[:, self.n_features_in_:] for view in combined]
        return self._fit_views(views, validation_views, network_views, validation_network_input)

    def fit_views(
        self, views: Sequence[ArrayLike], *, network_input: Sequence[ArrayLike] | None = None,
        validation_views: Sequence[ArrayLike] | None = None,
        validation_network_input: Sequence[ArrayLike] | None = None,
    ) -> Self:
        """Fit aligned views, optionally with corresponding residual input views."""
        self._validate_parameters()
        arrays = _view_arrays(views)
        validate_data(self, arrays[0], dtype=[np.float64, np.float32])
        return self._fit_views(arrays, validation_views, network_input, validation_network_input)

    def _validate_parameters(self) -> None:
        for name in ("steps", "batch_size", "eval_every"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least two.")
        if self.impute not in ("discard", "mean"):
            raise ValueError("impute must be 'discard' or 'mean'.")
        if self.grad_clip is not None and (not np.isfinite(self.grad_clip) or self.grad_clip <= 0):
            raise ValueError("grad_clip must be positive or None.")
        if self.random_state is not None and (
            not isinstance(self.random_state, Integral) or isinstance(self.random_state, bool)
            or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer or None.")
        if not np.isfinite(self.initial_alpha):
            raise ValueError("initial_alpha must be finite.")
        if self.scheduler is None and self.scheduler_kwargs is not None:
            raise ValueError("scheduler_kwargs requires a scheduler.")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64.")

    def _fit_views(
        self, views: Sequence[ArrayLike], validation_views: Sequence[ArrayLike] | None,
        network_input: Sequence[ArrayLike] | None,
        validation_network_input: Sequence[ArrayLike] | None,
    ) -> Self:
        arrays = _view_arrays(views)
        validation = None if validation_views is None else _view_arrays(validation_views, min_samples=3)
        if validation is not None and validation[0].shape[1] != self.n_features_in_:
            raise ValueError("Validation views must have the fitted feature count.")
        self.separate_network_input_ = network_input is not None
        network_arrays = _network_views(network_input, arrays)
        self.network_input_features_ = network_arrays[0].shape[1]
        if validation_network_input is not None and validation is None:
            raise ValueError("validation_network_input requires validation_views.")
        validation_network = None
        if validation is not None:
            if self.separate_network_input_ and validation_network_input is None:
                raise ValueError("validation_network_input is required when fitting separate inputs.")
            validation_network = _network_views(validation_network_input, validation)
            if validation_network[0].shape[1] != self.network_input_features_:
                raise ValueError("Validation network inputs must have the fitted feature count.")
        self.device_ = _resolve_device(self.device)
        # Isolate CPU/CUDA initialization and dropout randomness from callers.
        mps_state = torch.mps.get_rng_state() if self.device_.type == "mps" else None
        try:
            with torch.random.fork_rng():
                seed = int(self.random_state) if self.random_state is not None else int(
                    np.random.default_rng().integers(0, 2**32)
                )
                torch.manual_seed(seed)
                self.model_ = self._make_model().to(device=self.device_, dtype=self.dtype)
                self.loss_ = self._make_loss()
                if isinstance(self.loss_, nn.Module):
                    self.loss_.to(device=self.device_, dtype=self.dtype)
                self.n_views_, self.n_samples_ = len(arrays), len(arrays[0])
                self._train(arrays, network_arrays, validation, validation_network, seed)
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
        self.model_.eval()
        if isinstance(self.loss_, nn.Module):
            self.loss_.eval()
        return self

    def _make_model(self) -> ResidualNetwork:
        kwargs = dict(self.network_kwargs or {})
        if isinstance(self.network, str) and self.network == "default":
            return ResidualMLP(
                self.n_features_in_, network_input_features=self.network_input_features_,
                initial_alpha=self.initial_alpha,
                trainable_alpha=self.trainable_alpha, **kwargs,
            )
        if isinstance(self.network, nn.Module):
            if self.network_kwargs is not None:
                raise ValueError("network_kwargs cannot accompany a module instance.")
            network = self.network
        elif callable(self.network):
            if self.separate_network_input_:
                kwargs["network_input_features"] = self.network_input_features_
            network = self.network(n_features=self.n_features_in_, **kwargs)
        else:
            raise TypeError("network must be 'default', an nn.Module, or a factory.")
        if not isinstance(network, nn.Module) or isinstance(network, ResidualNetwork):
            raise TypeError("network must produce a residual branch, not a full residual model.")
        return ResidualNetwork(network, initial_alpha=self.initial_alpha, trainable_alpha=self.trainable_alpha)

    def _make_loss(self) -> Callable:
        if isinstance(self.loss, str) and self.loss == "default":
            return MultiViewContrastiveLoss(**dict(self.loss_kwargs or {}))
        if self.loss_kwargs is not None:
            raise ValueError("loss_kwargs is only supported for the default loss.")
        if not callable(self.loss):
            raise TypeError("loss must be 'default' or a callable.")
        return self.loss

    def _train(
        self, arrays: Sequence[NDArray], network_arrays: Sequence[NDArray],
        validation: Sequence[NDArray] | None,
        validation_network: Sequence[NDArray] | None, seed: int,
    ) -> None:
        views = [torch.tensor(X, device=self.device_, dtype=self.dtype) for X in arrays]
        network_views = views if network_arrays is arrays else [
            torch.tensor(X, device=self.device_, dtype=self.dtype) for X in network_arrays
        ]
        parameters = list(self.model_.parameters())
        if isinstance(self.loss_, nn.Module):
            parameters += list(self.loss_.parameters())
            self.loss_.train()
        parameters = [p for p in parameters if p.requires_grad]
        optimizer = self.optimizer(parameters, **({"lr": 1e-4, "weight_decay": 1e-4} | dict(self.optimizer_kwargs or {})))
        scheduler = None if self.scheduler is None else self.scheduler(optimizer, **dict(self.scheduler_kwargs or {}))
        generator = torch.Generator().manual_seed(seed)
        self.history_, self.n_steps_ = [], 0
        self.model_.train()
        for step in range(1, self.steps + 1):
            indices = torch.randperm(self.n_samples_, generator=generator)[:self.batch_size].to(self.device_)
            inputs = [view[indices] for view in views]
            outputs = [self.model_(view, source[indices]) for view, source in zip(inputs, network_views)]
            if any(out.shape != inp.shape for out, inp in zip(outputs, inputs)):
                raise ValueError("The residual network must preserve the input shape.")
            loss = self.loss_(outputs, inputs=inputs)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError("The loss must return a scalar tensor.")
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite.")
            optimizer.zero_grad()
            loss.backward()
            metrics = {"loss": float(loss.detach()), "lr": float(optimizer.param_groups[0]["lr"])}
            if self.grad_clip is not None:
                norm = nn.utils.clip_grad_norm_(parameters, self.grad_clip)
                metrics["grad_norm"] = float(norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            self.n_steps_ = step
            metrics["alpha"] = float(self.model_.alpha.detach())
            if validation is not None and (step % self.eval_every == 0 or step == self.steps):
                metrics.update(self._evaluate(validation, validation_network))
            self.history_.append({"step": step, **metrics})
            if self.logger is not None:
                self.logger.log(dict(metrics), step=step)
        flush = getattr(self.logger, "flush", None)
        if callable(flush):
            flush()

    def _evaluate(self, arrays: Sequence[NDArray], network_arrays: Sequence[NDArray]) -> dict[str, float]:
        self.model_.eval()
        if isinstance(self.loss_, nn.Module):
            self.loss_.eval()
        with torch.no_grad():
            inputs = [torch.tensor(X, device=self.device_, dtype=self.dtype) for X in arrays]
            outputs = [self.model_(view, torch.tensor(source, device=self.device_, dtype=self.dtype))
                       for view, source in zip(inputs, network_arrays)]
            loss = float(self.loss_(outputs, inputs=inputs))
            metrics = evaluate_views(outputs)
        self.model_.train()
        if isinstance(self.loss_, nn.Module):
            self.loss_.train()
        return {"validation_loss": loss, **{f"validation_{key}": value for key, value in metrics.items()}}

    def to_torch(self) -> ResidualNetwork:
        """Return the fitted internal PyTorch module without copying it.

        Changes to this module also affect the refiner. Its current device,
        dtype, training mode, and gradient settings are preserved.
        """
        check_is_fitted(self, ["model_", "n_steps_"])
        return self.model_

    def transform(self, X: ArrayLike, *, network_input: ArrayLike | None = None) -> NDArray:
        """Refine X; network_input is required if separate inputs were used in fit."""
        check_is_fitted(self, ["model_", "n_steps_"])
        X = validate_data(self, _as_array(X), reset=False, dtype=[np.float64, np.float32])
        if network_input is None and self.separate_network_input_:
            raise ValueError("network_input is required because fitting used separate inputs.")
        source = X if network_input is None else _network_array(network_input, X)
        if source.shape[1] != self.network_input_features_:
            raise ValueError("network_input must have the fitted feature count.")
        self.model_.eval()
        with torch.no_grad():
            result = self.model_(torch.tensor(X, device=self.device_, dtype=self.dtype),
                                 torch.tensor(source, device=self.device_, dtype=self.dtype))
        return result.cpu().numpy()

    def fit_transform(
        self, X: ArrayLike, y: Any = None, *, sample_ids: ArrayLike,
        view_ids: ArrayLike | None = None, network_input: ArrayLike | None = None,
        validation_views: Sequence[ArrayLike] | None = None,
        validation_network_input: Sequence[ArrayLike] | None = None,
    ) -> NDArray:
        """Fit on aligned training views and transform all measurements."""
        return self.fit(
            X, y, sample_ids=sample_ids, view_ids=view_ids, network_input=network_input,
            validation_views=validation_views, validation_network_input=validation_network_input,
        ).transform(X, network_input=network_input)


def _as_array(X: ArrayLike) -> NDArray:
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu()
        X = (X if X.dtype == torch.float64 else X.float()).numpy()
    return np.asarray(X)


def _view_arrays(views: Sequence[ArrayLike], min_samples: int = 2) -> list[NDArray]:
    if len(views) < 2:
        raise ValueError("At least two views are required.")
    arrays = [check_array(_as_array(X), dtype=[np.float64, np.float32], ensure_min_samples=min_samples) for X in views]
    if any(X.shape != arrays[0].shape for X in arrays):
        raise ValueError("Views must have equal shapes with corresponding samples and features.")
    return arrays


def _network_array(network_input: ArrayLike, base: NDArray) -> NDArray:
    array = check_array(_as_array(network_input), dtype=[np.float64, np.float32])
    if len(array) != len(base):
        raise ValueError("network_input must have the same number of rows as the base representations.")
    return array


def _network_views(network_input: Sequence[ArrayLike] | None, base: list[NDArray]) -> list[NDArray]:
    if network_input is None:
        return base
    arrays = _view_arrays(network_input)
    if len(arrays) != len(base) or len(arrays[0]) != len(base[0]):
        raise ValueError("network_input must match the base view count and row count.")
    return arrays


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)
