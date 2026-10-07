"""Feature weighting from reliability across repeated measurements."""

import itertools
from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ..utils.split_views import split_views

__all__ = ["FeatureReweighting"]

# SNR weighting caps reliability at 1 - _MAX_RELIABILITY_GAP to keep weights finite.
_MAX_RELIABILITY_GAP = 1e-6


class FeatureReweighting(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Weight corresponding features by their reliability across views.

    Reliability is computed separately for each feature across aligned
    samples, either as the mean Pearson correlation between each view and the
    mean of all other views, or as the mean correlation between pairs of
    single views. Negative reliability is clipped to zero. Transform
    multiplies each feature by its fitted weight.

    With SNR weighting, reliability r is converted to the signal-to-noise
    ratio r / (1 - r) of a single measurement, which requires the
    single-view reliability of method="pairwise". Weights sqrt(SNR) make each
    feature contribute in proportion to its SNR to inner products and cosine
    similarities between measurements, the optimal weighting for comparing
    noisy measurements of a shared signal with independent Gaussian noise.
    This suits the output of a multiview projection, whose dimensions have
    similar variance but very different reliability.

    Parameters
    ----------
    method : {"correlation", "pairwise"}, default="correlation"
        Estimate reliability using leave-one-view-out Pearson correlation
        (each view against the mean of the others), or the mean Pearson
        correlation between pairs of views, which estimates the reliability
        of a single measurement.
    weighting : {"linear", "sqrt", "snr", "sqrt_snr"} or callable, default="linear"
        "linear" uses max(reliability, 0) + eps, "sqrt" its square root, and
        a callable receives this vector and must return a finite real vector
        of the same shape, used without further clipping. "snr" uses
        r / (1 - r) + eps with r = max(reliability, 0), capped below one, and
        "sqrt_snr" its square root.
    eps : float, default=1e-6
        Nonnegative offset added after clipping negative reliability (or to
        the SNR). This is a weight floor, not a correlation-denominator
        regularizer.
    normalize : bool, default=False
        Rescale the weights to unit root mean square, which preserves the
        overall scale of the transformed features.

    Attributes
    ----------
    reliability_ : ndarray of shape (n_features,)
        Mean leave-one-view-out or pairwise correlations, before clipping
        and weighting. Comparisons involving a constant feature contribute
        zero.
    weights_ : ndarray of shape (n_features,)
        Multipliers applied by transform.
    n_features_in_ : int
        Number of fitted features.
    n_views_ : int
        Number of views used for fitting.
    n_samples_ : int
        Number of complete aligned samples used for fitting.

    Notes
    -----
    All views must share corresponding features. Matrix-based fitting uses
    only samples present in every observed view. At least two views and two
    complete samples are required. Inputs must be finite; missing views are
    represented by absent measurements, not imputed values.

    Examples
    --------
    >>> reweighting = FeatureReweighting()
    >>> X = [[1., 4.], [2., 3.], [3., 2.], [4., 1.]]
    >>> weighted = reweighting.fit_transform(X, sample_ids=[0, 0, 1, 1])
    >>> weighted.shape
    (4, 2)
    """

    def __init__(
        self,
        *,
        method: Literal["correlation", "pairwise"] = "correlation",
        weighting: (
            Literal["linear", "sqrt", "snr", "sqrt_snr"] | Callable[[NDArray[np.float64]], ArrayLike]
        ) = "linear",
        eps: float = 1e-6,
        normalize: bool = False,
    ) -> None:
        self.method = method
        self.weighting = weighting
        self.eps = eps
        self.normalize = normalize

    def _validate_parameters(self) -> None:
        if self.method not in ("correlation", "pairwise"):
            raise ValueError("method must be 'correlation' or 'pairwise'.")
        if not callable(self.weighting) and self.weighting not in ("linear", "sqrt", "snr", "sqrt_snr"):
            raise ValueError("weighting must be 'linear', 'sqrt', 'snr', 'sqrt_snr', or a callable.")
        if not isinstance(self.eps, Real) or not np.isfinite(self.eps) or self.eps < 0:
            raise ValueError("eps must be a finite nonnegative number.")
        if not isinstance(self.normalize, (bool, np.bool_)):
            raise ValueError("normalize must be a boolean.")

    def fit(
        self,
        X: ArrayLike,
        y: Any = None,
        *,
        sample_ids: ArrayLike,
        view_ids: ArrayLike | None = None,
    ) -> Self:
        """Learn weights from measurements and their sample/view identities.

        If view_ids is omitted, views follow order of occurrence within each
        sample. Incomplete samples are excluded from reliability estimation.
        The optional y argument is ignored.
        """
        self._validate_parameters()
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        views = split_views(X, sample_ids, view_ids, impute="discard")
        return self._fit_aligned_views(views)

    def fit_views(self, views: Sequence[ArrayLike]) -> Self:
        """Learn weights from at least two aligned, equally shaped matrices.

        Every row must represent the same sample across all views, and every
        column the same feature. Views must contain finite, observed values.
        """
        self._validate_parameters()
        if len(views) < 2:
            raise ValueError("At least two views are required.")
        arrays = [validate_data(self, views[0], dtype=np.float64, ensure_min_samples=2)]
        arrays.extend(
            validate_data(self, view, reset=False, dtype=np.float64, ensure_min_samples=2)
            for view in views[1:]
        )
        return self._fit_aligned_views(arrays)

    def _fit_aligned_views(self, views: Sequence[NDArray[Any]]) -> Self:
        if len(views) < 2:
            raise ValueError("At least two views are required.")
        if any(view.shape != views[0].shape for view in views):
            raise ValueError("All views must have the same shape and aligned samples.")
        if views[0].shape[0] < 2:
            raise ValueError("At least two complete samples are required.")
        if self.method == "correlation":
            self.reliability_ = _correlation_reliability(views)
        else:
            self.reliability_ = _pairwise_reliability(views)
        self.weights_ = self._weights(self.reliability_)
        self.n_views_ = len(views)
        self.n_samples_ = views[0].shape[0]
        return self

    def _set_reliability(self, reliability: NDArray[np.float64]) -> None:
        """Replace the fitted reliability (for example, by a cross-validated estimate) and its weights."""
        self.reliability_ = np.asarray(reliability, dtype=np.float64)
        self.weights_ = self._weights(self.reliability_)

    def _weights(self, reliability: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert reliability to weights according to weighting and normalize."""
        weights = np.maximum(reliability, 0)
        if not callable(self.weighting) and self.weighting in ("snr", "sqrt_snr"):
            weights = np.minimum(weights, 1 - _MAX_RELIABILITY_GAP)
            weights = weights / (1 - weights)
        weights = weights + self.eps
        if callable(self.weighting):
            weights = np.asarray(self.weighting(weights))
            if weights.shape != reliability.shape or weights.dtype.kind not in "iuf":
                raise ValueError("weighting callable must return a real vector with one value per feature.")
            weights = weights.astype(np.float64, copy=True)
            if not np.isfinite(weights).all():
                raise ValueError("weighting callable must return only finite values.")
        elif self.weighting in ("sqrt", "sqrt_snr"):
            weights = np.sqrt(weights)
        if self.normalize:
            scale = np.sqrt(np.mean(weights ** 2))
            if scale > 0:
                weights = weights / scale
        return weights

    def transform(self, X: ArrayLike) -> NDArray[Any]:
        """Apply fitted weights, preserving row order and leaving X unchanged."""
        check_is_fitted(self, ["reliability_", "weights_"])
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        return X * self.weights_

    def fit_transform(
        self,
        X: ArrayLike,
        y: Any = None,
        *,
        sample_ids: ArrayLike,
        view_ids: ArrayLike | None = None,
    ) -> NDArray[Any]:
        """Fit using complete samples, then transform every input measurement."""
        return self.fit(X, y, sample_ids=sample_ids, view_ids=view_ids).transform(X)


def _pairwise_reliability(views: Sequence[NDArray[Any]]) -> NDArray[np.float64]:
    """Average correlations between pairs of views, vectorized over features."""
    pairs = list(itertools.combinations(views, 2))
    return sum(_column_correlations(first, second) for first, second in pairs) / len(pairs)


def _correlation_reliability(views: Sequence[NDArray[Any]]) -> NDArray[np.float64]:
    """Average leave-one-view-out correlations, vectorized over features."""
    total = np.zeros_like(views[0], dtype=np.float64)
    for view in views:
        total += view
    reliability = np.zeros(views[0].shape[1], dtype=np.float64)
    for view in views:
        other_mean = (total - view) / (len(views) - 1)
        reliability += _column_correlations(view, other_mean)
    return reliability / len(views)


def _column_correlations(
    X: NDArray[Any], Y: NDArray[Any],
) -> NDArray[np.float64]:
    """Pearson correlations with zero for constant columns."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    denominator = np.sqrt(np.sum(X * X, axis=0) * np.sum(Y * Y, axis=0))
    correlations = np.zeros(X.shape[1], dtype=np.float64)
    np.divide(np.sum(X * Y, axis=0), denominator, out=correlations, where=denominator > 0)
    return np.clip(correlations, -1, 1)


def _validate_cv(cv: int | None, matching: str) -> None:
    """Check cross-validated reliability settings shared by the encoders and the refiner."""
    if cv is not None and (isinstance(cv, (bool, np.bool_)) or not isinstance(cv, Integral) or cv < 2):
        raise ValueError("output_reweighting_cv must be an integer of at least 2 or None.")
    if matching not in ("hungarian", "index", "soft"):
        raise ValueError("output_reweighting_matching must be 'hungarian', 'index', or 'soft'.")


def _cross_validated_reliability(
    fit_fold: Callable[[NDArray[np.bool_]], Callable[[NDArray[np.bool_]], NDArray[Any]]],
    full: NDArray[Any], sample_ids: NDArray[Any], view_ids: NDArray[Any], *,
    method: str, cv: int, matching: str, random_state: int | None,
) -> NDArray[np.float64]:
    """Out-of-sample reliability of the columns of full, estimated by cross-validation.

    Samples are split into cv folds. For each fold, fit_fold(train_rows)
    refits the model on the other folds and returns a function mapping a
    boolean row mask to that model's outputs. The reliability of each fold
    output on the held-out rows is transferred to the columns of full (the
    model fitted on all rows) by matching outputs on the held-out rows, and
    averaged over folds.
    """
    samples = np.unique(sample_ids)
    if cv > len(samples):
        raise ValueError("output_reweighting_cv cannot exceed the number of samples.")
    folds = np.array_split(np.random.default_rng(random_state).permutation(samples), cv)
    reliability = np.zeros(full.shape[1])
    for held_out in folds:
        test = np.isin(sample_ids, held_out)
        outputs = np.asarray(fit_fold(~test)(test), dtype=np.float64)
        fold_reliability = FeatureReweighting(method=method).fit(
            outputs, sample_ids=sample_ids[test], view_ids=view_ids[test],
        ).reliability_
        reliability += _transfer_reliability(fold_reliability, outputs, full[test], matching)
    return reliability / cv


def _transfer_reliability(
    reliability: NDArray[np.float64], fold: NDArray[Any], full: NDArray[Any], matching: str,
) -> NDArray[np.float64]:
    """Assign the reliability of fold outputs to the corresponding full-data outputs.

    "index" assumes the same column order. "hungarian" pairs columns one to
    one by maximal absolute correlation on the same rows. "soft" gives full
    column c the average sum_j C[j, c]^2 r_j / sum_j C[j, c]^2: with mutually
    uncorrelated outputs, column c is approximately sum_j C[j, c] times fold
    column j, and this is that combination's reliability.
    """
    if matching == "index":
        return reliability
    similarity = np.abs(_correlation_matrix(fold, full))
    if matching == "hungarian":
        from scipy.optimize import linear_sum_assignment

        fold_columns, full_columns = linear_sum_assignment(-similarity)
        transferred = np.zeros(full.shape[1])
        transferred[full_columns] = reliability[fold_columns]
        return transferred
    shares = similarity ** 2
    totals = shares.sum(axis=0)
    return np.divide(reliability @ shares, totals, out=np.zeros(full.shape[1]), where=totals > 0)


def _correlation_matrix(X: NDArray[Any], Y: NDArray[Any]) -> NDArray[np.float64]:
    """Pearson correlations between all columns of X and Y, zero for constant columns."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    X = np.divide(X, np.linalg.norm(X, axis=0), out=np.zeros_like(X, dtype=np.float64),
                  where=np.linalg.norm(X, axis=0) > 0)
    Y = np.divide(Y, np.linalg.norm(Y, axis=0), out=np.zeros_like(Y, dtype=np.float64),
                  where=np.linalg.norm(Y, axis=0) > 0)
    return np.clip(X.T @ Y, -1, 1)
