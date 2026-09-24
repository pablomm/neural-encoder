"""Feature weighting from reliability across repeated measurements."""

from collections.abc import Callable, Sequence
from numbers import Real
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ..utils.split_views import split_views

__all__ = ["FeatureReweighting"]


class FeatureReweighting(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Weight corresponding features by their reliability across views.

    Reliability is the mean Pearson correlation between each view and the
    mean of all other views, computed separately for each feature across
    aligned samples. Negative reliability is clipped to zero before adding
    eps. Transform multiplies each feature by its fitted weight.

    Parameters
    ----------
    method : {"correlation"}, default="correlation"
        Estimate reliability using leave-one-view-out Pearson correlation.
    weighting : {"linear", "sqrt"} or callable, default="linear"
        Use max(reliability, 0) + eps directly, its square root, or pass this
        vector to a callable. The callable must return a finite real vector
        of the same shape. Its output is used without further clipping.
    eps : float, default=1e-6
        Nonnegative offset added after clipping negative reliability. This
        is a weight floor, not a correlation-denominator regularizer.

    Attributes
    ----------
    reliability_ : ndarray of shape (n_features,)
        Mean leave-one-view-out correlations, before clipping and weighting.
        Comparisons involving a constant feature contribute zero.
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
        method: Literal["correlation"] = "correlation",
        weighting: Literal["linear", "sqrt"] | Callable[[NDArray[np.float64]], ArrayLike] = "linear",
        eps: float = 1e-6,
    ) -> None:
        self.method = method
        self.weighting = weighting
        self.eps = eps

    def _validate_parameters(self) -> None:
        if self.method != "correlation":
            raise ValueError("method must be 'correlation'.")
        if not callable(self.weighting) and self.weighting not in ("linear", "sqrt"):
            raise ValueError("weighting must be 'linear', 'sqrt', or a callable.")
        if not isinstance(self.eps, Real) or not np.isfinite(self.eps) or self.eps < 0:
            raise ValueError("eps must be a finite nonnegative number.")

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
        self.reliability_ = _correlation_reliability(views)
        weights = np.maximum(self.reliability_, 0) + self.eps
        if callable(self.weighting):
            weights = np.asarray(self.weighting(weights))
            if weights.shape != self.reliability_.shape or weights.dtype.kind not in "iuf":
                raise ValueError("weighting callable must return a real vector with one value per feature.")
            weights = weights.astype(np.float64, copy=True)
            if not np.isfinite(weights).all():
                raise ValueError("weighting callable must return only finite values.")
        elif self.weighting == "sqrt":
            weights = np.sqrt(weights)
        self.weights_ = weights
        self.n_views_ = len(views)
        self.n_samples_ = views[0].shape[0]
        return self

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
