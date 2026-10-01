"""Ridge regression from each view to the mean of the other views."""

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_is_fitted, validate_data

from ..utils.split_views import split_views

__all__ = ["CrossViewRidge"]


class CrossViewRidge(TransformerMixin, BaseEstimator):
    """Denoise measurements by predicting the mean of their other views.

    Every training measurement is paired with the mean of the other observed
    views of the same sample (its leave-one-view-out mean). A multi-output
    ridge regression from measurement to that mean is fitted, with the
    penalty selected by efficient leave-one-out cross-validation (RidgeCV).
    The fitted affine map estimates the component of a single measurement
    that is shared across views, shrinking directions dominated by
    view-specific noise. Transform needs no sample or view IDs, and outputs
    have the input feature count.

    Parameters
    ----------
    alphas : array-like of shape (n_alphas,), default=numpy.logspace(-1, 7, 17)
        Positive ridge penalties to select from.
    fit_intercept : bool, default=True
        Whether the regression learns an intercept.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features, n_features)
        Regression coefficients. Transform computes X @ coef_.T + intercept_.
    intercept_ : ndarray of shape (n_features,)
        Offset, or zeros when fit_intercept is False.
    alpha_ : float
        Selected ridge penalty.
    n_features_in_ : int
        Input feature count.
    n_views_ : int
        Number of views.
    n_samples_ : int
        Samples with at least two observed views, which define the targets.
    n_pairs_ : int
        Measurement-target pairs used in the regression.

    Notes
    -----
    Samples observed in a single view have no target and are not used.
    All views must have corresponding features in the same input space.
    The target is a noisy estimate of the shared signal; its noise is
    independent of the input measurement's when view noise is independent,
    so the fit is not biased towards reproducing the input's noise.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> signal = rng.normal(size=(50, 4))
    >>> views = [signal + rng.normal(scale=0.5, size=signal.shape) for _ in range(3)]
    >>> denoiser = CrossViewRidge().fit_views(views)
    >>> denoiser.transform(views[0]).shape
    (50, 4)
    """

    def __init__(self, *, alphas: ArrayLike = np.logspace(-1, 7, 17), fit_intercept: bool = True) -> None:
        self.alphas = alphas
        self.fit_intercept = fit_intercept

    def _validate_parameters(self) -> None:
        alphas = np.atleast_1d(np.asarray(self.alphas, dtype=np.float64))
        if alphas.ndim != 1 or alphas.size == 0 or not np.all(np.isfinite(alphas) & (alphas > 0)):
            raise ValueError("alphas must be a nonempty sequence of positive finite values.")
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be a boolean.")

    def fit(
        self, X: ArrayLike, y: Any = None, *, sample_ids: ArrayLike, view_ids: ArrayLike | None = None,
    ) -> Self:
        """Fit from measurements, using sample identities to group views.

        Omitted view IDs are inferred by occurrence within each sample.
        Missing views are allowed; samples need two views to contribute.
        The optional y argument is ignored.
        """
        self._validate_parameters()
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        return self._fit_stacked(np.stack(split_views(X, sample_ids, view_ids, impute=None)))

    def fit_views(self, views: Sequence[ArrayLike]) -> Self:
        """Fit from at least two aligned, finite, equally shaped matrices.

        Every row must represent the same sample across all views.
        """
        self._validate_parameters()
        if len(views) < 2:
            raise ValueError("At least two views are required.")
        arrays = [validate_data(self, views[0], dtype=np.float64, ensure_min_samples=2)]
        arrays.extend(
            validate_data(self, view, reset=False, dtype=np.float64, ensure_min_samples=2)
            for view in views[1:]
        )
        if any(view.shape != arrays[0].shape for view in arrays):
            raise ValueError("All views must have the same shape and aligned samples.")
        return self._fit_stacked(np.stack(arrays))

    def _fit_stacked(self, views: NDArray[np.float64]) -> Self:
        """Fit from a (n_views, n_samples, n_features) array; NaN rows are missing views."""
        present = ~np.isnan(views[..., 0])
        count = present.sum(axis=0)
        if np.count_nonzero(count >= 2) < 2:
            raise ValueError("At least two samples observed in two or more views are required.")
        total = np.nansum(views, axis=0)
        inputs, targets = [], []
        for view, observed in zip(views, present):
            rows = observed & (count >= 2)
            inputs.append(view[rows])
            targets.append((total[rows] - view[rows]) / (count[rows] - 1)[:, None])
        ridge = RidgeCV(alphas=np.atleast_1d(self.alphas), fit_intercept=self.fit_intercept)
        ridge.fit(np.vstack(inputs), np.vstack(targets))
        self.coef_ = np.asarray(ridge.coef_)
        self.intercept_ = np.broadcast_to(ridge.intercept_, (self.n_features_in_,)).astype(np.float64)
        self.alpha_ = float(ridge.alpha_)
        self.n_views_ = len(views)
        self.n_samples_ = int(np.count_nonzero(count >= 2))
        self.n_pairs_ = int(sum(len(rows) for rows in inputs))
        return self

    def transform(self, X: ArrayLike) -> NDArray[Any]:
        """Apply the fitted affine map to individual measurements."""
        check_is_fitted(self, ["coef_", "intercept_"])
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        return X @ self.coef_.T + self.intercept_

    def fit_transform(
        self, X: ArrayLike, y: Any = None, *, sample_ids: ArrayLike, view_ids: ArrayLike | None = None,
    ) -> NDArray[Any]:
        """Fit the regression and transform every input measurement."""
        return self.fit(X, y, sample_ids=sample_ids, view_ids=view_ids).transform(X)
