"""Multiview CCA distilled into a single affine projector."""

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted, validate_data

from ..utils.split_views import split_views

__all__ = ["DistilledMCCA"]


class DistilledMCCA(TransformerMixin, BaseEstimator):
    """Fit MCCA across views and distill their average into one ridge map.

    Every fitted view projector is applied to every training measurement.
    Their mean, including view-specific centering, is the regression target.
    Transform uses only the resulting affine map, without requiring view IDs.
    All views must have corresponding features in the same input space.

    Parameters
    ----------
    n_components : int, default=128
        Number of output components, at most the input feature count.
    mcca_reg : float, default=0.1
        MCCA covariance regularization, between 0 and 1.
    distill_reg : float, default=0.5
        Nonnegative ridge penalty for the distilled projector.
    impute : {"mean", "zeros", "discard"}, default="mean"
        Missing-view handling when fitting from a measurement matrix.
        Mean imputation uses the sample's available views. This affects MCCA
        fitting only: distillation always uses the original measurements.
        Has no effect on fit_views, which requires complete aligned views.
    fit_intercept : bool, default=True
        Whether ridge distillation learns an intercept. MCCA always centers
        each view independently.

    Attributes
    ----------
    mcca_ : neural_encoder.mcca.MCCA
        Fitted multiview model.
    coef_ : ndarray of shape (n_components, n_features)
        Distilled projection coefficients.
    intercept_ : ndarray of shape (n_components,)
        Distilled offset, or zeros when fit_intercept is False.
    n_features_in_ : int
        Input feature count.
    n_views_ : int
        Number of fitted views.
    n_samples_ : int
        Number of aligned samples used for MCCA.

    Examples
    --------
    >>> model = DistilledMCCA(n_components=2)
    >>> X = [[1., 2.], [2., 3.], [3., 1.], [4., 2.], [2., 4.], [3., 5.]]
    >>> Z = model.fit_transform(X, sample_ids=[0, 0, 1, 1, 2, 2])
    >>> Z.shape
    (6, 2)
    """

    def __init__(
        self,
        *,
        n_components: int = 128,
        mcca_reg: float = 0.1,
        distill_reg: float = 0.5,
        impute: Literal["mean", "zeros", "discard"] = "mean",
        fit_intercept: bool = True,
    ) -> None:
        self.n_components = n_components
        self.mcca_reg = mcca_reg
        self.distill_reg = distill_reg
        self.impute = impute
        self.fit_intercept = fit_intercept

    def _validate_parameters(self) -> None:
        if isinstance(self.n_components, (bool, np.bool_)) or not isinstance(self.n_components, Integral) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if not isinstance(self.mcca_reg, Real) or not np.isfinite(self.mcca_reg) or not 0 <= self.mcca_reg <= 1:
            raise ValueError("mcca_reg must be between 0 and 1.")
        if not isinstance(self.distill_reg, Real) or not np.isfinite(self.distill_reg) or self.distill_reg < 0:
            raise ValueError("distill_reg must be finite and nonnegative.")
        if self.impute not in ("mean", "zeros", "discard"):
            raise ValueError("impute must be 'mean', 'zeros', or 'discard'.")
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be a boolean.")

    def fit(
        self, X: ArrayLike, y: Any = None, *,
        sample_ids: ArrayLike, view_ids: ArrayLike | None = None,
    ) -> Self:
        """Fit from measurements, using sample identities to align views.

        Omitted view IDs are inferred by occurrence within each sample.
        The optional y argument is ignored.
        """
        self._validate_parameters()
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        views = split_views(X, sample_ids, view_ids, impute=self.impute)
        return self._fit_projector(views, X)

    def fit_views(self, views: Sequence[ArrayLike]) -> Self:
        """Fit aligned, finite views and distill on their concatenation."""
        self._validate_parameters()
        if len(views) < 2:
            raise ValueError("At least two views are required.")
        arrays = [validate_data(self, views[0], dtype=np.float64, ensure_min_samples=2)]
        arrays.extend(
            validate_data(self, view, reset=False, dtype=np.float64, ensure_min_samples=2)
            for view in views[1:]
        )
        return self._fit_projector(arrays, np.concatenate(arrays, axis=0))

    def _fit_projector(self, views: Sequence[NDArray[Any]], X: NDArray[Any]) -> Self:
        from ..mcca import MCCA

        if len(views) < 2:
            raise ValueError("At least two views are required.")
        if any(view.shape != views[0].shape for view in views):
            raise ValueError("All views must have the same shape and aligned samples.")
        if views[0].shape[0] < 2:
            raise ValueError("At least two aligned samples are required for MCCA.")
        if self.n_components > self.n_features_in_:
            raise ValueError("n_components must not exceed the input feature count.")
        mcca = MCCA(n_components=self.n_components, regs=self.mcca_reg, center=True)
        mcca.fit(list(views))
        # Average affine maps before projecting to avoid a large stack of
        # per-view targets. In particular, average mu_v @ W_v, not mu and W separately.
        mean_loading = np.mean(mcca.loadings_, axis=0)
        mean_offset = -np.mean([
            mean @ loading for mean, loading in zip(mcca.means_, mcca.loadings_)
        ], axis=0)
        target = X @ mean_loading + mean_offset
        ridge = Ridge(alpha=self.distill_reg, fit_intercept=self.fit_intercept, solver="svd")
        ridge.fit(X, target)
        self.mcca_ = mcca
        self.coef_ = np.asarray(ridge.coef_).reshape(self.n_components, self.n_features_in_)
        self.intercept_ = np.broadcast_to(ridge.intercept_, (self.n_components,)).copy()
        self.n_views_ = len(views)
        self.n_samples_ = views[0].shape[0]
        return self

    def transform(self, X: ArrayLike) -> NDArray[Any]:
        """Apply the distilled affine projection to individual measurements."""
        check_is_fitted(self, ["coef_", "intercept_"])
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        return X @ self.coef_.T + self.intercept_

    def fit_transform(
        self, X: ArrayLike, y: Any = None, *,
        sample_ids: ArrayLike, view_ids: ArrayLike | None = None,
    ) -> NDArray[Any]:
        """Fit the projector and transform every input measurement."""
        return self.fit(X, y, sample_ids=sample_ids, view_ids=view_ids).transform(X)
