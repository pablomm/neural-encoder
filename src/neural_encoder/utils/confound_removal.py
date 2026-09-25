"""Remove linear confound effects from sample-by-feature measurements."""

from numbers import Real
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

__all__ = ["ConfoundRemover"]


class ConfoundRemover(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Remove confound effects estimated by ridge regression.

    Given measurements X and confound design matrix C, fit minimizes
    ||X - C B||_F^2 + alpha ||B||_F^2. Transform returns X - C B using the
    fitted coefficients. Confounds may be continuous or dummy-coded variables.

    Parameters
    ----------
    alpha : float, default=1e-3
        Nonnegative ridge penalty. Zero uses unregularized least squares.
    dtype : numpy floating dtype, default=numpy.float32
        Working and output dtype, either float32 or float64.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features, n_confounds)
        Regression coefficients. The confound prediction is C @ coef_.T.
    n_features_in_ : int
        Number of measurement features fitted.
    n_confounds_in_ : int
        Number of confound columns fitted.

    Notes
    -----
    The y argument supplies the confound matrix in fit, transform, and
    fit_transform. Rows must align with X, and confound columns must retain
    the same meaning and order across calls. Fit on training data only.

    No intercept is added and neither input is centered. Include a constant
    confound column to remove an intercept; its coefficient is regularized
    like every other confound coefficient. Inputs are never modified.

    Transform requires confounds for the new measurements. A standard sklearn
    Pipeline does not automatically forward its target y to transform.

    Examples
    --------
    >>> remover = ConfoundRemover()
    >>> X = [[1., 2.], [2., 3.], [4., 5.]]
    >>> C = [[1., 0.], [1., 0.], [0., 1.]]
    >>> residuals = remover.fit_transform(X, C)
    >>> residuals.shape
    (3, 2)
    """

    def __init__(self, *, alpha: float = 1e-3, dtype: DTypeLike = np.float32) -> None:
        self.alpha = alpha
        self.dtype = dtype

    def _validate_parameters(self) -> None:
        if not isinstance(self.alpha, Real) or not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be a finite nonnegative number.")
        if np.dtype(self.dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype must be float32 or float64.")

    def _validate_confounds(self, y: ArrayLike | None, n_samples: int) -> NDArray[Any]:
        if y is None:
            raise ValueError("y must provide a two-dimensional confound matrix.")
        confounds = check_array(y, dtype=self.dtype, copy=True)
        if confounds.shape[0] != n_samples:
            raise ValueError("X and the confound matrix y must have the same number of rows.")
        return confounds

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Estimate confound coefficients from training measurements X and y."""
        self._validate_parameters()
        X = validate_data(self, X, dtype=self.dtype, copy=True)
        confounds = self._validate_confounds(y, X.shape[0])
        # SVD also supports rank-deficient designs when alpha is zero.
        regression = Ridge(alpha=self.alpha, fit_intercept=False, solver="svd")
        regression.fit(confounds, X)
        self.coef_ = np.asarray(regression.coef_, dtype=self.dtype).reshape(
            X.shape[1], confounds.shape[1],
        )
        self.n_confounds_in_ = confounds.shape[1]
        return self

    def transform(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray[Any]:
        """Subtract fitted confound effects using the supplied design matrix y."""
        check_is_fitted(self, ["coef_", "n_confounds_in_"])
        X = validate_data(self, X, reset=False, dtype=self.dtype, copy=True)
        confounds = self._validate_confounds(y, X.shape[0])
        if confounds.shape[1] != self.n_confounds_in_:
            raise ValueError(
                f"Expected {self.n_confounds_in_} confound columns, "
                f"received {confounds.shape[1]}."
            )
        return X - confounds @ self.coef_.T

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray[Any]:
        """Fit and remove confound effects from the training measurements."""
        return self.fit(X, y).transform(X, y)
