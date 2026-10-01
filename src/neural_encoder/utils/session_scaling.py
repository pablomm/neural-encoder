"""Per-session standardization of measurements."""

from numbers import Integral
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

__all__ = ["SessionStandardScaler"]


class SessionStandardScaler(TransformerMixin, BaseEstimator):
    """Standardize every feature within each acquisition session.

    Measurements recorded in one session share offset and gain changes that
    are unrelated to the samples. For each session seen during fit, the mean
    and standard deviation (ddof=0) of every feature are estimated from that
    session's training measurements. Transform applies them to any
    measurement of the same session, so held-out measurements are scaled
    with statistics learned from training data only. Sessions with fewer
    than min_samples training measurements, or not seen during fit, use
    statistics pooled over all training measurements.

    Parameters
    ----------
    with_mean : bool, default=True
        Subtract the session mean.
    with_std : bool, default=True
        Divide by the session standard deviation. Zero deviations are
        replaced by 1.
    min_samples : int, default=50
        Minimum training measurements for session-specific statistics.
    dtype : numpy floating dtype, default=numpy.float32
        Working and output dtype. Inputs are copied and never modified.

    Attributes
    ----------
    sessions_ : ndarray of shape (n_sessions,)
        Sessions with their own statistics.
    means_ : ndarray of shape (n_sessions, n_features)
        Per-session feature means, in the order of ``sessions_``.
    scales_ : ndarray of shape (n_sessions, n_features)
        Per-session feature standard deviations, zeros replaced by 1.
    mean_ : ndarray of shape (n_features,)
        Pooled training means, used for other sessions.
    scale_ : ndarray of shape (n_features,)
        Pooled training standard deviations, used for other sessions.
    n_features_in_ : int
        Number of features observed during fit.

    Notes
    -----
    Session labels are matched by value, not used as indices: any
    integers (e.g. starting at 1, or non-contiguous) or strings work, with
    one label per row. They are required by both fit and transform; in a
    scikit-learn Pipeline they must be routed explicitly. Inputs must be
    finite.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1., 0.], [3., 2.], [10., 5.], [14., 9.]])
    >>> scaler = SessionStandardScaler(min_samples=2)
    >>> scaler.fit_transform(X, sessions=[0, 0, 1, 1])
    array([[-1., -1.],
           [ 1.,  1.],
           [-1., -1.],
           [ 1.,  1.]], dtype=float32)
    """

    def __init__(
        self, *, with_mean: bool = True, with_std: bool = True, min_samples: int = 50,
        dtype: type = np.float32,
    ) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.min_samples = min_samples
        self.dtype = dtype

    def _validate_parameters(self) -> None:
        for name in ("with_mean", "with_std"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean.")
        if not isinstance(self.min_samples, Integral) or isinstance(self.min_samples, bool) or self.min_samples < 1:
            raise ValueError("min_samples must be a positive integer.")
        if np.dtype(self.dtype).kind != "f":
            raise ValueError("dtype must be a floating dtype.")

    def _check(self, X: ArrayLike, sessions: ArrayLike) -> tuple[NDArray, NDArray]:
        X = check_array(X, dtype=self.dtype, copy=True)
        sessions = np.asarray(sessions).ravel()
        if len(sessions) != len(X):
            raise ValueError("sessions must have one entry per measurement.")
        return X, sessions

    def fit(self, X: ArrayLike, y: Any = None, *, sessions: ArrayLike) -> Self:
        """Estimate per-session and pooled statistics; y is ignored."""
        self._validate_parameters()
        X, sessions = self._check(X, sessions)
        self.mean_, self.scale_ = X.mean(axis=0), _safe_scale(X.std(axis=0), X.dtype)
        kept, means, scales = [], [], []
        for session in np.unique(sessions):
            rows = sessions == session
            if rows.sum() >= self.min_samples:
                kept.append(session)
                means.append(X[rows].mean(axis=0))
                scales.append(_safe_scale(X[rows].std(axis=0), X.dtype))
        n_features = X.shape[1]
        self.sessions_ = np.asarray(kept)
        self.means_ = np.asarray(means, dtype=X.dtype).reshape(len(kept), n_features)
        self.scales_ = np.asarray(scales, dtype=X.dtype).reshape(len(kept), n_features)
        self.n_features_in_ = n_features
        return self

    def transform(self, X: ArrayLike, *, sessions: ArrayLike) -> NDArray:
        """Standardize a copy of X with the statistics of each row's session."""
        check_is_fitted(self, ["means_", "scales_"])
        X, sessions = self._check(X, sessions)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        for session in np.unique(sessions):
            rows = sessions == session
            index = np.flatnonzero(self.sessions_ == session)
            if index.size:
                mean, scale = self.means_[index[0]], self.scales_[index[0]]
            else:
                mean, scale = self.mean_, self.scale_
            if self.with_mean:
                X[rows] -= mean
            if self.with_std:
                X[rows] /= scale
        return X

    def fit_transform(self, X: ArrayLike, y: Any = None, *, sessions: ArrayLike) -> NDArray:
        """Fit on X and return its standardized copy."""
        return self.fit(X, sessions=sessions).transform(X, sessions=sessions)


def _safe_scale(std: NDArray, dtype: np.dtype) -> NDArray:
    return np.where(std > 0, std, 1.0).astype(dtype)
