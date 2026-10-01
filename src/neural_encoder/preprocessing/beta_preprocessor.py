"""Preprocessing of single-trial fMRI response estimates (betas)."""

from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

__all__ = ["BetaPreprocessor"]


class BetaPreprocessor(TransformerMixin, BaseEstimator):
    """Scale, clip, center, and optionally normalize single-trial fMRI betas.

    Preprocessing of single-trial GLM response estimates applied in
    *Platonic Representations in the Human Brain: Unsupervised Recovery of
    Universal Geometry* [1]_. Rows are trials and columns are voxels or
    vertices.

    Applies the following steps: replace NaNs, scale, clip, center samples,
    center features, scale features, normalize samples. Clipping bounds and
    feature statistics are learned from training data.

    Parameters
    ----------
    scaling : float or None, default=300.0
        Positive divisor applied before clipping. None skips scaling. Use
        300.0 for the Natural Scenes Dataset (NSD) [2]_, which stores betas
        multiplied by 300; use None for betas already in percent signal
        change.
    quantile_clip : float, pair of floats, or None, default=0.0005
        Quantile probabilities estimated over all scaled training values.
        A scalar a specifies (a, 1-a), with 0 <= a < 0.5. A pair specifies
        (lower, upper), with 0 <= lower < upper <= 1. None skips quantile
        clipping. Cannot be combined with clip_bounds.
    clip_bounds : pair of floats or None, default=None
        Fixed lower and upper clipping values, in scaled units.
    sample_centering : bool, default=False
        Subtract each sample's mean across features, independently per row.
    feature_centering : bool, default=True
        Subtract feature means learned from the clipped, sample-centered
        training data. Means are reused unchanged during transform.
    feature_scaling : bool, default=False
        Divide by training feature standard deviations (ddof=0), estimated
        after clipping and optional sample centering. Independent of feature
        centering. Constant features use a divisor of 1.
    normalize : bool, default=False
        Normalize each row to unit L2 norm after feature scaling. Zero rows stay zero.
    fill_value : float or None, default=0.0
        Replace NaNs with this finite constant before scaling. None rejects
        NaNs. Infinite input values are always rejected.
    dtype : numpy floating dtype, default=numpy.float32
        Working and output dtype. Inputs are copied and never modified.

    Attributes
    ----------
    clip_bounds_ : ndarray of shape (2,) or None
        Learned quantile values or fixed bounds; None when clipping is disabled.
    feature_mean_ : ndarray of shape (n_features,) or None
        Training feature means, or None when feature centering is disabled.
    feature_scale_ : ndarray of shape (n_features,) or None
        Training feature standard deviations, with zeros replaced by 1,
        or None when feature scaling is disabled.
    n_features_in_ : int
        Number of features observed during fit.

    References
    ----------
    .. [1] Marcos-Manchon et al. (2026).
       https://arxiv.org/abs/2605.20496
    .. [2] Allen, E. J., St-Yves, G., Wu, Y., et al. (2022).
       A massive 7T fMRI dataset to bridge cognitive neuroscience and
       artificial intelligence. Nature Neuroscience, 25, 116-126.
       https://doi.org/10.1038/s41593-021-00962-x

    Examples
    --------
    >>> preprocessor = BetaPreprocessor(
    ...     scaling=300.0, quantile_clip=0.0005, feature_centering=True
    ... )
    >>> train = preprocessor.fit_transform([[0., 300.], [600., 900.]])
    >>> test = preprocessor.transform([[300., 600.]])
    """

    def __init__(
        self,
        *,
        scaling=300.0,
        quantile_clip=0.0005,
        clip_bounds=None,
        sample_centering=False,
        feature_centering=True,
        feature_scaling=False,
        normalize=False,
        fill_value=0.0,
        dtype=np.float32,
    ):
        self.scaling = scaling
        self.quantile_clip = quantile_clip
        self.clip_bounds = clip_bounds
        self.sample_centering = sample_centering
        self.feature_centering = feature_centering
        self.feature_scaling = feature_scaling
        self.normalize = normalize
        self.fill_value = fill_value
        self.dtype = dtype

    @staticmethod
    def _pair(value, name):
        pair = np.asarray(value, dtype=np.float64)
        if pair.shape != (2,) or not np.isfinite(pair).all():
            raise ValueError(f"{name} must contain two finite numbers.")
        if pair[0] > pair[1]:
            raise ValueError(f"{name} lower bound must not exceed its upper bound.")
        return pair

    def _validate_parameters(self):
        if not np.issubdtype(np.dtype(self.dtype), np.floating):
            raise ValueError("dtype must be a floating-point dtype.")
        if self.scaling is not None and (
            not isinstance(self.scaling, Real) or not np.isfinite(self.scaling) or self.scaling <= 0
        ):
            raise ValueError("scaling must be a finite positive number or None.")
        if self.fill_value is not None and (
            not isinstance(self.fill_value, Real) or not np.isfinite(self.fill_value)
        ):
            raise ValueError("fill_value must be a finite number or None.")
        for name in ("sample_centering", "feature_centering", "feature_scaling", "normalize"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean.")
        if self.quantile_clip is not None and self.clip_bounds is not None:
            raise ValueError("Specify only one of quantile_clip and clip_bounds.")
        if self.clip_bounds is not None:
            self._pair(self.clip_bounds, "clip_bounds")
        if self.quantile_clip is None:
            return None
        value = self.quantile_clip
        if isinstance(value, Real):
            value = (value, 1 - value)
        probabilities = self._pair(value, "quantile_clip")
        if not 0 <= probabilities[0] < probabilities[1] <= 1:
            raise ValueError("quantile_clip must satisfy 0 <= lower < upper <= 1.")
        return probabilities

    def _prepare(self, X):
        # check_array performs shape/type validation; handle NaNs explicitly
        # so infinities are never silently converted to large finite values.
        X = check_array(X, dtype=self.dtype, copy=True, ensure_all_finite=False)
        if np.isinf(X).any():
            raise ValueError("X contains infinite values.")
        if np.isnan(X).any():
            if self.fill_value is None:
                raise ValueError("X contains NaNs; set fill_value to replace them.")
            X[np.isnan(X)] = self.fill_value
        if self.scaling is not None:
            X /= self.scaling
        if not np.isfinite(X).all():
            raise ValueError("Preprocessing produced nonfinite values; check dtype and scaling.")
        return X

    def _clip_and_center_samples(self, X):
        if self.clip_bounds_ is not None:
            np.clip(X, *self.clip_bounds_, out=X)
        if self.sample_centering:
            X -= X.mean(axis=1, keepdims=True)
        return X

    def fit(self, X, y=None):
        """Learn clipping bounds and feature statistics from training measurements."""
        probabilities = self._validate_parameters()
        X = self._prepare(X)
        self.n_features_in_ = X.shape[1]
        if probabilities is not None:
            self.clip_bounds_ = np.quantile(X, probabilities)
        elif self.clip_bounds is not None:
            self.clip_bounds_ = self._pair(self.clip_bounds, "clip_bounds").copy()
        else:
            self.clip_bounds_ = None
        X = self._clip_and_center_samples(X)
        self.feature_mean_ = X.mean(axis=0) if self.feature_centering else None
        self.feature_scale_ = None
        if self.feature_scaling:
            self.feature_scale_ = X.std(axis=0, ddof=0, dtype=np.float64)
            self.feature_scale_[self.feature_scale_ == 0] = 1.0
        return self

    def transform(self, X):
        """Apply fitted preprocessing without modifying X or fitted statistics."""
        check_is_fitted(self, ["n_features_in_", "clip_bounds_", "feature_mean_", "feature_scale_"])
        X = self._prepare(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but BetaPreprocessor is expecting "
                f"{self.n_features_in_} features as input."
            )
        X = self._clip_and_center_samples(X)
        if self.feature_mean_ is not None:
            X -= self.feature_mean_
        if self.feature_scale_ is not None:
            X /= self.feature_scale_
        if self.normalize:
            # Rescale first to avoid overflow when calculating squared norms.
            scale = np.max(np.abs(X), axis=1, keepdims=True)
            np.divide(X, scale, out=X, where=scale != 0)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            np.divide(X, norms, out=X, where=norms != 0)
        return X
