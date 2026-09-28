"""Compose feature reweighting, PCA, and distilled MCCA."""

from collections.abc import Mapping
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted, validate_data

from ..utils.split_views import _encode_ids, _views_by_occurrence
from .distilled_mcca import DistilledMCCA
from .feature_reweighting import FeatureReweighting

__all__ = ["LinearEncoder"]

if TYPE_CHECKING:
    import torch


class LinearEncoder(TransformerMixin, BaseEstimator):
    """Learn an affine encoder from repeated or multiview measurements.

    Fits feature reweighting, PCA, and distilled MCCA in that order. Each
    stage can use its default estimator, a supplied estimator to clone, or
    None to skip the stage. Preprocessing is performed outside this encoder.

    Parameters
    ----------
    feature_reweighting : {"default", None} or FeatureReweighting, default="default"
        Feature reliability stage. The default is FeatureReweighting().
    pca : {"default", None} or sklearn.decomposition.PCA, default="default"
        Dimensionality reduction. The default uses 768 components. Supports
        PCA whitening; other dimensionality-reduction classes are not accepted.
    distilled_mcca : {"default", None} or DistilledMCCA, default="default"
        Multiview projection. The default uses 128 components.
    feature_reweighting_kwargs, pca_kwargs, distilled_mcca_kwargs : dict or None
        Constructor overrides for the corresponding default stage. Supplying
        kwargs with an explicit estimator or disabled stage raises an error.
        Component counts must be valid for the data; they are not reduced
        automatically.
    shuffle_views : bool, default=False
        Independently permute view assignments within each sample once before
        fitting. Both multiview stages use the same assignments. Measurements
        remain in their original row order, and no measurements are imputed
        by this operation.
    random_state : int or None, default=None
        Nonnegative seed for view shuffling and the default PCA, unless its
        random_state is overridden in pca_kwargs. Explicit estimator instances
        retain their own random-state configuration.

    Attributes
    ----------
    feature_reweighting_ : FeatureReweighting or None
        Fitted feature reweighting stage.
    pca_ : sklearn.decomposition.PCA or None
        Fitted PCA stage.
    distilled_mcca_ : DistilledMCCA or None
        Fitted distilled MCCA stage.
    coef_ : ndarray of shape (n_components, n_features)
        Combined coefficients. Transform computes X @ coef_.T + intercept_.
    intercept_ : ndarray of shape (n_components,)
        Combined offset, accounting for PCA centering and MCCA distillation.
    n_features_in_ : int
        Number of input features.
    n_components_ : int
        Output dimensionality.

    Notes
    -----
    Reliability uses complete samples; PCA uses all training measurements;
    MCCA handles missing views according to its impute parameter. PCA is
    unweighted across samples. Corresponding features are required across
    views. With no dimensionality reduction, the combined coefficient matrix
    can be as large as n_features by n_features.

    Examples
    --------
    >>> encoder = LinearEncoder(
    ...     pca_kwargs={"n_components": 2},
    ...     distilled_mcca_kwargs={"n_components": 1},
    ... )
    >>> X = [[1., 2.], [2., 3.], [3., 1.], [4., 2.], [2., 4.], [3., 5.]]
    >>> encoder.fit_transform(X, sample_ids=[0, 0, 1, 1, 2, 2]).shape
    (6, 1)
    """

    def __init__(
        self,
        *,
        feature_reweighting: Literal["default"] | FeatureReweighting | None = "default",
        pca: Literal["default"] | PCA | None = "default",
        distilled_mcca: Literal["default"] | DistilledMCCA | None = "default",
        feature_reweighting_kwargs: Mapping[str, Any] | None = None,
        pca_kwargs: Mapping[str, Any] | None = None,
        distilled_mcca_kwargs: Mapping[str, Any] | None = None,
        shuffle_views: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.feature_reweighting = feature_reweighting
        self.pca = pca
        self.distilled_mcca = distilled_mcca
        self.feature_reweighting_kwargs = feature_reweighting_kwargs
        self.pca_kwargs = pca_kwargs
        self.distilled_mcca_kwargs = distilled_mcca_kwargs
        self.shuffle_views = shuffle_views
        self.random_state = random_state

    def fit(
        self, X: ArrayLike, y: Any = None, *,
        sample_ids: ArrayLike | None = None, view_ids: ArrayLike | None = None,
    ) -> Self:
        """Fit all enabled stages on training measurements.

        sample_ids is required when either multiview stage is enabled.
        Omitted view_ids are inferred by occurrence within each sample.
        The optional y argument is ignored.
        """
        if not isinstance(self.shuffle_views, (bool, np.bool_)):
            raise ValueError("shuffle_views must be a boolean.")
        if self.random_state is not None and (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, Integral) or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer or None.")
        reweighting = _make_stage(
            self.feature_reweighting, self.feature_reweighting_kwargs,
            FeatureReweighting, {}, "feature_reweighting",
        )
        pca = _make_stage(
            self.pca, self.pca_kwargs, PCA,
            {"n_components": 768, "random_state": self.random_state}, "pca",
        )
        mcca = _make_stage(
            self.distilled_mcca, self.distilled_mcca_kwargs,
            DistilledMCCA, {"n_components": 128}, "distilled_mcca",
        )
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        if reweighting is not None or mcca is not None:
            if sample_ids is None:
                raise ValueError("sample_ids is required for multiview stages.")
        if view_ids is not None and sample_ids is None:
            raise ValueError("view_ids requires sample_ids.")
        if sample_ids is not None:
            sample_ids, view_ids = _resolve_view_ids(
                sample_ids, view_ids, len(X), self.shuffle_views, self.random_state,
            )
        Z = X
        if reweighting is not None:
            Z = reweighting.fit_transform(Z, sample_ids=sample_ids, view_ids=view_ids)
        if pca is not None:
            # Use transform explicitly so fitting downstream stages follows
            # exactly the same whitening convention as inference.
            Z = pca.fit(Z).transform(Z)
        if mcca is not None:
            mcca.fit(Z, sample_ids=sample_ids, view_ids=view_ids)
        self.coef_, self.intercept_ = _combine_projections(
            reweighting, pca, mcca, self.n_features_in_,
        )
        self.feature_reweighting_ = reweighting
        self.pca_ = pca
        self.distilled_mcca_ = mcca
        self.n_components_ = self.coef_.shape[0]
        return self

    def get_projection(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the combined affine projection as independent copies.

        Returns
        -------
        W : ndarray of shape (n_features, n_components)
            Joint projection matrix for all enabled stages.
        b : ndarray of shape (n_components,)
            Combined offset. Encoded measurements are X @ W + b.
            The offset includes centering and must be applied along with W.
        """
        check_is_fitted(self, ["coef_", "intercept_"])
        return self.coef_.T.copy(), self.intercept_.copy()

    def to_torch(
        self, *, device: "str | torch.device | None" = None,
        dtype: "torch.dtype | None" = None,
    ) -> "torch.nn.Linear":
        """Return an independent PyTorch linear layer with the fitted weights.

        The layer computes the same affine map as transform. It defaults to
        CPU and the fitted coefficient dtype. Pass dtype=torch.float32 for
        use with float32 networks. Parameters are trainable and do not share
        storage with the encoder; use requires_grad_(False) to freeze them.
        """
        import torch

        check_is_fitted(self, ["coef_", "intercept_"])
        weight = torch.as_tensor(self.coef_.copy(), device=device, dtype=dtype)
        bias = torch.as_tensor(self.intercept_.copy(), device=device, dtype=weight.dtype)
        # Replacing an empty layer's parameters avoids random initialization
        # of a potentially large projection matrix.
        layer = torch.nn.Linear(self.n_features_in_, self.n_components_, device="meta", dtype=weight.dtype)
        layer.weight = torch.nn.Parameter(weight)
        layer.bias = torch.nn.Parameter(bias)
        return layer

    def transform(self, X: ArrayLike) -> NDArray[Any]:
        """Encode measurements using the combined affine map; no IDs needed."""
        check_is_fitted(self, ["coef_", "intercept_"])
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        return X @ self.coef_.T + self.intercept_

    def fit_transform(
        self, X: ArrayLike, y: Any = None, *,
        sample_ids: ArrayLike | None = None, view_ids: ArrayLike | None = None,
    ) -> NDArray[Any]:
        """Fit on training measurements and return their embeddings."""
        return self.fit(X, y, sample_ids=sample_ids, view_ids=view_ids).transform(X)


def _make_stage(
    specification: Any, kwargs: Mapping[str, Any] | None,
    estimator_type: type[BaseEstimator], defaults: dict[str, Any], name: str,
) -> Any:
    """Resolve a stage configuration without modifying user parameters."""
    if kwargs is not None and not isinstance(kwargs, Mapping):
        raise TypeError(f"{name}_kwargs must be a mapping or None.")
    if isinstance(specification, str) and specification == "default":
        return estimator_type(**(defaults | dict(kwargs or {})))
    if kwargs is not None:
        raise ValueError(f"{name}_kwargs is only valid with {name}='default'.")
    if specification is None:
        return None
    if not isinstance(specification, estimator_type):
        raise TypeError(f"{name} must be 'default', None, or {estimator_type.__name__}.")
    return clone(specification)


def _resolve_view_ids(
    sample_ids: ArrayLike, view_ids: ArrayLike | None, n_samples: int,
    shuffle: bool, seed: int | None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Resolve one consistent assignment for all multiview stages."""
    samples, sample_index = _encode_ids(sample_ids, "sample_ids", n_samples)
    if view_ids is None:
        view_index = _views_by_occurrence(sample_index)
    else:
        _, view_index = _encode_ids(view_ids, "view_ids", n_samples)
    n_views = int(view_index.max()) + 1
    pairs = sample_index * n_views + view_index
    if np.unique(pairs).size != n_samples:
        raise ValueError("Each (sample_id, view_id) pair must be unique.")
    if shuffle:
        assignments = np.broadcast_to(np.arange(n_views)[:, None], (n_views, len(samples)))
        permutations = np.random.default_rng(seed).permuted(assignments, axis=0)
        view_index = permutations[view_index, sample_index]
    return sample_index, view_index


def _combine_projections(
    reweighting: FeatureReweighting | None, pca: PCA | None,
    mcca: DistilledMCCA | None, n_features: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compose backwards, avoiding a full input-space diagonal weight matrix."""
    if pca is not None:
        projection = pca.components_.T.copy()
        if pca.whiten:
            scale = np.sqrt(pca.explained_variance_)
            scale = np.maximum(scale, np.finfo(scale.dtype).eps)
            projection /= scale
        if mcca is not None:
            projection = projection @ mcca.coef_.T
        offset = -pca.mean_ @ projection
        if mcca is not None:
            offset += mcca.intercept_
    elif mcca is not None:
        projection = mcca.coef_.T.copy()
        offset = mcca.intercept_.copy()
    else:
        projection = np.eye(n_features)
        offset = np.zeros(n_features)
    if reweighting is not None:
        projection *= reweighting.weights_[:, None]
    return projection.T, offset
