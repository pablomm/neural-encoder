"""Compose feature reweighting, PCA, cross-view ridge, distilled MCCA, and output reweighting."""

from collections.abc import Mapping
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted, validate_data

from ..utils.split_views import _encode_ids, _views_by_occurrence
from .cross_view_ridge import CrossViewRidge
from .distilled_mcca import DistilledMCCA
from .feature_reweighting import FeatureReweighting, _cross_validated_reliability, _validate_cv
from .gram_pca import GramPCA

__all__ = ["LinearEncoder"]

# The default output stage weights each embedding dimension by the square
# root of its single-measurement signal-to-noise ratio.
OUTPUT_REWEIGHTING_DEFAULTS = {"method": "pairwise", "weighting": "sqrt_snr", "normalize": True}

if TYPE_CHECKING:
    import torch


class LinearEncoder(TransformerMixin, BaseEstimator):
    """Learn an affine encoder from repeated or multiview measurements.

    Fits feature reweighting, PCA, cross-view ridge denoising, distilled
    MCCA, and output reweighting in that order. Each stage can use its
    default estimator, a supplied estimator to clone, or None to skip the
    stage. All stages are affine, so the fitted encoder is a single
    projection. Preprocessing is performed outside this encoder.

    Parameters
    ----------
    feature_reweighting : {"default", None} or FeatureReweighting, default="default"
        Feature reliability stage. The default is FeatureReweighting().
    pca : {"default", None}, sklearn.decomposition.PCA, or GramPCA, default="default"
        Dimensionality reduction. The default uses 768 components. Supports
        PCA whitening. GramPCA computes the same PCA in feature chunks, for
        data with many more features than samples; other
        dimensionality-reduction classes are not accepted.
    cross_view_ridge : {"default", None} or CrossViewRidge, default="default"
        Denoising map from each measurement to the mean of its other views,
        fitted on the PCA scores (or on the preceding stage's output) and
        applied before MCCA. None skips it, fitting MCCA directly on the PCA
        scores as in the original architecture.
    distilled_mcca : {"default", None} or DistilledMCCA, default="default"
        Multiview projection. The default uses 128 components.
    output_reweighting : {"default", None} or FeatureReweighting, default="default"
        Scaling of each output dimension by its reliability across views,
        fitted on the training embeddings. The default is
        FeatureReweighting(method="pairwise", weighting="sqrt_snr",
        normalize=True): each dimension is multiplied by the square root of
        its signal-to-noise ratio, so unreliable MCCA dimensions contribute
        little to cosine similarities, rescaled to preserve the overall
        scale. None returns the unweighted embedding.
    output_reweighting_cv : int or None, default=None
        Estimate the reliability used by the output weights by K-fold
        cross-validation over samples instead of on the training embeddings,
        whose reliability is optimistic for the least reliable dimensions
        because the encoder was fitted to them. The stages up to PCA stay
        fitted on all samples; the cross-view ridge and distilled MCCA are
        refitted on K-1 folds, and the reliability of their outputs on the
        held-out fold is transferred to the fitted dimensions (see
        output_reweighting_matching) and averaged over folds. This costs K
        extra ridge and MCCA fits. None uses in-sample reliability.
    output_reweighting_matching : {"hungarian", "index", "soft"}, default="hungarian"
        How fold dimensions are matched to the fitted dimensions when
        output_reweighting_cv is set: "hungarian" pairs them one to one by
        maximal absolute correlation on the held-out samples, "index" by
        position, and "soft" averages fold reliabilities weighted by squared
        correlations.
    feature_reweighting_kwargs, pca_kwargs, cross_view_ridge_kwargs, distilled_mcca_kwargs, output_reweighting_kwargs : dict or None
        Constructor overrides for the corresponding default stage. Supplying
        kwargs with an explicit estimator or disabled stage raises an error.
        Component counts must be valid for the data; they are not reduced
        automatically.
    shuffle_views : bool, default=True
        Independently permute view assignments within each sample once before
        fitting. All multiview stages use the same assignments. Measurements
        remain in their original row order, and no measurements are imputed
        by this operation.
    random_state : int or None, default=None
        Nonnegative seed for view shuffling, cross-validation folds, and the
        default PCA, unless its random_state is overridden in pca_kwargs. Explicit estimator instances
        retain their own random-state configuration.

    Attributes
    ----------
    feature_reweighting_ : FeatureReweighting or None
        Fitted feature reweighting stage.
    pca_ : sklearn.decomposition.PCA, GramPCA, or None
        Fitted PCA stage.
    cross_view_ridge_ : CrossViewRidge or None
        Fitted denoising stage.
    distilled_mcca_ : DistilledMCCA or None
        Fitted distilled MCCA stage.
    output_reweighting_ : FeatureReweighting or None
        Fitted output reweighting stage. Its ``reliability_`` is cross-validated
        when output_reweighting_cv is set.
    coef_ : ndarray of shape (n_components, n_features)
        Combined coefficients. Transform computes ``X @ coef_.T + intercept_``.
    intercept_ : ndarray of shape (n_components,)
        Combined offset, accounting for PCA centering, denoising, MCCA
        distillation, and output weights.
    n_features_in_ : int
        Number of input features.
    n_components_ : int
        Output dimensionality.

    Notes
    -----
    Reliability uses complete samples; PCA uses all training measurements;
    the cross-view ridge uses samples with at least two views; MCCA handles
    missing views according to its impute parameter. Output reliability is
    estimated in-sample on the training embeddings unless
    output_reweighting_cv is set; the in-sample estimate is optimistic for
    the least reliable dimensions. PCA is
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
        pca: Literal["default"] | PCA | GramPCA | None = "default",
        cross_view_ridge: Literal["default"] | CrossViewRidge | None = "default",
        distilled_mcca: Literal["default"] | DistilledMCCA | None = "default",
        output_reweighting: Literal["default"] | FeatureReweighting | None = "default",
        feature_reweighting_kwargs: Mapping[str, Any] | None = None,
        pca_kwargs: Mapping[str, Any] | None = None,
        cross_view_ridge_kwargs: Mapping[str, Any] | None = None,
        distilled_mcca_kwargs: Mapping[str, Any] | None = None,
        output_reweighting_kwargs: Mapping[str, Any] | None = None,
        output_reweighting_cv: int | None = None,
        output_reweighting_matching: Literal["hungarian", "index", "soft"] = "hungarian",
        shuffle_views: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.feature_reweighting = feature_reweighting
        self.pca = pca
        self.cross_view_ridge = cross_view_ridge
        self.distilled_mcca = distilled_mcca
        self.output_reweighting = output_reweighting
        self.feature_reweighting_kwargs = feature_reweighting_kwargs
        self.pca_kwargs = pca_kwargs
        self.cross_view_ridge_kwargs = cross_view_ridge_kwargs
        self.distilled_mcca_kwargs = distilled_mcca_kwargs
        self.output_reweighting_kwargs = output_reweighting_kwargs
        self.output_reweighting_cv = output_reweighting_cv
        self.output_reweighting_matching = output_reweighting_matching
        self.shuffle_views = shuffle_views
        self.random_state = random_state

    def fit(
        self, X: ArrayLike, y: Any = None, *,
        sample_ids: ArrayLike | None = None, view_ids: ArrayLike | None = None,
    ) -> Self:
        """Fit all enabled stages on training measurements.

        sample_ids is required when any multiview stage is enabled.
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
            accepted=(PCA, GramPCA),
        )
        ridge = _make_stage(
            self.cross_view_ridge, self.cross_view_ridge_kwargs,
            CrossViewRidge, {}, "cross_view_ridge",
        )
        mcca = _make_stage(
            self.distilled_mcca, self.distilled_mcca_kwargs,
            DistilledMCCA, {"n_components": 128}, "distilled_mcca",
        )
        output = _make_stage(
            self.output_reweighting, self.output_reweighting_kwargs,
            FeatureReweighting, OUTPUT_REWEIGHTING_DEFAULTS, "output_reweighting",
        )
        _validate_cv(self.output_reweighting_cv, self.output_reweighting_matching)
        if self.output_reweighting_cv is not None and output is None:
            raise ValueError("output_reweighting_cv requires output reweighting.")
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        if any(stage is not None for stage in (reweighting, ridge, mcca, output)):
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
        scores = Z
        if ridge is not None:
            Z = ridge.fit_transform(Z, sample_ids=sample_ids, view_ids=view_ids)
        if mcca is not None:
            Z = mcca.fit(Z, sample_ids=sample_ids, view_ids=view_ids).transform(Z)
        if output is not None:
            output.fit(Z, sample_ids=sample_ids, view_ids=view_ids)
            if self.output_reweighting_cv is not None:
                _cross_validate_output(
                    output, scores, Z, ridge, mcca, sample_ids, view_ids,
                    self.output_reweighting_cv, self.output_reweighting_matching, self.random_state,
                )
        self.coef_, self.intercept_ = _combine_projections(
            None if reweighting is None else reweighting.weights_, pca, ridge, mcca, self.n_features_in_,
            None if output is None else output.weights_,
        )
        self.feature_reweighting_ = reweighting
        self.pca_ = pca
        self.cross_view_ridge_ = ridge
        self.distilled_mcca_ = mcca
        self.output_reweighting_ = output
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

    def transform_until(
        self, X: ArrayLike, *,
        stage: Literal["feature_reweighting", "pca", "cross_view_ridge", "distilled_mcca", "output_reweighting"],
    ) -> NDArray[Any]:
        """Transform measurements through the selected fitted stage, inclusive.

        For example, stage="pca" returns PCA scores after feature reweighting,
        stage="cross_view_ridge" returns the denoised PCA scores, and
        stage="distilled_mcca" returns the embedding before output
        reweighting. Disabled preceding stages are skipped; a disabled target
        raises an error. No sample or view IDs are required.
        """
        check_is_fitted(self, ["coef_", "intercept_"])
        stages = ("feature_reweighting", "pca", "cross_view_ridge", "distilled_mcca", "output_reweighting")
        if stage not in stages:
            raise ValueError(f"stage must be one of {stages}.")
        if getattr(self, f"{stage}_") is None:
            raise ValueError(f"The requested stage '{stage}' is disabled.")
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        for name in stages[:stages.index(stage) + 1]:
            estimator = getattr(self, f"{name}_")
            if estimator is not None:
                X = estimator.transform(X)
        return X

    def fit_transform(
        self, X: ArrayLike, y: Any = None, *,
        sample_ids: ArrayLike | None = None, view_ids: ArrayLike | None = None,
    ) -> NDArray[Any]:
        """Fit on training measurements and return their embeddings."""
        return self.fit(X, y, sample_ids=sample_ids, view_ids=view_ids).transform(X)


def _make_stage(
    specification: Any, kwargs: Mapping[str, Any] | None,
    estimator_type: type[BaseEstimator], defaults: dict[str, Any], name: str,
    accepted: tuple[type[BaseEstimator], ...] | None = None,
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
    accepted = accepted or (estimator_type,)
    if not isinstance(specification, accepted):
        names = " or ".join(cls.__name__ for cls in accepted)
        raise TypeError(f"{name} must be 'default', None, or {names}.")
    return clone(specification)


def _cross_validate_output(
    output: FeatureReweighting, scores: NDArray[Any], embedding: NDArray[Any],
    ridge: CrossViewRidge | None, mcca: DistilledMCCA | None,
    sample_ids: NDArray[np.intp], view_ids: NDArray[np.intp],
    cv: int, matching: str, random_state: int | None,
) -> None:
    """Replace the output reliability by a cross-validated estimate.

    scores are the training inputs of the cross-view ridge (PCA scores), and
    embedding the fitted outputs before reweighting. For each fold, the ridge
    and MCCA are refitted on the other folds.
    """
    def fit_fold(train: NDArray[np.bool_]) -> Any:
        ids = {"sample_ids": sample_ids[train], "view_ids": view_ids[train]}
        denoised = scores if ridge is None else clone(ridge).fit(scores[train], **ids).transform(scores)
        fold_mcca = None if mcca is None else clone(mcca).fit(denoised[train], **ids)
        return lambda rows: denoised[rows] if fold_mcca is None else fold_mcca.transform(denoised[rows])

    output._set_reliability(_cross_validated_reliability(
        fit_fold, embedding, sample_ids, view_ids,
        method=output.method, cv=cv, matching=matching, random_state=random_state,
    ))


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
    weights: NDArray[np.float64] | None, pca: PCA | GramPCA | None,
    ridge: CrossViewRidge | None, mcca: DistilledMCCA | None, n_features: int,
    output_weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compose the enabled stages into one affine map (coef, intercept).

    Stages after reweighting are x @ A + b maps, composed in order; the PCA
    offset is -mean @ projection. Input reweighting is applied last as a row
    scaling, avoiding a full input-space diagonal weight matrix; output
    weights scale the columns and the offset.
    """
    stages = []
    if pca is not None:
        components = pca.components_.T
        if pca.whiten:
            scale = np.sqrt(pca.explained_variance_)
            scale = np.maximum(scale, np.finfo(scale.dtype).eps)
            components = components / scale
        stages.append((components, None))
    if ridge is not None:
        stages.append((ridge.coef_.T, ridge.intercept_))
    if mcca is not None:
        stages.append((mcca.coef_.T, mcca.intercept_))
    if not stages:
        projection, offset = np.eye(n_features), np.zeros(n_features)
    else:
        # Multiply the small later stages first so that no intermediate has
        # the input dimension times the PCA dimension.
        tail = None
        for A, _ in stages[1:]:
            tail = A if tail is None else tail @ A
        projection = stages[0][0].copy() if tail is None else stages[0][0] @ tail
        offset = -pca.mean_ @ projection if pca is not None else np.zeros(projection.shape[1])
        # Each stage's intercept passes through the stages that follow it.
        for i, (_, b) in enumerate(stages):
            if b is None:
                continue
            for A, _ in stages[i + 1:]:
                b = b @ A
            offset = offset + b
    if weights is not None:
        projection *= weights[:, None]
    if output_weights is not None:
        projection = projection * output_weights
        offset = offset * output_weights
    return projection.T, offset
