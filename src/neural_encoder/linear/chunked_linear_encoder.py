"""Linear encoder for many features, fitted in chunks of features."""

import itertools
from collections.abc import Mapping
from numbers import Integral
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..preprocessing.beta_preprocessor import BetaPreprocessor
from ..preprocessing.session_standard_scaler import SessionStandardScaler
from .cross_view_ridge import CrossViewRidge
from .distilled_mcca import DistilledMCCA
from .feature_reweighting import FeatureReweighting, _validate_cv
from .gram_pca import GramPCA, _as_matrix, _load_chunk
from .linear_encoder import (
    OUTPUT_REWEIGHTING_DEFAULTS, _combine_projections, _cross_validate_output, _make_stage, _resolve_view_ids,
)

__all__ = ["ChunkedLinearEncoder"]


class ChunkedLinearEncoder(TransformerMixin, BaseEstimator):
    """Preprocessing and linear encoding for a very large number of features.

    Fits the same model as SessionStandardScaler (optional), BetaPreprocessor,
    and LinearEncoder with a GramPCA stage, applied in sequence, without
    holding more than one chunk of features in memory. X is typically a
    NumPy memory map of raw measurements (rows are measurements); it is
    never modified or copied in full. Every pass reads one block of
    features for all selected rows, so storing X in Fortran order makes
    reads contiguous. Integer data such as int16 betas are converted to
    floating point on the device.

    Statistics of individual features (session statistics, feature means
    and scales, reliability) are computed from each chunk the first time it
    is read. Statistics over all features need their own pass: clipping
    quantiles, row means for sample centering, and row norms for
    normalization. With the default settings, fitting reads X three times:
    once for the clipping quantiles and twice for the Gram PCA. After PCA,
    the cross-view ridge, distilled MCCA, and output reweighting stages
    operate on the PCA scores in memory, as in LinearEncoder.

    Parameters
    ----------
    session_scaler : {"default", None} or SessionStandardScaler, default=None
        Per-session standardization applied to the raw measurements before
        the beta preprocessing. None skips it. Requires sessions in fit and
        transform.
    preprocessor : {"default", None} or BetaPreprocessor, default="default"
        Beta preprocessing. Its dtype parameter is ignored; computation uses
        the dtype of the PCA stage.
    feature_reweighting : {"default", None} or FeatureReweighting, default="default"
        Feature reliability stage. Callable weightings and normalize=True are
        not supported, because weights are computed one chunk at a time.
    pca : {"default"} or GramPCA, default="default"
        PCA stage, which cannot be disabled. The default uses 768
        components. Its chunk_size, dtype, and device apply to every pass.
    cross_view_ridge : {"default", None} or CrossViewRidge, default="default"
        Denoising map fitted on the PCA scores. None skips it.
    distilled_mcca : {"default", None} or DistilledMCCA, default="default"
        Multiview projection. The default uses 128 components.
    output_reweighting : {"default", None} or FeatureReweighting, default="default"
        Scaling of each output dimension by its reliability, as in
        LinearEncoder. The default weights each dimension by the square root
        of its signal-to-noise ratio, normalized to unit root mean square.
    output_reweighting_cv : int or None, default=None
        Cross-validated reliability for the output weights, as in
        LinearEncoder: the cross-view ridge and distilled MCCA are refitted on
        K-1 folds of the in-memory PCA scores, so X is not read again.
    output_reweighting_matching : {"hungarian", "index", "soft"}, default="hungarian"
        Matching of fold dimensions to fitted dimensions, as in LinearEncoder.
    session_scaler_kwargs, preprocessor_kwargs, feature_reweighting_kwargs, pca_kwargs, cross_view_ridge_kwargs, distilled_mcca_kwargs, output_reweighting_kwargs : dict or None
        Constructor overrides for the corresponding default stage.
    quantile_samples : int, default=10_000_000
        Number of values sampled uniformly at random to estimate clipping
        quantiles. When the training data has at most this many values, all
        of them are used and the bounds match BetaPreprocessor exactly.
    shuffle_views : bool, default=True
        Independently permute view assignments within each sample once
        before fitting, as in LinearEncoder.
    random_state : int or None, default=None
        Nonnegative seed for view shuffling, quantile sampling, and
        cross-validation folds.

    Attributes
    ----------
    session_scaler_ : SessionStandardScaler or None
        Fitted session statistics.
    preprocessor_ : BetaPreprocessor or None
        Fitted clipping bounds and feature statistics.
    feature_reweighting_ : FeatureReweighting or None
        Fitted reliability weights.
    pca_ : GramPCA
        Fitted PCA of the reweighted, preprocessed measurements.
    cross_view_ridge_ : CrossViewRidge or None
        Fitted denoising stage.
    distilled_mcca_ : DistilledMCCA or None
        Fitted multiview projection.
    output_reweighting_ : FeatureReweighting or None
        Fitted output weights.
    coef_ : ndarray of shape (n_components, n_features)
        Combined projection of the preprocessed measurements.
    intercept_ : ndarray of shape (n_components,)
        Combined offset.
    n_features_in_ : int
        Number of features observed during fit.
    n_components_ : int
        Output dimension.

    Notes
    -----
    Preprocessing includes clipping and is therefore not affine; the
    combined projection ``coef_``, ``intercept_`` applies to preprocessed
    measurements. Missing values are filled before session standardization.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 300)).astype(np.float32)
    >>> encoder = ChunkedLinearEncoder(
    ...     preprocessor_kwargs={"scaling": None},
    ...     pca_kwargs={"n_components": 8, "chunk_size": 128},
    ...     distilled_mcca_kwargs={"n_components": 2},
    ...     random_state=0,
    ... )
    >>> encoder.fit_transform(X, sample_ids=np.repeat(np.arange(20), 2)).shape
    (40, 2)
    """

    def __init__(
        self,
        *,
        session_scaler: Literal["default"] | SessionStandardScaler | None = None,
        preprocessor: Literal["default"] | BetaPreprocessor | None = "default",
        feature_reweighting: Literal["default"] | FeatureReweighting | None = "default",
        pca: Literal["default"] | GramPCA = "default",
        cross_view_ridge: Literal["default"] | CrossViewRidge | None = "default",
        distilled_mcca: Literal["default"] | DistilledMCCA | None = "default",
        output_reweighting: Literal["default"] | FeatureReweighting | None = "default",
        session_scaler_kwargs: Mapping[str, Any] | None = None,
        preprocessor_kwargs: Mapping[str, Any] | None = None,
        feature_reweighting_kwargs: Mapping[str, Any] | None = None,
        pca_kwargs: Mapping[str, Any] | None = None,
        cross_view_ridge_kwargs: Mapping[str, Any] | None = None,
        distilled_mcca_kwargs: Mapping[str, Any] | None = None,
        output_reweighting_kwargs: Mapping[str, Any] | None = None,
        output_reweighting_cv: int | None = None,
        output_reweighting_matching: Literal["hungarian", "index", "soft"] = "hungarian",
        quantile_samples: int = 10_000_000,
        shuffle_views: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.session_scaler = session_scaler
        self.preprocessor = preprocessor
        self.feature_reweighting = feature_reweighting
        self.pca = pca
        self.cross_view_ridge = cross_view_ridge
        self.distilled_mcca = distilled_mcca
        self.output_reweighting = output_reweighting
        self.session_scaler_kwargs = session_scaler_kwargs
        self.preprocessor_kwargs = preprocessor_kwargs
        self.feature_reweighting_kwargs = feature_reweighting_kwargs
        self.pca_kwargs = pca_kwargs
        self.cross_view_ridge_kwargs = cross_view_ridge_kwargs
        self.distilled_mcca_kwargs = distilled_mcca_kwargs
        self.output_reweighting_kwargs = output_reweighting_kwargs
        self.output_reweighting_cv = output_reweighting_cv
        self.output_reweighting_matching = output_reweighting_matching
        self.quantile_samples = quantile_samples
        self.shuffle_views = shuffle_views
        self.random_state = random_state

    def fit(
        self, X: ArrayLike, y: Any = None, *, sample_ids: ArrayLike,
        view_ids: ArrayLike | None = None, sessions: ArrayLike | None = None,
        rows: ArrayLike | None = None,
    ) -> Self:
        """Fit on the selected rows of X.

        rows selects training measurements without copying X; None uses all
        rows. sample_ids, view_ids, and sessions have one entry per selected
        row. The optional y argument is ignored.
        """
        self._fit(X, sample_ids, view_ids, sessions, rows)
        return self

    def fit_transform(
        self, X: ArrayLike, y: Any = None, *, sample_ids: ArrayLike,
        view_ids: ArrayLike | None = None, sessions: ArrayLike | None = None,
        rows: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Fit and return the training embeddings, without another pass over X."""
        return self._fit(X, sample_ids, view_ids, sessions, rows)

    def transform(
        self, X: ArrayLike, *, sessions: ArrayLike | None = None, rows: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Encode the selected rows of X; sessions is required with a session scaler."""
        check_is_fitted(self, ["coef_", "intercept_"])
        chunks = self._chunks(X, rows, sessions)
        return (chunks.project(self.coef_, "preprocessed") + self.intercept_)

    def transform_until(
        self, X: ArrayLike, *,
        stage: Literal["pca", "cross_view_ridge", "distilled_mcca", "output_reweighting"],
        sessions: ArrayLike | None = None, rows: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Transform the selected rows of X through a fitted stage, inclusive.

        stage="distilled_mcca" returns the embedding before output reweighting.
        """
        check_is_fitted(self, ["coef_", "intercept_"])
        stages = ("pca", "cross_view_ridge", "distilled_mcca", "output_reweighting")
        if stage not in stages:
            raise ValueError(f"stage must be one of {stages}.")
        if getattr(self, f"{stage}_") is None:
            raise ValueError(f"The requested stage '{stage}' is disabled.")
        if stage == "output_reweighting" or (stage == "distilled_mcca" and self.output_reweighting_ is None):
            return self.transform(X, sessions=sessions, rows=rows)
        if stage == "distilled_mcca":
            weights = None if self.feature_reweighting_ is None else self.feature_reweighting_.weights_
            coef, intercept = _combine_projections(
                weights, self.pca_, self.cross_view_ridge_, self.distilled_mcca_, self.n_features_in_,
            )
            return self._chunks(X, rows, sessions).project(coef, "preprocessed") + intercept
        pca = self.pca_
        Z = self._chunks(X, rows, sessions).project(pca.components_, "weighted")
        Z -= pca.mean_ @ pca.components_.T.astype(np.float64)
        if pca.whiten:
            Z /= np.sqrt(pca.explained_variance_)
        if stage == "cross_view_ridge":
            Z = self.cross_view_ridge_.transform(Z)
        return Z

    def get_projection(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return W, b such that embeddings are preprocessed(X) @ W + b."""
        check_is_fitted(self, ["coef_", "intercept_"])
        return self.coef_.T.copy(), self.intercept_.copy()

    def _fit(
        self, X: ArrayLike, sample_ids: ArrayLike, view_ids: ArrayLike | None,
        sessions: ArrayLike | None, rows: ArrayLike | None,
    ) -> NDArray[np.float64]:
        import torch

        self._validate_parameters()
        scaler = _make_stage(
            self.session_scaler, self.session_scaler_kwargs, SessionStandardScaler, {}, "session_scaler",
        )
        preprocessor = _make_stage(
            self.preprocessor, self.preprocessor_kwargs, BetaPreprocessor, {}, "preprocessor",
        )
        reweighting = _make_stage(
            self.feature_reweighting, self.feature_reweighting_kwargs,
            FeatureReweighting, {}, "feature_reweighting",
        )
        pca = _make_stage(self.pca, self.pca_kwargs, GramPCA, {"n_components": 768}, "pca")
        ridge = _make_stage(
            self.cross_view_ridge, self.cross_view_ridge_kwargs, CrossViewRidge, {}, "cross_view_ridge",
        )
        mcca = _make_stage(
            self.distilled_mcca, self.distilled_mcca_kwargs,
            DistilledMCCA, {"n_components": 128}, "distilled_mcca",
        )
        output = _make_stage(
            self.output_reweighting, self.output_reweighting_kwargs,
            FeatureReweighting, OUTPUT_REWEIGHTING_DEFAULTS, "output_reweighting",
        )
        if output is not None:
            output._validate_parameters()
        _validate_cv(self.output_reweighting_cv, self.output_reweighting_matching)
        if self.output_reweighting_cv is not None and output is None:
            raise ValueError("output_reweighting_cv requires output reweighting.")
        if pca is None:
            raise ValueError("pca cannot be None.")
        pca._validate_parameters()
        if scaler is not None:
            scaler._validate_parameters()
        probabilities = None if preprocessor is None else preprocessor._validate_parameters()
        if reweighting is not None:
            reweighting._validate_parameters()
            if callable(reweighting.weighting):
                raise ValueError("Callable feature weightings are not supported.")
            if reweighting.normalize:
                raise ValueError("Feature reweighting with normalize=True is not supported.")

        X = _as_matrix(X)
        rows = _check_rows(rows, len(X))
        n_samples = len(X) if rows is None else len(rows)
        n_features = X.shape[1]
        if n_samples < 2:
            raise ValueError("At least two measurements are required.")
        sample_index, view_index = _resolve_view_ids(
            sample_ids, view_ids, n_samples, self.shuffle_views, self.random_state,
        )
        self.n_features_in_ = n_features
        self.session_scaler_ = scaler
        self.preprocessor_ = preprocessor
        self.feature_reweighting_ = reweighting
        self.pca_ = pca
        chunks = _Chunks(self, X, rows, fitting=True)
        if scaler is not None:
            chunks.fit_sessions(_check_sessions(sessions, n_samples, "fit"))
        elif sessions is not None:
            raise ValueError("sessions requires a session_scaler.")
        if reweighting is not None:
            chunks.views = torch.as_tensor(_complete_views(sample_index, view_index), device=chunks.device)
            reweighting.reliability_ = np.zeros(n_features)
            reweighting.weights_ = np.zeros(n_features)
            reweighting.n_features_in_ = n_features
            reweighting.n_views_, reweighting.n_samples_ = chunks.views.shape
        if preprocessor is not None:
            preprocessor.n_features_in_ = n_features
            preprocessor.feature_mean_ = np.zeros(n_features) if preprocessor.feature_centering else None
            preprocessor.feature_scale_ = np.ones(n_features) if preprocessor.feature_scaling else None
            if probabilities is not None:
                preprocessor.clip_bounds_ = chunks.quantiles(
                    probabilities, self.quantile_samples, self.random_state,
                )
            elif preprocessor.clip_bounds is not None:
                preprocessor.clip_bounds_ = np.asarray(preprocessor.clip_bounds, dtype=np.float64)
            else:
                preprocessor.clip_bounds_ = None
        chunks.fit_rows()

        Z = pca._fit_chunks(lambda chunk_slice: chunks.apply(chunk_slice, "weighted"), n_samples, n_features)
        Z = scores = Z.astype(np.float64)
        if ridge is not None:
            Z = ridge.fit_transform(Z, sample_ids=sample_index, view_ids=view_index)
        if mcca is not None:
            Z = mcca.fit(Z, sample_ids=sample_index, view_ids=view_index).transform(Z)
        if output is not None:
            output.fit(Z, sample_ids=sample_index, view_ids=view_index)
            if self.output_reweighting_cv is not None:
                _cross_validate_output(
                    output, scores, Z, ridge, mcca, sample_index, view_index,
                    self.output_reweighting_cv, self.output_reweighting_matching, self.random_state,
                )
            Z = output.transform(Z)
        coef, self.intercept_ = _combine_projections(
            None if reweighting is None else reweighting.weights_, pca, ridge, mcca, n_features,
            None if output is None else output.weights_,
        )
        self.coef_ = np.ascontiguousarray(coef)
        self.cross_view_ridge_ = ridge
        self.distilled_mcca_ = mcca
        self.output_reweighting_ = output
        self.n_components_ = self.coef_.shape[0]
        return Z

    def _validate_parameters(self) -> None:
        if not isinstance(self.shuffle_views, (bool, np.bool_)):
            raise ValueError("shuffle_views must be a boolean.")
        if self.random_state is not None and (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, Integral) or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer or None.")
        if (
            isinstance(self.quantile_samples, (bool, np.bool_))
            or not isinstance(self.quantile_samples, Integral) or self.quantile_samples < 1
        ):
            raise ValueError("quantile_samples must be a positive integer.")

    def _chunks(self, X: ArrayLike, rows: ArrayLike | None, sessions: ArrayLike | None) -> "_Chunks":
        """Prepare fitted preprocessing, including row statistics, for new rows."""
        X = _as_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but ChunkedLinearEncoder is expecting "
                f"{self.n_features_in_} features."
            )
        rows = _check_rows(rows, len(X))
        chunks = _Chunks(self, X, rows, fitting=False)
        n_samples = len(X) if rows is None else len(rows)
        if self.session_scaler_ is not None:
            chunks.set_sessions(_check_sessions(sessions, n_samples, "transform"))
        elif sessions is not None:
            raise ValueError("sessions requires a session_scaler.")
        chunks.fit_rows()
        return chunks


class _Chunks:
    """Fitted, or fitting, preprocessing applied to one chunk of features at a time.

    During fitting, statistics of individual features are computed the first
    time their chunk is processed and stored on the fitted stage objects.
    """

    _steps = ("scaled", "clipped", "feature_scaled", "preprocessed", "weighted")

    def __init__(self, encoder: ChunkedLinearEncoder, X: NDArray[Any], rows: NDArray[np.intp] | None,
                 fitting: bool) -> None:
        self.X, self.rows, self.fitting = X, rows, fitting
        self.n_samples = len(X) if rows is None else len(rows)
        self.n_features = X.shape[1]
        self.scaler = encoder.session_scaler_
        self.preprocessor = encoder.preprocessor_
        self.reweighting = encoder.feature_reweighting_
        self.chunk_size = encoder.pca_.chunk_size
        self.device, self.dtype = encoder.pca_._torch_options()
        self.views = None
        self.codes = None
        self.row_mean = None
        self.row_norm = None
        self.done: set[tuple[str, int]] = set()

    def slices(self) -> list[slice]:
        return [
            slice(start, min(start + self.chunk_size, self.n_features))
            for start in range(0, self.n_features, self.chunk_size)
        ]

    def apply(self, chunk_slice: slice, until: str) -> Any:
        """Return the chunk processed up to and including the given step."""
        import torch

        stop = self._steps.index(until)
        preprocessor = self.preprocessor
        fill_value = None if preprocessor is None else preprocessor.fill_value
        x = _load_chunk(self.X, chunk_slice, self.device, self.dtype, rows=self.rows, fill_value=fill_value)
        if self.scaler is not None:
            x = self._standardize_sessions(x, chunk_slice)
        if preprocessor is not None:
            if preprocessor.scaling is not None:
                x /= preprocessor.scaling
            if stop == 0:
                return x
            if preprocessor.clip_bounds_ is not None:
                x.clamp_(*(float(bound) for bound in preprocessor.clip_bounds_))
            if stop == 1:
                return x
            if preprocessor.sample_centering:
                x -= self.row_mean[:, None].to(self.dtype)
            x = self._scale_features(x, chunk_slice)
            if stop == 2:
                return x
            if preprocessor.normalize:
                x /= self.row_norm[:, None].to(self.dtype)
        if stop == 3 or self.reweighting is None:
            return x
        if self.fitting and ("weights", chunk_slice.start) not in self.done:
            reliability = _reliability(x, self.views, self.reweighting.method).cpu().numpy()
            weights = self.reweighting._weights(reliability)
            self.reweighting.reliability_[chunk_slice] = reliability
            self.reweighting.weights_[chunk_slice] = weights
            self.done.add(("weights", chunk_slice.start))
        x *= torch.as_tensor(self.reweighting.weights_[chunk_slice], dtype=self.dtype, device=self.device)
        return x

    def quantiles(self, probabilities: NDArray[np.float64], n_values: int, seed: int | None) -> NDArray[np.float64]:
        """Clipping quantiles of all scaled values, or of a uniform random sample of them."""
        import torch

        fraction = n_values / (self.n_samples * self.n_features)
        generator = torch.Generator(device=self.device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        values = []
        for chunk_slice in self.slices():
            x = self.apply(chunk_slice, "scaled")
            if fraction < 1:
                x = x[torch.rand(x.shape, generator=generator, device=self.device) < fraction]
            values.append(x.ravel().cpu().numpy())
        return np.quantile(np.concatenate(values).astype(np.float64), probabilities)

    def fit_rows(self) -> None:
        """Compute the row statistics needed by sample centering and normalization."""
        if self.preprocessor is None:
            return
        if self.preprocessor.sample_centering:
            self.row_mean = self._row_sums("clipped") / self.n_features
        if self.preprocessor.normalize:
            norm = self._row_sums("feature_scaled", square=True).sqrt()
            self.row_norm = norm.where(norm > 0, 1.0)

    def project(self, matrix: NDArray[Any], until: str) -> NDArray[np.float64]:
        """Return the processed rows times matrix.T, accumulated over chunks."""
        import torch

        out = torch.zeros((self.n_samples, matrix.shape[0]), dtype=torch.float64, device=self.device)
        for chunk_slice in self.slices():
            x = self.apply(chunk_slice, until)
            out += (x @ torch.as_tensor(matrix[:, chunk_slice], dtype=self.dtype, device=self.device).T).to(
                torch.float64
            )
        return out.cpu().numpy()

    def fit_sessions(self, sessions: NDArray[Any]) -> None:
        """Assign session codes; sessions with too few rows use pooled statistics."""
        scaler = self.scaler
        values, inverse, counts = np.unique(sessions, return_inverse=True, return_counts=True)
        kept = counts >= scaler.min_samples
        scaler.sessions_ = values[kept]
        codes = np.full(len(values), kept.sum())
        codes[kept] = np.arange(kept.sum())
        self._set_codes(codes[inverse])
        shape = (int(kept.sum()), self.n_features)
        scaler.means_ = np.zeros(shape, dtype=scaler.dtype)
        scaler.scales_ = np.ones(shape, dtype=scaler.dtype)
        scaler.mean_ = np.zeros(self.n_features, dtype=scaler.dtype)
        scaler.scale_ = np.ones(self.n_features, dtype=scaler.dtype)
        scaler.n_features_in_ = self.n_features

    def set_sessions(self, sessions: NDArray[Any]) -> None:
        """Assign codes of fitted sessions; unseen sessions use pooled statistics."""
        fitted = self.scaler.sessions_
        position = np.searchsorted(fitted, sessions).clip(max=max(len(fitted) - 1, 0))
        known = (fitted[position] == sessions) if len(fitted) else np.zeros(len(sessions), dtype=bool)
        self._set_codes(np.where(known, position, len(fitted)))

    def _set_codes(self, codes: NDArray[np.intp]) -> None:
        import torch

        self.codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)

    def _standardize_sessions(self, x: Any, chunk_slice: slice) -> Any:
        import torch

        scaler = self.scaler
        n_kept = len(scaler.sessions_)
        if self.fitting and ("sessions", chunk_slice.start) not in self.done:
            values = x.to(torch.float64)
            kept = self.codes < n_kept
            counts = torch.bincount(self.codes[kept], minlength=n_kept).to(torch.float64)[:, None]
            means = torch.zeros((n_kept, values.shape[1]), dtype=torch.float64, device=self.device)
            means.index_add_(0, self.codes[kept], values[kept])
            means /= counts
            squares = torch.zeros_like(means).index_add_(
                0, self.codes[kept], (values[kept] - means[self.codes[kept]]) ** 2,
            )
            scales = (squares / counts).sqrt()
            mean = values.mean(dim=0)
            scale = values.std(dim=0, correction=0)
            scaler.means_[:, chunk_slice] = means.cpu().numpy()
            scaler.scales_[:, chunk_slice] = torch.where(scales > 0, scales, 1.0).cpu().numpy()
            scaler.mean_[chunk_slice] = mean.cpu().numpy()
            scaler.scale_[chunk_slice] = torch.where(scale > 0, scale, 1.0).cpu().numpy()
            self.done.add(("sessions", chunk_slice.start))

        def table(per_session: NDArray[Any], pooled: NDArray[Any]) -> Any:
            stacked = np.concatenate([per_session[:, chunk_slice], pooled[None, chunk_slice]])
            return torch.as_tensor(stacked, dtype=self.dtype, device=self.device)[self.codes]

        if scaler.with_mean:
            x -= table(scaler.means_, scaler.mean_)
        if scaler.with_std:
            x /= table(scaler.scales_, scaler.scale_)
        return x

    def _scale_features(self, x: Any, chunk_slice: slice) -> Any:
        import torch

        preprocessor = self.preprocessor
        if self.fitting and ("features", chunk_slice.start) not in self.done:
            values = x.to(torch.float64)
            if preprocessor.feature_centering:
                preprocessor.feature_mean_[chunk_slice] = values.mean(dim=0).cpu().numpy()
            if preprocessor.feature_scaling:
                scale = values.std(dim=0, correction=0)
                preprocessor.feature_scale_[chunk_slice] = torch.where(scale > 0, scale, 1.0).cpu().numpy()
            self.done.add(("features", chunk_slice.start))
        if preprocessor.feature_mean_ is not None:
            x -= torch.as_tensor(preprocessor.feature_mean_[chunk_slice], dtype=self.dtype, device=self.device)
        if preprocessor.feature_scale_ is not None:
            x /= torch.as_tensor(preprocessor.feature_scale_[chunk_slice], dtype=self.dtype, device=self.device)
        return x

    def _row_sums(self, until: str, square: bool = False) -> Any:
        import torch

        total = torch.zeros(self.n_samples, dtype=torch.float64, device=self.device)
        for chunk_slice in self.slices():
            x = self.apply(chunk_slice, until)
            total += (x * x if square else x).sum(dim=1, dtype=torch.float64)
        return total


def _reliability(x: Any, views: Any, method: str = "correlation") -> Any:
    """Leave-one-view-out or pairwise reliability of each column, as in FeatureReweighting."""
    import torch

    groups = x[views].to(torch.float64)
    if method == "pairwise":
        pairs = list(itertools.combinations(range(len(groups)), 2))
        return sum(_column_correlations(groups[i], groups[j]) for i, j in pairs) / len(pairs)
    total = groups.sum(dim=0)
    reliability = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
    for view in groups:
        reliability += _column_correlations(view, (total - view) / (len(groups) - 1))
    return reliability / len(groups)


def _column_correlations(X: Any, Y: Any) -> Any:
    """Pearson correlations of matching columns, zero for constant columns."""
    import torch

    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)
    denominator = ((X * X).sum(dim=0) * (Y * Y).sum(dim=0)).sqrt()
    numerator = (X * Y).sum(dim=0)
    correlations = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))
    return correlations.clamp(-1, 1)


def _complete_views(sample_index: NDArray[np.intp], view_index: NDArray[np.intp]) -> NDArray[np.intp]:
    """Row positions of samples observed in every view, shaped (n_views, n_complete)."""
    n_views = int(view_index.max()) + 1
    table = np.full((int(sample_index.max()) + 1, n_views), -1)
    table[sample_index, view_index] = np.arange(len(sample_index))
    table = table[(table >= 0).all(axis=1)]
    if n_views < 2:
        raise ValueError("At least two views are required.")
    if len(table) < 2:
        raise ValueError("At least two complete samples are required.")
    return table.T


def _check_rows(rows: ArrayLike | None, n_rows: int) -> NDArray[np.intp] | None:
    if rows is None:
        return None
    rows = np.asarray(rows)
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise ValueError("rows must be a one-dimensional array of integer indices.")
    if len(rows) and (rows.min() < -n_rows or rows.max() >= n_rows):
        raise ValueError("rows contains indices outside X.")
    return rows


def _check_sessions(sessions: ArrayLike | None, n_samples: int, method: str) -> NDArray[Any]:
    if sessions is None:
        raise ValueError(f"sessions is required to {method} with a session_scaler.")
    sessions = np.asarray(sessions).ravel()
    if len(sessions) != n_samples:
        raise ValueError("sessions must have one entry per selected row.")
    return sessions
