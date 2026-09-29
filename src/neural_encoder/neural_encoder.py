"""Combined linear encoding and PCA-driven nonlinear refinement."""

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import nn

from .linear import LinearEncoder
from .linear.linear_encoder import _combine_projections
from .nonlinear import NonlinearRefiner, ResidualNetwork

__all__ = ["NeuralEncoder", "NeuralEncoderModule"]


class NeuralEncoderModule(nn.Module):
    """PyTorch encoder with a shared or separate PCA residual input.

    PCA layers include feature reweighting and centering. When refinement_pca
    is None, the linear PCA output is computed once and used by both branches.
    """

    def __init__(
        self, pca: nn.Linear, mcca: nn.Linear, refiner: ResidualNetwork,
        refinement_pca: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        self.pca = pca
        self.mcca = mcca
        self.refiner = refiner
        self.refinement_pca = refinement_pca

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        scores = self.pca(X)
        source = scores if self.refinement_pca is None else self.refinement_pca(X)
        return self.refiner(self.mcca(scores), network_input=source)


class NeuralEncoder(TransformerMixin, BaseEstimator):
    """Learn a linear embedding and a nonlinear residual from PCA scores.

    Preprocessing is external. Fit uses repeated measurements identified by
    sample_ids and optional view_ids; transform accepts individual measurements.

    Parameters
    ----------
    n_components_pca : int, default=768
        PCA dimension feeding the linear encoder's distilled MCCA.
    n_components_mcca : int, default=128
        Final embedding dimension.
    n_components_pca_refinement : int or None, default=None
        PCA dimension feeding the residual network. None uses the linear PCA
        dimension. Equal dimensions reuse the same fitted PCA and its scores.
    feature_reweighting_kwargs, pca_kwargs, distilled_mcca_kwargs : dict or None
        Settings for the linear stages. Component counts are controlled by the
        dimension parameters above. pca_kwargs applies to both PCAs.
    refiner_kwargs : dict or None
        NonlinearRefiner settings, including network, loss, optimizer, training,
        device, and logger options. Supplied network and loss modules are used
        directly.
    shuffle_views : bool, default=False
        Shuffle view assignments when fitting the linear encoder.
    random_state : int or None, default=None
        Default seed for PCA, view shuffling, and nonlinear training. Individual
        estimator kwargs can override the corresponding random_state.

    Attributes
    ----------
    linear_encoder_ : LinearEncoder
        Fitted feature reweighting, PCA, and distilled MCCA pipeline.
    refinement_pca_ : PCA
        PCA feeding the residual branch; identical to linear_encoder_.pca_
        when the requested PCA dimensions match.
    refiner_ : NonlinearRefiner
        Fitted residual estimator.
    n_features_in_, n_components_ : int
        Input and output feature counts.
    """

    def __init__(
        self, *, n_components_pca: int = 768, n_components_mcca: int = 128,
        n_components_pca_refinement: int | None = None,
        feature_reweighting_kwargs: Mapping[str, Any] | None = None,
        pca_kwargs: Mapping[str, Any] | None = None,
        distilled_mcca_kwargs: Mapping[str, Any] | None = None,
        refiner_kwargs: Mapping[str, Any] | None = None,
        shuffle_views: bool = False, random_state: int | None = None,
    ) -> None:
        self.n_components_pca = n_components_pca
        self.n_components_mcca = n_components_mcca
        self.n_components_pca_refinement = n_components_pca_refinement
        self.feature_reweighting_kwargs = feature_reweighting_kwargs
        self.pca_kwargs = pca_kwargs
        self.distilled_mcca_kwargs = distilled_mcca_kwargs
        self.refiner_kwargs = refiner_kwargs
        self.shuffle_views = shuffle_views
        self.random_state = random_state

    def fit(
        self, X: ArrayLike, y: Any = None, *, sample_ids: ArrayLike,
        view_ids: ArrayLike | None = None,
    ) -> Self:
        """Fit both stages on training measurements; y is ignored."""
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        pca_kwargs = dict(self.pca_kwargs or {})
        mcca_kwargs = dict(self.distilled_mcca_kwargs or {})
        if "n_components" in pca_kwargs or "n_components" in mcca_kwargs:
            raise ValueError("Set component counts through NeuralEncoder's dimension parameters.")
        self.linear_encoder_ = LinearEncoder(
            feature_reweighting_kwargs=self.feature_reweighting_kwargs,
            pca_kwargs=pca_kwargs | {"n_components": self.n_components_pca},
            distilled_mcca_kwargs=mcca_kwargs | {"n_components": self.n_components_mcca},
            shuffle_views=self.shuffle_views, random_state=self.random_state,
        ).fit(X, sample_ids=sample_ids, view_ids=view_ids)
        dimension = self.n_components_pca_refinement
        if dimension is None or dimension == self.n_components_pca:
            self.refinement_pca_ = self.linear_encoder_.pca_
        else:
            self.refinement_pca_ = PCA(**(
                {"random_state": self.random_state} | pca_kwargs | {"n_components": dimension}
            ))
            self.refinement_pca_.fit(self.linear_encoder_.feature_reweighting_.transform(X))
        Z, source = self._representations(X)
        self.refiner_ = NonlinearRefiner(**(
            {"random_state": self.random_state} | dict(self.refiner_kwargs or {})
        )).fit(Z, network_input=source, sample_ids=sample_ids, view_ids=view_ids)
        self.n_components_ = Z.shape[1]
        return self

    def _representations(self, X: NDArray) -> tuple[NDArray, NDArray]:
        weighted = self.linear_encoder_.feature_reweighting_.transform(X)
        scores = self.linear_encoder_.pca_.transform(weighted)
        source = scores if self.refinement_pca_ is self.linear_encoder_.pca_ else self.refinement_pca_.transform(weighted)
        return self.linear_encoder_.distilled_mcca_.transform(scores), source

    def transform(self, X: ArrayLike) -> NDArray:
        """Return refined embeddings for individual measurements."""
        check_is_fitted(self, ["refiner_", "n_components_"])
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        Z, source = self._representations(X)
        return self.refiner_.transform(Z, network_input=source)

    def fit_transform(
        self, X: ArrayLike, y: Any = None, *, sample_ids: ArrayLike,
        view_ids: ArrayLike | None = None,
    ) -> NDArray:
        """Fit both stages and return training embeddings."""
        return self.fit(X, y, sample_ids=sample_ids, view_ids=view_ids).transform(X)

    def to_pytorch(self) -> NeuralEncoderModule:
        """Return the complete architecture in evaluation mode.

        Linear layers copy the fitted projections to the refiner's device and
        dtype. The residual module is the fitted network itself, without a copy.
        Changes to that module also affect this estimator. Equal PCA dimensions
        share one layer and one forward computation. All parameters are trainable.
        """
        check_is_fitted(self, ["refiner_", "n_components_"])

        def pca_layer(pca: PCA) -> nn.Linear:
            coef, intercept = _combine_projections(
                self.linear_encoder_.feature_reweighting_, pca, None, self.n_features_in_,
            )
            return _linear_layer(coef, intercept, self.refiner_.device_, self.refiner_.dtype)

        mcca = self.linear_encoder_.distilled_mcca_
        separate = None if self.refinement_pca_ is self.linear_encoder_.pca_ else pca_layer(self.refinement_pca_)
        model = NeuralEncoderModule(
            pca_layer(self.linear_encoder_.pca_),
            _linear_layer(mcca.coef_, mcca.intercept_, self.refiner_.device_, self.refiner_.dtype),
            self.refiner_.to_torch(), separate,
        )
        return model.eval()

    def to_torch(self) -> NeuralEncoderModule:
        """Alias for to_pytorch."""
        return self.to_pytorch()


def _linear_layer(coef: NDArray, intercept: NDArray, device: torch.device, dtype: torch.dtype) -> nn.Linear:
    layer = nn.Linear(coef.shape[1], coef.shape[0], device="meta", dtype=dtype)
    layer.weight = nn.Parameter(torch.tensor(coef, device=device, dtype=dtype))
    layer.bias = nn.Parameter(torch.tensor(intercept, device=device, dtype=dtype))
    return layer
