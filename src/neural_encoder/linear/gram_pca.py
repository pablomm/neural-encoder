"""Exact PCA from the sample Gram matrix, computed in feature chunks."""

from numbers import Integral
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

__all__ = ["GramPCA"]


class GramPCA(TransformerMixin, BaseEstimator):
    """Exact PCA for data with many more features than samples.

    The centered training matrix X (n_samples, n_features) has the same
    principal components as its sample Gram matrix K = X X^T, which is only
    n_samples x n_samples. K is accumulated over chunks of features, its
    leading eigenvectors U and eigenvalues S^2 give the training scores
    U S, and the components V = X^T U S^-1 are computed chunk by chunk in a
    second pass. The full matrix is never copied or moved to the device at
    once, so X may be a NumPy memory map. Results match
    sklearn.decomposition.PCA with a full SVD, including its sign convention:
    the largest-magnitude entry of each component is positive.

    Computation uses PyTorch and runs on the CPU or a GPU.

    Parameters
    ----------
    n_components : int, default=768
        Number of components to keep. At most min(n_samples, n_features).
    whiten : bool, default=False
        Divide scores by the square root of the explained variance, as in
        sklearn.decomposition.PCA.
    chunk_size : int, default=8192
        Number of features loaded at once. Device memory holds K
        (n_samples^2 float64 values) plus one n_samples x chunk_size chunk;
        the eigendecomposition then needs workspace of a few times the size
        of K.
    dtype : {"float64", "float32"}, default="float64"
        Precision of the chunk products, the eigendecomposition, and the
        fitted components. K is always accumulated in float64. "float32" is
        several times faster on most GPUs: it decomposes K in float32 and
        then refines the leading eigenvectors with one float64
        Rayleigh-Ritz step, which keeps them close to float64 accuracy.
    device : str or torch.device or None, default=None
        Device for the computation, such as "cuda". None uses the CPU.
        Fitted attributes are always NumPy arrays.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes, sorted by decreasing explained variance.
    mean_ : ndarray of shape (n_features,)
        Training feature means.
    explained_variance_ : ndarray of shape (n_components,)
        Variance of each component, with n_samples - 1 degrees of freedom.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of the total training variance explained by each component.
    singular_values_ : ndarray of shape (n_components,)
        Singular values of the centered training matrix.
    n_components_ : int
        Number of fitted components.
    n_samples_ : int
        Number of training samples.
    n_features_in_ : int
        Number of features observed during fit.

    Notes
    -----
    Building K costs about n_samples^2 x n_features operations and its
    eigendecomposition about n_samples^3, so the method suits data with
    up to a few tens of thousands of samples and any number of features.
    For row-major memory maps, reading column chunks is strided; storing
    the data in Fortran order makes each chunk contiguous.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(0).normal(size=(20, 1000))
    >>> pca = GramPCA(n_components=5, chunk_size=256)
    >>> pca.fit_transform(X).shape
    (20, 5)
    """

    def __init__(
        self,
        n_components: int = 768,
        *,
        whiten: bool = False,
        chunk_size: int = 8192,
        dtype: Literal["float64", "float32"] = "float64",
        device: Any = None,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.device = device

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Fit the components. The optional y argument is ignored."""
        self._fit(X)
        return self

    def fit_transform(self, X: ArrayLike, y: Any = None) -> NDArray[Any]:
        """Fit the components and return the training scores.

        The scores come from the Gram eigenvectors, so X is not projected
        a third time.
        """
        return self._fit(X)

    def transform(self, X: ArrayLike) -> NDArray[Any]:
        """Project measurements onto the fitted components, in feature chunks."""
        import torch

        check_is_fitted(self, ["components_", "mean_"])
        X = _as_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but GramPCA is expecting "
                f"{self.n_features_in_} features."
            )
        device, dtype = self._torch_options()
        scores = torch.zeros((X.shape[0], self.n_components_), dtype=torch.float64, device=device)
        for chunk_slice in self._chunks():
            chunk = _load_chunk(X, chunk_slice, device, dtype)
            chunk -= torch.as_tensor(self.mean_[chunk_slice], dtype=dtype, device=device)
            components = torch.as_tensor(self.components_[:, chunk_slice], device=device)
            scores += (chunk @ components.T).to(torch.float64)
        if self.whiten:
            scores /= torch.as_tensor(np.sqrt(self.explained_variance_), device=device)
        return scores.cpu().numpy().astype(self.dtype, copy=False)

    def _fit(self, X: ArrayLike) -> NDArray[Any]:
        self._validate_parameters()
        X = _as_matrix(X)
        device, dtype = self._torch_options()
        return self._fit_chunks(lambda chunk_slice: _load_chunk(X, chunk_slice, device, dtype), *X.shape)

    def _fit_chunks(self, load: Any, n_samples: int, n_features: int) -> NDArray[Any]:
        """Fit from load(chunk_slice), which returns a new writable device tensor.

        load is called twice for every chunk of features, once per pass, and
        must return the same values both times. Returns the training scores.
        """
        import torch

        self._validate_parameters()
        if n_samples < 2:
            raise ValueError("At least two samples are required.")
        k = self.n_components
        if k > min(n_samples, n_features):
            raise ValueError(
                f"n_components={k} must be at most min(n_samples, n_features)="
                f"{min(n_samples, n_features)}."
            )
        self.n_features_in_ = n_features
        device, dtype = self._torch_options()

        # Pass 1: feature means and the Gram matrix of the centered data.
        mean = np.empty(n_features)
        gram = torch.zeros((n_samples, n_samples), dtype=torch.float64, device=device)
        for chunk_slice in self._chunks():
            chunk = load(chunk_slice)
            chunk_mean = chunk.sum(dim=0, dtype=torch.float64) / n_samples
            chunk -= chunk_mean.to(dtype)
            gram += (chunk @ chunk.T).to(torch.float64)
            mean[chunk_slice] = chunk_mean.cpu().numpy()
        del chunk

        total_variance = torch.trace(gram)
        eigenvalues, U = _top_eigenpairs(gram, k, dtype)
        del gram
        eigenvalues = eigenvalues.clamp_min(0)
        # Eigenvalues within rounding error of zero are directions with no
        # variance; they get zero components instead of amplified noise.
        tolerance = eigenvalues[0] * n_samples * torch.finfo(torch.float64).eps
        eigenvalues = torch.where(eigenvalues > tolerance, eigenvalues, torch.zeros_like(eigenvalues))
        singular = eigenvalues.sqrt()
        inverse = torch.where(singular > 0, 1 / singular, torch.zeros_like(singular))

        # Pass 2: components V = X^T U S^-1, one chunk of features at a time.
        projector = (U * inverse).to(dtype)
        components = np.empty((k, n_features), dtype=self.dtype)
        for chunk_slice in self._chunks():
            chunk = load(chunk_slice)
            chunk -= torch.as_tensor(mean[chunk_slice], dtype=dtype, device=device)
            components[:, chunk_slice] = (chunk.T @ projector).T.cpu().numpy()

        # sklearn's convention: the largest-magnitude entry of each component is positive.
        largest = components[np.arange(k), np.abs(components).argmax(axis=1)]
        signs = np.where(largest < 0, -1.0, 1.0)
        components *= signs[:, None].astype(self.dtype)

        self.components_ = components
        self.mean_ = mean
        self.explained_variance_ = (eigenvalues / (n_samples - 1)).cpu().numpy()
        self.explained_variance_ratio_ = (eigenvalues / total_variance).cpu().numpy()
        self.singular_values_ = singular.cpu().numpy()
        self.n_components_ = k
        self.n_samples_ = n_samples

        U = U * torch.as_tensor(signs, device=device)
        scores = U * (np.sqrt(n_samples - 1) if self.whiten else singular)
        return scores.cpu().numpy().astype(self.dtype, copy=False)

    def _validate_parameters(self) -> None:
        for name in ("n_components", "chunk_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.whiten, (bool, np.bool_)):
            raise ValueError("whiten must be a boolean.")
        if self.dtype not in ("float64", "float32"):
            raise ValueError("dtype must be 'float64' or 'float32'.")

    def _torch_options(self) -> tuple[Any, Any]:
        import torch

        return torch.device(self.device or "cpu"), getattr(torch, self.dtype)

    def _chunks(self) -> list[slice]:
        return [
            slice(start, min(start + self.chunk_size, self.n_features_in_))
            for start in range(0, self.n_features_in_, self.chunk_size)
        ]


def _top_eigenpairs(gram: Any, k: int, dtype: Any) -> tuple[Any, Any]:
    """Return the k largest eigenvalues of gram and their eigenvectors, descending.

    float64 decomposes gram directly. float32 decomposes a float32 copy,
    which is several times faster on GPUs, then recovers float64 accuracy
    with one subspace-iteration and Rayleigh-Ritz step on k + k // 4 vectors.
    Without that step, components with small eigenvalue gaps mix with their
    neighbours.
    """
    import torch

    if dtype == torch.float64:
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        return eigenvalues.flip(0)[:k], eigenvectors.flip(1)[:, :k]
    n_basis = min(gram.shape[0], k + max(64, k // 4))
    _, eigenvectors = torch.linalg.eigh(gram.to(torch.float32))
    basis = eigenvectors[:, -n_basis:].to(torch.float64)
    del eigenvectors
    basis, _ = torch.linalg.qr(gram @ basis)
    eigenvalues, rotation = torch.linalg.eigh(basis.T @ gram @ basis)
    return eigenvalues.flip(0)[:k], (basis @ rotation).flip(1)[:, :k]


def _as_matrix(X: ArrayLike) -> NDArray[Any]:
    """Return X as a 2D array without copying arrays or memory maps."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array, got an array with shape {X.shape}.")
    if X.dtype.kind not in "iuf":
        raise ValueError("X must contain real numbers.")
    return X


def _load_chunk(
    X: NDArray[Any], chunk_slice: slice, device: Any, dtype: Any, *,
    rows: NDArray[np.intp] | None = None, fill_value: float | None = None,
) -> Any:
    """Copy one block of features, optionally of selected rows, to the device.

    The block is transferred in its stored dtype (for example int16) and
    converted on the device. The result never shares memory with X. NaNs
    are replaced by fill_value when it is given; other nonfinite values
    raise an error.
    """
    import torch

    block = X[:, chunk_slice]
    if block.T.flags.c_contiguous and block.size:
        # Fortran-ordered storage: copy the contiguous columns as they are,
        # then transpose and select rows on the device.
        chunk = torch.from_numpy(np.array(block.T)).to(device).T
        if rows is not None:
            chunk = chunk[torch.as_tensor(rows, device=device)]
        chunk = chunk.to(dtype).contiguous()
    else:
        if rows is not None:
            block = X[rows, chunk_slice]
        block = torch.from_numpy(np.require(block, requirements=["C_CONTIGUOUS", "WRITEABLE"]))
        chunk = block.to(device).to(dtype, copy=True)
    if chunk.is_floating_point():
        if fill_value is not None:
            chunk.nan_to_num_(nan=fill_value, posinf=np.inf, neginf=-np.inf)
        if not torch.isfinite(chunk).all():
            raise ValueError("X must contain only finite values.")
    return chunk
