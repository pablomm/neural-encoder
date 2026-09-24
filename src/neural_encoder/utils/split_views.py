"""Operations for organizing sample-by-feature data."""

from numbers import Integral
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["split_views"]


def split_views(
    X: ArrayLike,
    sample_ids: ArrayLike,
    view_ids: ArrayLike | None = None,
    *,
    n_views: int | None = None,
    impute: Literal["zeros", "mean", "discard"] | None = None,
    shuffle_views: bool = False,
    seed: int | None = None,
) -> tuple[NDArray[Any], ...]:
    """Split measurements into aligned sample-by-feature view matrices.

    Parameters
    ----------
    X : array-like of shape (n_measurements, n_features)
        Real numeric measurements. Inputs are not modified.
    sample_ids : array-like of shape (n_measurements,)
        Sample identities, such as stimulus IDs. Output rows follow sorted
        unique sample IDs. IDs must be nonmissing, mutually sortable scalars.
    view_ids : array-like of shape (n_measurements,) or None, default=None
        View identities, such as repetition IDs. Outputs follow sorted unique
        view IDs. Each (sample, view) pair must occur at most once.
        When omitted, views are assigned by order of appearance within each
        sample: first occurrence to view 0, second to view 1, and so on.
    n_views : int or None, default=None
        Number of output matrices. None uses the number of observed views.
        Without view_ids, this is the largest number of occurrences per sample.
        Must be at least that number. Extra views are appended as entirely
        missing matrices. IDs are labels, not zero-based output indices.
    impute : {None, "zeros", "mean", "discard"}, default=None
        How to fill missing sample-view measurements. None leaves NaNs;
        "zeros" fills with zeros; "mean" uses that sample's mean across its
        observed views. Existing values, including NaNs, are not replaced.
        NaNs in observed views propagate through mean imputation.
        "discard" keeps only samples present in every requested view; it
        checks measurement presence, not individual feature values. If no
        samples are complete, each returned matrix has zero rows.
    shuffle_views : bool, default=False
        Independently permute view assignments within each sample after
        imputation or discarding. Entire feature vectors move together;
        sample ordering is unchanged. Missing or imputed entries are shuffled
        too. Output positions no longer correspond to the original view IDs.
    seed : int or None, default=None
        Nonnegative random seed for reproducible shuffling. None uses fresh
        randomness. The global NumPy random state is not modified.

    Returns
    -------
    views : tuple of ndarray, each of shape (n_samples, n_features)
        One matrix per view, with identical sample ordering. Integer inputs
        are promoted to floating point for NaNs or mean imputation; floating
        inputs retain their dtype. Zero imputation and discarding preserve
        the input dtype.

    Examples
    --------
    >>> X = [[10, 20], [30, 40], [14, 24]]
    >>> first, second = split_views(
    ...     X, sample_ids=[0, 1, 0], view_ids=[0, 0, 1], impute="mean"
    ... )
    >>> second
    array([[14., 24.],
           [30., 40.]])
    """
    X = np.asarray(X)
    if X.ndim != 2 or not all(X.shape):
        raise ValueError("X must be a nonempty two-dimensional matrix.")
    if X.dtype.kind not in "iuf":
        raise ValueError("X must contain real numeric measurements.")
    if impute not in (None, "zeros", "mean", "discard"):
        raise ValueError("impute must be None, 'zeros', 'mean', or 'discard'.")
    if not isinstance(shuffle_views, (bool, np.bool_)):
        raise ValueError("shuffle_views must be a boolean.")
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer or None.")

    samples, sample_index = _encode_ids(sample_ids, "sample_ids", len(X))
    if view_ids is None:
        view_index = _views_by_occurrence(sample_index)
        observed_views = int(view_index.max()) + 1
    else:
        views, view_index = _encode_ids(view_ids, "view_ids", len(X))
        observed_views = len(views)
    if n_views is None:
        n_views = observed_views
    elif isinstance(n_views, (bool, np.bool_)) or not isinstance(n_views, Integral) or n_views < observed_views:
        raise ValueError("n_views must be an integer at least the number of observed views.")

    n_samples = len(samples)
    occupied = np.zeros((n_views, n_samples), dtype=bool)
    occupied[view_index, sample_index] = True
    if occupied.sum() != len(X):
        raise ValueError("Each (sample_id, view_id) pair must be unique.")

    handlers = {
        None: _impute_nan,
        "zeros": _impute_zeros,
        "mean": _impute_mean,
        "discard": _discard_incomplete,
    }
    output = handlers[impute](X, sample_index, view_index, occupied)
    if shuffle_views:
        output = _shuffle_views(output, seed)
    return tuple(output)


def _views_by_occurrence(sample_index: NDArray[np.intp]) -> NDArray[np.intp]:
    """Return each measurement's zero-based occurrence within its sample."""
    order = np.argsort(sample_index, kind="stable")
    counts = np.bincount(sample_index)
    starts = np.cumsum(counts) - counts
    view_index = np.empty_like(sample_index)
    view_index[order] = np.arange(len(sample_index)) - np.repeat(starts, counts)
    return view_index


def _shuffle_views(views: NDArray[Any], seed: int | None) -> NDArray[Any]:
    """Permute whole measurements independently within each sample."""
    rng = np.random.default_rng(seed)
    indices = np.broadcast_to(np.arange(views.shape[0])[:, None], views.shape[:2])
    permutations = rng.permuted(indices, axis=0)
    return np.take_along_axis(views, permutations[..., None], axis=0)


def _encode_ids(ids: ArrayLike, name: str, n_measurements: int) -> tuple[NDArray[Any], NDArray[np.intp]]:
    ids = np.asarray(ids)
    if ids.ndim != 1 or len(ids) != n_measurements:
        raise ValueError(f"{name} must be one-dimensional with len(X) entries.")
    if ids.dtype.kind in "fc" and not np.isfinite(ids).all():
        raise ValueError(f"{name} must not contain missing or infinite IDs.")
    if ids.dtype.kind in "mM" and np.isnat(ids).any():
        raise ValueError(f"{name} must not contain missing IDs.")
    if ids.dtype.kind == "O":
        for value in ids:
            if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
                raise ValueError(f"{name} must not contain missing or infinite IDs.")
    try:
        return np.unique(ids, return_inverse=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain mutually sortable scalar IDs.") from exc


def _filled_views(
    X: NDArray[Any],
    sample_index: NDArray[np.intp],
    view_index: NDArray[np.intp],
    occupied: NDArray[np.bool_],
    *,
    fill_value: float,
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    """Allocate aligned views and insert observed measurements."""
    output = np.full((*occupied.shape, X.shape[1]), fill_value, dtype=dtype)
    output[view_index, sample_index] = X
    return output


def _impute_nan(
    X: NDArray[Any],
    sample_index: NDArray[np.intp],
    view_index: NDArray[np.intp],
    occupied: NDArray[np.bool_],
) -> NDArray[Any]:
    """Leave missing measurements as NaNs."""
    dtype = X.dtype if X.dtype.kind == "f" else np.dtype(np.float64)
    return _filled_views(
        X, sample_index, view_index, occupied, fill_value=np.nan, dtype=dtype,
    )


def _impute_zeros(
    X: NDArray[Any],
    sample_index: NDArray[np.intp],
    view_index: NDArray[np.intp],
    occupied: NDArray[np.bool_],
) -> NDArray[Any]:
    """Fill missing measurements with zeros, preserving the input dtype."""
    return _filled_views(
        X, sample_index, view_index, occupied, fill_value=0, dtype=X.dtype,
    )


def _impute_mean(
    X: NDArray[Any],
    sample_index: NDArray[np.intp],
    view_index: NDArray[np.intp],
    occupied: NDArray[np.bool_],
) -> NDArray[Any]:
    """Fill missing measurements with each sample's observed-view mean."""
    dtype = X.dtype if X.dtype.kind == "f" else np.dtype(np.float64)
    output = _filled_views(
        X, sample_index, view_index, occupied, fill_value=0, dtype=dtype,
    )
    n_samples = occupied.shape[1]
    sums = np.zeros((n_samples, X.shape[1]), dtype=np.float64)
    np.add.at(sums, sample_index, X)
    means = sums / np.bincount(sample_index, minlength=n_samples)[:, None]
    missing_views, missing_samples = np.nonzero(~occupied)
    output[missing_views, missing_samples] = means[missing_samples]
    return output


def _discard_incomplete(
    X: NDArray[Any],
    sample_index: NDArray[np.intp],
    view_index: NDArray[np.intp],
    occupied: NDArray[np.bool_],
) -> NDArray[Any]:
    """Build views containing only samples observed in every view."""
    complete = occupied.all(axis=0)
    retained = complete[sample_index]
    compact_index = np.cumsum(complete) - 1
    output = np.empty(
        (occupied.shape[0], int(complete.sum()), X.shape[1]), dtype=X.dtype,
    )
    output[view_index[retained], compact_index[sample_index[retained]]] = X[retained]
    return output
