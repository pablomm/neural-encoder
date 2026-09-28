"""Retrieval and representational similarity metrics for paired measurements."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.utils.validation import check_array

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["retrieval_metrics", "compute_rsa", "evaluate_pair", "evaluate_views"]

DistanceMetric = Literal["pearson", "cosine", "euclidean"]
ComparisonMetric = Literal["pearson", "spearman", "cosine", "euclidean"]


def retrieval_metrics(
    queries: ArrayLike,
    candidates: ArrayLike,
    *,
    return_similarity_matrix: bool = False,
) -> dict[str, float] | tuple[dict[str, float], NDArray[np.float64]]:
    """Evaluate cosine retrieval of paired candidates from query measurements.

    Both inputs have shape (n_samples, n_features); row i is the correct
    match for row i. Returns mean_rank (one-based, lower is better), recall@1
    (higher is better), and cosine (mean matched cosine similarity).

    Exact ties receive average rank. Recall@1 gives 1/k credit when the
    correct candidate is among k candidates tied for first place. Zero
    vectors have cosine zero. All values must be finite.

    If return_similarity_matrix is True, also return the similarity matrix,
    with queries as rows and candidates as columns.

    Array-like inputs and PyTorch tensors are accepted. Tensors are detached
    and copied to CPU as needed; evaluation uses NumPy float64 arrays.
    """
    X, Y = _paired_arrays(queries, candidates, same_features=True)
    similarities = np.clip(_normalize_rows(X) @ _normalize_rows(Y).T, -1, 1)
    metrics = _retrieval(similarities)
    return (metrics, similarities) if return_similarity_matrix else metrics


def compute_rsa(
    X: ArrayLike,
    Y: ArrayLike,
    *,
    first_metric: DistanceMetric = "pearson",
    second_metric: ComparisonMetric = "pearson",
) -> float:
    """Compare representational dissimilarity matrices (RDMs) of paired data.

    Rows must represent the same samples in the same order. Feature counts
    may differ. At least three samples are required. first_metric defines
    within-representation dissimilarity: 1-Pearson correlation, 1-cosine
    similarity, or Euclidean distance. second_metric compares the condensed
    upper triangles, excluding their diagonals, using Pearson correlation,
    Spearman correlation (average ranks for ties), cosine similarity, or
    Euclidean distance. Only the last comparison is lower-is-better.

    Constant rows are invalid for Pearson distance. Zero vectors use cosine
    zero for cosine distance. Undefined comparisons (constant RDMs for
    correlation or zero RDM norms for cosine) return NaN. Inputs must be finite.
    Condensed RDM storage is O(n_samples^2). PyTorch tensors are accepted and
    evaluated on CPU, as in retrieval_metrics.
    """
    X, Y = _paired_arrays(X, Y, same_features=False, min_samples=3)
    return _compare_rdms(_rdm(X, first_metric), _rdm(Y, first_metric), second_metric)


def evaluate_pair(
    queries: ArrayLike,
    candidates: ArrayLike,
    *,
    first_metric: DistanceMetric = "pearson",
    second_metric: ComparisonMetric = "pearson",
) -> dict[str, float]:
    """Return mean_rank, recall@1, cosine, and rsa for one directed pair.

    Inputs must have matching sample and feature counts, with at least three
    paired samples. Metric definitions follow retrieval_metrics and compute_rsa.
    """
    X, Y = _paired_arrays(queries, candidates, same_features=True, min_samples=3)
    result = _retrieval(np.clip(_normalize_rows(X) @ _normalize_rows(Y).T, -1, 1))
    result["rsa"] = _compare_rdms(_rdm(X, first_metric), _rdm(Y, first_metric), second_metric)
    return result


def evaluate_views(
    views: Sequence[ArrayLike],
    *,
    first_metric: DistanceMetric = "pearson",
    second_metric: ComparisonMetric = "pearson",
    return_pairs: bool = False,
) -> dict[str, float] | tuple[dict[str, float], "pd.DataFrame"]:
    """Average retrieval and RSA across all ordered pairs of distinct views.

    views contains at least two equally shaped matrices, with corresponding
    rows and features and at least three samples. Evaluates V*(V-1) directions;
    self-pairs are excluded. Retrieval is directional; RSA is symmetric.
    Each pair has equal weight in the arithmetic mean. Undefined RSA values
    propagate to the aggregate instead of being silently excluded.

    By default, return a dict with mean_rank, recall@1, cosine, and rsa.
    With return_pairs=True, return (means, dataframe), whose columns are
    query_view, candidate_view (zero-based indices), and the four metrics.
    Normalized views and condensed RDMs are computed once per view; RSA is
    computed once per unordered pair and reused for the reverse direction.
    """
    if len(views) < 2:
        raise ValueError("At least two views are required.")
    arrays = [_as_matrix(view, min_samples=3) for view in views]
    if any(view.shape != arrays[0].shape for view in arrays):
        raise ValueError("All views must have the same shape and aligned samples.")
    normalized = [_normalize_rows(view) for view in arrays]
    rdms = [_rdm(view, first_metric) for view in arrays]
    rsa_scores = {}
    rows = []
    for i in range(len(views)):
        for j in range(len(views)):
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key not in rsa_scores:
                rsa_scores[key] = _compare_rdms(rdms[i], rdms[j], second_metric)
            metrics = _retrieval(np.clip(normalized[i] @ normalized[j].T, -1, 1))
            metrics["rsa"] = rsa_scores[key]
            rows.append({"query_view": i, "candidate_view": j, **metrics})
    means = {name: float(np.mean([row[name] for row in rows]))
             for name in ("mean_rank", "recall@1", "cosine", "rsa")}
    if return_pairs:
        import pandas as pd

        return means, pd.DataFrame(rows)
    return means


def _as_matrix(X: ArrayLike, min_samples: int = 1) -> NDArray[np.float64]:
    if hasattr(X, "detach") and hasattr(X, "cpu"):
        X = X.detach().cpu().double().numpy()
    return check_array(X, dtype=np.float64, ensure_min_samples=min_samples)


def _paired_arrays(
    X: ArrayLike, Y: ArrayLike, *, same_features: bool, min_samples: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    X, Y = _as_matrix(X, min_samples), _as_matrix(Y, min_samples)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Inputs must have the same number of aligned samples.")
    if same_features and X.shape[1] != Y.shape[1]:
        raise ValueError("Retrieval requires the same number of features.")
    return X, Y


def _normalize_rows(X: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norms, out=np.zeros_like(X), where=norms != 0)


def _retrieval(similarities: NDArray[np.float64]) -> dict[str, float]:
    matched = np.diag(similarities)
    greater = (similarities > matched[:, None]).sum(axis=1)
    tied = (similarities == matched[:, None]).sum(axis=1)
    return {
        "mean_rank": float(np.mean(1 + greater + (tied - 1) / 2)),
        "recall@1": float(np.mean((greater == 0) / tied)),
        "cosine": float(matched.mean()),
    }


def _rdm(X: NDArray[np.float64], metric: DistanceMetric) -> NDArray[np.float64]:
    if metric == "euclidean":
        return pdist(X, metric="euclidean")
    if metric == "pearson":
        X = X - X.mean(axis=1, keepdims=True)
        if np.any(np.max(np.abs(X), axis=1) == 0):
            raise ValueError("Pearson RDMs are undefined for constant rows; use cosine or euclidean.")
    elif metric != "cosine":
        raise ValueError(f"Unknown distance metric: {metric!r}")
    normalized = _normalize_rows(X)
    distances = 1 - np.clip(normalized @ normalized.T, -1, 1)
    return distances[np.triu_indices(len(X), k=1)]


def _compare_rdms(
    first: NDArray[np.float64], second: NDArray[np.float64], metric: ComparisonMetric,
) -> float:
    if metric == "euclidean":
        return float(np.linalg.norm(first - second))
    if metric == "spearman":
        first, second = rankdata(first), rankdata(second)
    if metric in ("pearson", "spearman"):
        first, second = first - first.mean(), second - second.mean()
    elif metric != "cosine":
        raise ValueError(f"Unknown comparison metric: {metric!r}")
    if not np.any(first) or not np.any(second):
        return float("nan")
    first, second = _normalize_rows(first[None, :])[0], _normalize_rows(second[None, :])[0]
    return float(np.clip(first @ second, -1, 1))
