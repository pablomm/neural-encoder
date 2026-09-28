import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, rankdata, spearmanr

from neural_encoder.utils import compute_rsa, evaluate_pair, evaluate_views, retrieval_metrics


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.X = rng.normal(size=(12, 5))
        self.Y = rng.normal(size=(12, 5))

    def test_perfect_and_permuted_retrieval(self):
        X = np.eye(4)
        result = retrieval_metrics(X, X)
        self.assertEqual(result, {"mean_rank": 1., "recall@1": 1., "cosine": 1.})
        result = retrieval_metrics(X, X[[1, 2, 3, 0]])
        self.assertEqual(result, {"mean_rank": 3., "recall@1": 0., "cosine": 0.})

    def test_retrieval_reference_direction(self):
        expected_matrix = (self.X / np.linalg.norm(self.X, axis=1, keepdims=True)) @ (
            self.Y / np.linalg.norm(self.Y, axis=1, keepdims=True)).T
        expected_ranks = np.diag(rankdata(-expected_matrix, axis=1))
        actual, similarities = retrieval_metrics(
            self.X, self.Y, return_similarity_matrix=True,
        )
        assert_allclose(similarities, expected_matrix, atol=1e-15)
        assert_allclose(actual["mean_rank"], expected_ranks.mean())
        assert_allclose(actual["recall@1"], np.mean(expected_matrix.argmax(axis=1) == np.arange(12)))
        assert_allclose(actual["cosine"], np.diag(expected_matrix).mean())
        reverse = retrieval_metrics(self.Y, self.X)
        assert_allclose(reverse["mean_rank"], np.diag(rankdata(-expected_matrix.T, axis=1)).mean())

    def test_ties_and_zero_vectors(self):
        result = retrieval_metrics(np.zeros((4, 2)), np.zeros((4, 2)))
        self.assertEqual(result, {"mean_rank": 2.5, "recall@1": 0.25, "cosine": 0.})
        duplicate = retrieval_metrics(np.ones((3, 2)), np.ones((3, 2)))
        assert_allclose(duplicate["recall@1"], 1 / 3)
        assert_allclose(duplicate["mean_rank"], 2)

    def test_rsa_reference_metrics_and_different_feature_counts(self):
        Y = np.column_stack((self.Y, self.X[:, 0]))
        for first in ("pearson", "cosine", "euclidean"):
            a = pdist(self.X, metric="correlation" if first == "pearson" else first)
            b = pdist(Y, metric="correlation" if first == "pearson" else first)
            expected = {
                "pearson": pearsonr(a, b).statistic,
                "spearman": spearmanr(a, b).statistic,
                "cosine": a @ b / (np.linalg.norm(a) * np.linalg.norm(b)),
                "euclidean": np.linalg.norm(a - b),
            }
            for second, value in expected.items():
                with self.subTest(first=first, second=second):
                    assert_allclose(compute_rsa(self.X, Y, first_metric=first, second_metric=second), value, atol=1e-13)

    def test_rsa_spearman_ties_and_undefined(self):
        X = np.array([[0.], [1.], [2.], [3.]])
        Y = np.array([[0.], [2.], [1.], [3.]])
        expected = spearmanr(pdist(X), pdist(Y)).statistic
        assert_allclose(compute_rsa(X, Y, first_metric="euclidean", second_metric="spearman"), expected)
        self.assertTrue(np.isnan(compute_rsa(np.eye(4), np.eye(4), first_metric="cosine")))
        with self.assertRaisesRegex(ValueError, "constant rows"):
            compute_rsa(np.ones((4, 3)), np.ones((4, 3)))
        X = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        self.assertAlmostEqual(compute_rsa(X, X, first_metric="cosine"), 1)

    def test_multiview_ordered_pairs_and_means(self):
        views = [self.X, self.Y, self.X + self.Y]
        means, pairs = evaluate_views(views, return_pairs=True)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(set(zip(pairs.query_view, pairs.candidate_view)),
                         {(i, j) for i in range(3) for j in range(3) if i != j})
        for _, row in pairs.iterrows():
            metrics = evaluate_pair(views[int(row.query_view)], views[int(row.candidate_view)])
            for name, value in metrics.items():
                assert_allclose(row[name], value)
        for name, value in means.items():
            assert_allclose(value, pairs[name].mean())
        self.assertEqual(means, evaluate_views(views))
        undefined = evaluate_views([np.eye(4), np.eye(4)], first_metric="cosine")
        self.assertTrue(np.isnan(undefined["rsa"]))

    def test_inputs_unchanged_and_torch(self):
        import torch

        original = self.X.copy()
        expected = evaluate_pair(self.X, self.Y)
        actual = evaluate_pair(torch.tensor(self.X, requires_grad=True), torch.tensor(self.Y))
        for name in expected:
            assert_allclose(actual[name], expected[name])
        assert_array_equal(self.X, original)

    def test_validation(self):
        with self.assertRaises(ValueError):
            retrieval_metrics(self.X, self.Y[:, :2])
        with self.assertRaises(ValueError):
            evaluate_views([self.X])
        with self.assertRaises(ValueError):
            evaluate_views([self.X, self.Y[:-1]])
        with self.assertRaises(ValueError):
            compute_rsa(self.X[:2], self.Y[:2])
        with self.assertRaises(ValueError):
            compute_rsa(self.X, self.Y, first_metric="invalid")
        with self.assertRaises(ValueError):
            compute_rsa(self.X, self.Y, second_metric="invalid")
        with self.assertRaises(ValueError):
            retrieval_metrics([[np.nan]], [[1]])


if __name__ == "__main__":
    unittest.main()
