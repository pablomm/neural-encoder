import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from neural_encoder.utils.split_views import split_views


class TestSplitViews(unittest.TestCase):
    def test_shuffle_reproducibility_and_alignment(self):
        samples = np.repeat(np.arange(20), 3)
        view_ids = np.tile(np.arange(3), 20)
        X = np.column_stack((samples * 10 + view_ids, samples * 10 + view_ids + 1000))
        original = X.copy()
        baseline = np.stack(split_views(X, samples, view_ids))
        shuffled = np.stack(split_views(X, samples, view_ids, shuffle_views=True, seed=42))
        repeated = np.stack(split_views(X, samples, view_ids, shuffle_views=True, seed=42))
        different = np.stack(split_views(X, samples, view_ids, shuffle_views=True, seed=43))
        assert_array_equal(shuffled, repeated)
        self.assertFalse(np.array_equal(shuffled, different))
        assert_array_equal(np.sort(shuffled, axis=0), np.sort(baseline, axis=0))
        assert_array_equal(shuffled[..., 1] - shuffled[..., 0], 1000)
        self.assertGreater(len(np.unique(shuffled[..., 0].T % 10, axis=0)), 1)
        assert_array_equal(X, original)
        assert_array_equal(split_views(X, samples, view_ids, seed=42), baseline)

    def test_shuffle_with_missing_views_and_empty_output(self):
        for impute in (None, "zeros", "mean", "discard"):
            with self.subTest(impute=impute):
                kwargs = dict(n_views=3, impute=impute)
                baseline = np.stack(split_views([[1, 2], [3, 4]], [0, 0], [0, 1], **kwargs))
                shuffled = np.stack(split_views(
                    [[1, 2], [3, 4]], [0, 0], [0, 1],
                    shuffle_views=True, seed=0, **kwargs,
                ))
                assert_allclose(np.sort(shuffled, axis=0), np.sort(baseline, axis=0), equal_nan=True)
                self.assertEqual(shuffled.dtype, baseline.dtype)
        only = split_views([[1, 2]], [0], [0], shuffle_views=True, seed=0)
        assert_array_equal(only[0], [[1, 2]])

    def test_discard_incomplete_samples(self):
        X = np.array([[30], [12], [10], [40], [42]])
        views = split_views(X, [3, 1, 1, 4, 4], [0, 1, 0, 0, 1], impute="discard")
        assert_array_equal(views[0], [[10], [40]])
        assert_array_equal(views[1], [[12], [42]])
        self.assertEqual(views[0].dtype, X.dtype)

    def test_discard_no_complete_samples(self):
        for kwargs in ({}, {"n_views": 3}):
            views = split_views([[1], [2]], [0, 1], [0, 1], impute="discard", **kwargs)
            self.assertEqual(len(views), kwargs.get("n_views", 2))
            self.assertTrue(all(view.shape == (0, 1) for view in views))

    def test_discard_preserves_observed_nan(self):
        views = split_views([[np.nan], [2]], [0, 0], [0, 1], impute="discard")
        assert_allclose(views[0], [[np.nan]], equal_nan=True)
        assert_array_equal(views[1], [[2]])

    def test_alignment_and_missing(self):
        X = np.array([[30, 40], [14, 24], [10, 20]])
        a, b = split_views(X, ["b", "a", "a"], ["first", "second", "first"])
        assert_array_equal(a, [[10, 20], [30, 40]])
        assert_allclose(b, [[14, 24], [np.nan, np.nan]], equal_nan=True)
        a[0, 0] = 99
        assert_array_equal(X, [[30, 40], [14, 24], [10, 20]])

    def test_mean_uses_all_observed_views_per_sample(self):
        X = np.array([[2], [5], [11], [20]], dtype=np.float32)
        views = split_views(X, [0, 0, 0, 1], [1, 2, 3, 2], n_views=4, impute="mean")
        assert_allclose(views[3], [[6], [20]])
        assert_allclose(views[0], [[2], [20]])
        self.assertTrue(all(v.dtype == np.float32 for v in views))

    def test_zeros_and_integer_mean(self):
        views = split_views([[1], [2]], [0, 0], [0, 1], n_views=3, impute="zeros")
        assert_array_equal(views[2], [[0]])
        self.assertEqual(views[0].dtype.kind, "i")
        views = split_views([[1], [2]], [0, 0], [0, 1], n_views=3, impute="mean")
        assert_allclose(views[2], [[1.5]])

    def test_validation(self):
        cases = [
            ([[1], [2]], [0, 0], [0, 0], {}),
            ([[1], [2]], [0], [0, 1], {}),
            ([[1], [2]], [0, 1], [0, 1], {"n_views": 1}),
            ([[1]], [0], [0], {"n_views": 1.5}),
            ([[1]], [None], [0], {}),
            ([[1]], [0], [np.nan], {}),
            ([[1]], [0], [0], {"impute": "other"}),
            ([[1]], [0], [0], {"shuffle_views": "yes"}),
            ([[1]], [0], [0], {"seed": -1}),
            ([[1]], [0], [0], {"seed": 1.5}),
            ([], [], [], {}),
        ]
        for X, samples, views, kwargs in cases:
            with self.subTest(kwargs=kwargs, X=X), self.assertRaises(ValueError):
                split_views(X, samples, views, **kwargs)


if __name__ == "__main__":
    unittest.main()
