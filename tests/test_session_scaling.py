import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from neural_encoder.utils import SessionStandardScaler


class TestSessionStandardScaler(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.sessions = np.repeat([0, 1, 2], [40, 30, 5])
        offsets = np.array([[0., 10.], [5., -3.], [1., 1.]])
        gains = np.array([[1., 2.], [4., 0.5], [1., 1.]])
        noise = rng.normal(size=(75, 2))
        self.X = noise * gains[self.sessions] + offsets[self.sessions]

    def test_per_session_statistics(self):
        scaler = SessionStandardScaler(min_samples=10, dtype=np.float64).fit(self.X, sessions=self.sessions)
        assert_array_equal(scaler.sessions_, [0, 1])
        for i, session in enumerate((0, 1)):
            rows = self.X[self.sessions == session]
            assert_allclose(scaler.means_[i], rows.mean(axis=0))
            assert_allclose(scaler.scales_[i], rows.std(axis=0))
        assert_allclose(scaler.mean_, self.X.mean(axis=0))
        result = scaler.transform(self.X, sessions=self.sessions)
        for session in (0, 1):
            rows = result[self.sessions == session]
            assert_allclose(rows.mean(axis=0), 0, atol=1e-12)
            assert_allclose(rows.std(axis=0), 1, atol=1e-12)
        # Session 2 has fewer than min_samples rows: pooled statistics.
        expected = (self.X[self.sessions == 2] - scaler.mean_) / scaler.scale_
        assert_allclose(result[self.sessions == 2], expected)

    def test_held_out_rows_use_training_statistics(self):
        scaler = SessionStandardScaler(min_samples=10, dtype=np.float64).fit(self.X, sessions=self.sessions)
        new = np.array([[1., 2.], [3., 4.], [5., 6.]])
        result = scaler.transform(new, sessions=[1, 0, 7])
        assert_allclose(result[0], (new[0] - scaler.means_[1]) / scaler.scales_[1])
        assert_allclose(result[1], (new[1] - scaler.means_[0]) / scaler.scales_[0])
        assert_allclose(result[2], (new[2] - scaler.mean_) / scaler.scale_)
        # Rows are transformed independently of the batch they arrive in.
        assert_allclose(scaler.transform(new[:1], sessions=[1]), result[:1])

    def test_arbitrary_session_labels(self):
        reference = SessionStandardScaler(min_samples=10).fit(self.X, sessions=self.sessions)
        expected = reference.transform(self.X, sessions=self.sessions)
        mappings = ({0: 1, 1: 2, 2: 3}, {0: 1000, 1: 7, 2: 40}, {0: "s01", 1: "s02", 2: "s03"})
        for mapping in mappings:
            with self.subTest(labels=list(mapping.values())):
                labels = np.array([mapping[s] for s in self.sessions])
                scaler = SessionStandardScaler(min_samples=10).fit(self.X, sessions=labels)
                assert_array_equal(np.sort(scaler.sessions_), sorted([mapping[0], mapping[1]]))
                assert_array_equal(scaler.transform(self.X, sessions=labels), expected)
        scaler = SessionStandardScaler(min_samples=10).fit(self.X, sessions=[f"s{s}" for s in self.sessions])
        assert_allclose(scaler.transform(self.X[:1], sessions=["unseen"]),
                        (self.X[:1] - scaler.mean_) / scaler.scale_, rtol=1e-6)

    def test_options_dtype_and_copy(self):
        X = self.X.copy()
        scaler = SessionStandardScaler(min_samples=10, with_std=False).fit(X, sessions=self.sessions)
        result = scaler.transform(X, sessions=self.sessions)
        self.assertEqual(result.dtype, np.float32)
        assert_allclose(result[:40], (X[:40] - X[:40].mean(axis=0)).astype(np.float32), atol=1e-5)
        assert_array_equal(X, self.X)
        unscaled = SessionStandardScaler(with_mean=False, with_std=False).fit_transform(X, sessions=self.sessions)
        assert_allclose(unscaled, X.astype(np.float32))
        constant = SessionStandardScaler(min_samples=2).fit_transform(np.ones((4, 2)), sessions=[0, 0, 1, 1])
        assert_array_equal(constant, np.zeros((4, 2)))

    def test_clone_and_validation(self):
        template = SessionStandardScaler(min_samples=3)
        fitted = clone(template).fit(self.X, sessions=self.sessions)
        self.assertFalse(hasattr(template, "means_"))
        self.assertEqual(fitted.n_features_in_, 2)
        with self.assertRaises(NotFittedError):
            SessionStandardScaler().transform(self.X, sessions=self.sessions)
        with self.assertRaises(ValueError):
            SessionStandardScaler().fit(self.X, sessions=self.sessions[:-1])
        with self.assertRaises(ValueError):
            fitted.transform(np.ones((2, 3)), sessions=[0, 0])
        for params in ({"min_samples": 0}, {"with_mean": "yes"}, {"dtype": np.int32}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                SessionStandardScaler(**params).fit(self.X, sessions=self.sessions)
        with self.assertRaises(ValueError):
            SessionStandardScaler().fit(np.array([[np.nan, 1.], [1., 2.]]), sessions=[0, 0])


if __name__ == "__main__":
    unittest.main()
