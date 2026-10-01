import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from neural_encoder.preprocessing import BetaPreprocessor


class TestBetaPreprocessor(unittest.TestCase):
    def test_feature_scaling_and_fitted_statistics(self):
        train = np.array([[1., 10., 5.], [3., 14., 5.]], dtype=np.float32)
        model = BetaPreprocessor(
            scaling=None, quantile_clip=None, feature_scaling=True,
        )
        result = model.fit_transform(train)
        assert_allclose(model.feature_scale_, [1, 2, 1])
        assert_allclose(result, [[-1, -1, 0], [1, 1, 0]])
        assert_allclose(model.transform([[5, 18, 5]]), [[3, 3, 0]])
        assert_allclose(model.feature_scale_, [1, 2, 1])
        self.assertEqual(result.dtype, np.float32)
        model.set_params(feature_centering=False).fit(train)
        assert_allclose(model.transform([[5, 18, 5]]), [[5, 9, 5]])
        model.set_params(feature_scaling=False).fit(train)
        self.assertIsNone(model.feature_scale_)

    def test_feature_scaling_order(self):
        model = BetaPreprocessor(
            scaling=2, quantile_clip=None, clip_bounds=(0, 8),
            sample_centering=True, feature_scaling=True, normalize=True,
        )
        train = np.array([[0., 4., 8.], [8., 20., 2.], [2., 6., 12.]])
        prepared = np.clip(train / 2, 0, 8)
        prepared -= prepared.mean(axis=1, keepdims=True)
        expected = (prepared - prepared.mean(axis=0)) / prepared.std(axis=0)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        assert_allclose(model.fit_transform(train), expected, atol=1e-6)
        assert_allclose(model.feature_scale_, prepared.std(axis=0), atol=1e-6)

    def test_fitted_statistics_and_batch_independence(self):
        train = np.array([[0., 2.], [4., 6.], [8., 10.]])
        original = train.copy()
        model = BetaPreprocessor(scaling=2, quantile_clip=0.2, feature_centering=True)
        model.fit(train)
        assert_allclose(model.clip_bounds_, [1, 4])
        assert_allclose(model.feature_mean_, [7 / 3, 8 / 3])
        test = np.array([[-100., 100.], [3., 7.]])
        result = model.transform(test)
        assert_allclose(result[0], [-4 / 3, 4 / 3])
        assert_allclose(result[:1], model.transform(test[:1]))
        assert_array_equal(train, original)
        assert_array_equal(test, [[-100, 100], [3, 7]])
        assert_allclose(model.clip_bounds_, [1, 4])

    def test_centering_order_and_normalization(self):
        model = BetaPreprocessor(
            scaling=None, quantile_clip=None, clip_bounds=(0, 8), sample_centering=True,
            feature_centering=True, normalize=True,
        )
        result = model.fit_transform([[0, 2], [2, 8], [4, 8]])
        assert_allclose(model.feature_mean_, [-2, 2])
        assert_allclose(result, [[2**-0.5, -2**-0.5],
                                 [-2**-0.5, 2**-0.5], [0, 0]], atol=1e-7)

    def test_defaults_nan_policy_and_refit(self):
        model = BetaPreprocessor()
        train = np.array([[0., 300.], [600., 900.]], dtype=np.float32)
        scaled = train / 300
        bounds = np.quantile(scaled, [0.0005, 0.9995])
        clipped = np.clip(scaled, *bounds)
        assert_allclose(model.fit_transform(train), clipped - clipped.mean(axis=0), atol=1e-7)
        assert_allclose(model.clip_bounds_, bounds)
        assert_allclose(model.feature_mean_, clipped.mean(axis=0))
        self.assertFalse(model.sample_centering)
        model = BetaPreprocessor(scaling=None, quantile_clip=None, feature_centering=False)
        assert_array_equal(model.fit_transform([[1, 2], [3, 4]]), [[1, 2], [3, 4]])
        assert_array_equal(model.transform([[np.nan, 1]]), [[0, 1]])
        with self.assertRaises(ValueError):
            BetaPreprocessor(fill_value=None).fit([[1, 2]]).transform([[np.nan, 1]])
        filled = BetaPreprocessor(fill_value=0, scaling=2, quantile_clip=None, feature_centering=False)
        assert_array_equal(filled.fit_transform([[np.nan, 2]]), [[0, 1]])
        with self.assertRaises(ValueError):
            filled.transform([[np.inf, 1]])
        model.set_params(quantile_clip=(0.1, 0.9)).fit([[0, 10], [20, 30]])
        assert_allclose(model.clip_bounds_, [3, 27])
        model.set_params(quantile_clip=None).fit([[1, 2]])
        self.assertIsNone(model.clip_bounds_)

    def test_validation(self):
        with self.assertRaises(NotFittedError):
            BetaPreprocessor().transform([[1, 2]])
        for params in [dict(scaling=0), dict(quantile_clip=0.6),
                       dict(quantile_clip=(-0.1, 0.9)), dict(clip_bounds=(2, 1)),
                       dict(quantile_clip=0.1, clip_bounds=(0, 1)), dict(dtype=int)]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                BetaPreprocessor(**params).fit([[1, 2]])
        with self.assertRaises(ValueError):
            BetaPreprocessor().fit([[1, 2]]).transform([[1, 2, 3]])

    def test_pipeline_clone_and_large_norms(self):
        model = BetaPreprocessor(
            scaling=None, quantile_clip=None, feature_centering=False,
            normalize=True, dtype=np.float64,
        )
        pipeline = Pipeline([("preprocessing", clone(model))])
        result = pipeline.fit_transform([[1e200, 1e200], [0, 0]])
        assert_allclose(result, [[2**-0.5, 2**-0.5], [0, 0]])
        self.assertFalse(hasattr(model, "n_features_in_"))


if __name__ == "__main__":
    unittest.main()
