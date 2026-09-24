import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from neural_encoder.linear import FeatureReweighting


class TestFeatureReweighting(unittest.TestCase):
    def test_callable_weighting(self):
        views = [np.array([[1., 1.], [2., 1.], [3., 1.]])] * 2
        linear = FeatureReweighting(eps=0.01).fit_views(views)
        model = FeatureReweighting(weighting=lambda w: (w ** 2).tolist(), eps=0.01)
        model.fit_views(views)
        assert_allclose(model.weights_, linear.weights_ ** 2)
        assert_allclose(model.transform(views[0]), views[0] * linear.weights_ ** 2)
        assert_allclose(
            FeatureReweighting(weighting=np.sqrt).fit_views(views).weights_,
            FeatureReweighting(weighting="sqrt").fit_views(views).weights_,
        )
        def in_place(weights):
            weights[:] = 2
            return weights
        model = FeatureReweighting(weighting=in_place).fit_views(views)
        assert_allclose(model.reliability_, [1, 0])
        assert_allclose(model.weights_, [2, 2])

    def test_invalid_callable_weighting_output(self):
        views = [np.ones((3, 2))] * 2
        for result in (1, [1], [[1, 2]], [np.nan, 1], [np.inf, 1], [1j, 2], ["a", "b"]):
            with self.subTest(result=result), self.assertRaises(ValueError):
                FeatureReweighting(weighting=lambda w: result).fit_views(views)

    def test_correlations_weights_and_constant_features(self):
        first = np.array([[1, 1, 7], [2, 2, 7], [3, 3, 7]], dtype=float)
        second = np.array([[2, 3, 9], [4, 2, 9], [6, 1, 9]], dtype=float)
        model = FeatureReweighting().fit_views([first, second])
        assert_allclose(model.reliability_, [1, -1, 0])
        assert_allclose(model.weights_, [1 + 1e-6, 1e-6, 1e-6])
        original = first.copy()
        assert_allclose(model.transform(first), first * model.weights_)
        assert_array_equal(first, original)
        model = FeatureReweighting(weighting="sqrt", eps=0.01).fit_views([first, second])
        assert_allclose(model.weights_, np.sqrt([1.01, 0.01, 0.01]))
        assert_array_equal(model.get_feature_names_out(), ["x0", "x1", "x2"])

    def test_multiview_leave_one_out_reference(self):
        rng = np.random.default_rng(0)
        views = rng.normal(size=(4, 15, 5))
        expected = np.mean([
            [np.corrcoef(views[i, :, j], np.delete(views, i, axis=0).mean(axis=0)[:, j])[0, 1]
             for j in range(5)] for i in range(4)
        ], axis=0)
        model = FeatureReweighting().fit_views(views)
        assert_allclose(model.reliability_, expected, atol=1e-14)

    def test_matrix_fit_discards_incomplete_samples_but_transforms_all(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [6, 8], [1000, -1000]], dtype=float)
        samples = [0, 0, 1, 1, 2]
        model = FeatureReweighting()
        transformed = model.fit_transform(X, sample_ids=samples)
        reference = FeatureReweighting().fit_views([X[[0, 2]], X[[1, 3]]])
        assert_allclose(model.weights_, reference.weights_)
        assert_allclose(transformed, X * reference.weights_)
        self.assertEqual(model.n_samples_, 2)
        self.assertEqual(model.n_views_, 2)
        order = [3, 0, 4, 2, 1]
        explicit = FeatureReweighting().fit(
            X[order], sample_ids=np.array(samples)[order],
            view_ids=np.array([0, 1, 0, 1, 0])[order],
        )
        assert_allclose(explicit.weights_, reference.weights_)

    def test_pipeline_and_clone(self):
        X = np.arange(24, dtype=float).reshape(8, 3)
        model = FeatureReweighting(eps=0)
        pipeline = Pipeline([("weights", clone(model)), ("pca", PCA(n_components=2))])
        result = pipeline.fit_transform(X, weights__sample_ids=np.repeat(np.arange(4), 2))
        self.assertEqual(result.shape, (8, 2))
        self.assertFalse(hasattr(model, "weights_"))

    def test_validation(self):
        with self.assertRaises(NotFittedError):
            FeatureReweighting().transform([[1, 2]])
        views = [np.ones((3, 2)), np.ones((3, 2))]
        for kwargs in ({"method": "other"}, {"weighting": "other"}, {"eps": -1}, {"eps": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FeatureReweighting(**kwargs).fit_views(views)
        for invalid in ([views[0]], [views[0], np.ones((2, 2))],
                        [views[0], np.ones((3, 3))],
                        [views[0], np.full((3, 2), np.nan)]):
            with self.assertRaises(ValueError):
                FeatureReweighting().fit_views(invalid)
        with self.assertRaisesRegex(ValueError, "complete samples"):
            FeatureReweighting().fit([[1], [2], [3]], sample_ids=[0, 0, 1])
        with self.assertRaises(ValueError):
            FeatureReweighting().fit_views(views).transform([[1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
