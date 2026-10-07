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

    def test_pairwise_reliability_and_snr_weights(self):
        rng = np.random.default_rng(3)
        signal = rng.normal(size=(5000, 4))
        noise_scale = np.array([0.2, 0.5, 1.0, 3.0])
        views = [signal + noise_scale * rng.normal(size=signal.shape) for _ in range(3)]
        model = FeatureReweighting(method="pairwise", weighting="snr", eps=0).fit_views(views)
        pairs = [(0, 1), (0, 2), (1, 2)]
        expected = np.mean([[np.corrcoef(views[i][:, f], views[j][:, f])[0, 1] for f in range(4)]
                            for i, j in pairs], axis=0)
        assert_allclose(model.reliability_, expected, atol=1e-12)
        assert_allclose(model.weights_, expected / (1 - expected), atol=1e-12)
        # True single-measurement SNR is 1 / noise_scale**2.
        assert_allclose(model.weights_, 1 / noise_scale ** 2, rtol=0.05, atol=0.03)
        sqrt = FeatureReweighting(method="pairwise", weighting="sqrt_snr", eps=0).fit_views(views)
        assert_allclose(sqrt.weights_, np.sqrt(model.weights_))
        # Leave-one-out reliability is higher than single-view reliability.
        loo = FeatureReweighting(weighting="snr").fit_views(views)
        self.assertTrue(np.all(loo.reliability_ > model.reliability_))

    def test_snr_handles_negative_and_perfect_reliability(self):
        signal = np.arange(10.0)[:, None]
        noise = np.random.default_rng(0).normal(size=(10, 1))
        views = [np.hstack([signal, noise[::-1] * (i + 1) + i * noise]) for i in range(2)]
        views[1][:, 1] = -views[0][:, 1]
        model = FeatureReweighting(method="pairwise", weighting="snr", eps=0.01).fit_views(views)
        self.assertTrue(np.all(np.isfinite(model.weights_)))
        self.assertGreater(model.weights_[0], 1e5)
        self.assertAlmostEqual(model.weights_[1], 0.01)

    def test_normalize(self):
        rng = np.random.default_rng(4)
        views = [rng.normal(size=(50, 5)) + np.arange(5) for _ in range(2)]
        views[1] += views[0]
        for weighting in ("linear", "sqrt", "snr", "sqrt_snr"):
            with self.subTest(weighting=weighting):
                plain = FeatureReweighting(weighting=weighting).fit_views(views)
                normalized = FeatureReweighting(weighting=weighting, normalize=True).fit_views(views)
                self.assertAlmostEqual(float(np.sqrt(np.mean(normalized.weights_ ** 2))), 1.0)
                assert_allclose(normalized.weights_ / plain.weights_,
                                np.full(5, normalized.weights_[0] / plain.weights_[0]))

    def test_reliability_transfer_between_fitted_models(self):
        from neural_encoder.linear.feature_reweighting import _correlation_matrix, _transfer_reliability

        rng = np.random.default_rng(5)
        full = rng.normal(size=(5000, 4))
        order = np.array([2, 0, 3, 1])
        fold = full[:, order] * np.array([1, -1, 1, -1]) + 0.05 * rng.normal(size=(5000, 4))
        reliability = np.array([0.9, 0.7, 0.5, 0.3])
        assert_allclose(_transfer_reliability(reliability, fold, full, "index"), reliability)
        expected = np.zeros(4)
        expected[order] = reliability
        assert_allclose(_transfer_reliability(reliability, fold, full, "hungarian"), expected)
        assert_allclose(_transfer_reliability(reliability, fold, full, "soft"), expected, atol=0.01)
        # A fitted column that mixes two fold columns equally gets their mean reliability.
        mixed = np.column_stack([fold[:, 0] + fold[:, 1], fold[:, 2], fold[:, 3]])
        soft = _transfer_reliability(reliability, fold, mixed, "soft")
        self.assertAlmostEqual(soft[0], reliability[:2].mean(), places=2)
        correlations = _correlation_matrix(fold, full)
        assert_allclose(correlations, np.corrcoef(fold, full, rowvar=False)[:4, 4:], atol=1e-12)
        self.assertTrue(np.all(_correlation_matrix(np.ones((5, 1)), full[:5]) == 0))

    def test_cross_validated_reliability_of_a_fixed_model(self):
        from neural_encoder.linear.feature_reweighting import _cross_validated_reliability, _validate_cv

        rng = np.random.default_rng(6)
        signal = rng.normal(size=(40, 3))
        Z = np.concatenate([signal + rng.normal(scale=0.5, size=signal.shape) for _ in range(3)])
        samples, views = np.tile(np.arange(40), 3), np.repeat(np.arange(3), 40)
        # Without refitting, CV averages the held-out reliabilities of the same outputs.
        result = _cross_validated_reliability(
            lambda train: (lambda rows: Z[rows]), Z, samples, views,
            method="pairwise", cv=4, matching="index", random_state=0,
        )
        folds = np.array_split(np.random.default_rng(0).permutation(np.arange(40)), 4)
        expected = np.mean([
            FeatureReweighting(method="pairwise").fit(
                Z[np.isin(samples, fold)], sample_ids=samples[np.isin(samples, fold)],
                view_ids=views[np.isin(samples, fold)],
            ).reliability_ for fold in folds
        ], axis=0)
        assert_allclose(result, expected, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "exceed"):
            _cross_validated_reliability(lambda train: (lambda rows: Z[rows]), Z, samples, views,
                                         method="pairwise", cv=41, matching="index", random_state=0)
        for cv, matching in ((1, "hungarian"), (2.5, "hungarian"), (True, "hungarian"), (3, "other")):
            with self.subTest(cv=cv, matching=matching), self.assertRaises(ValueError):
                _validate_cv(cv, matching)
        _validate_cv(None, "soft")

    def test_validation(self):
        with self.assertRaises(NotFittedError):
            FeatureReweighting().transform([[1, 2]])
        views = [np.ones((3, 2)), np.ones((3, 2))]
        for kwargs in ({"method": "other"}, {"weighting": "other"}, {"eps": -1}, {"eps": np.nan},
                       {"normalize": "yes"}):
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
