import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline

from neural_encoder.linear import CrossViewRidge


class TestCrossViewRidge(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.signal = rng.normal(size=(60, 5))
        self.views = [self.signal + rng.normal(scale=0.7, size=self.signal.shape) for _ in range(3)]

    def manual_pairs(self, views):
        inputs, targets = [], []
        for i, view in enumerate(views):
            others = [other for j, other in enumerate(views) if j != i]
            inputs.append(view)
            targets.append(np.mean(others, axis=0))
        return np.vstack(inputs), np.vstack(targets)

    def test_matches_ridge_to_leave_one_view_out_mean(self):
        alphas = [0.1, 1.0, 10.0]
        model = CrossViewRidge(alphas=alphas).fit_views(self.views)
        inputs, targets = self.manual_pairs(self.views)
        reference = RidgeCV(alphas=alphas).fit(inputs, targets)
        assert_allclose(model.coef_, reference.coef_, atol=1e-12)
        assert_allclose(model.intercept_, reference.intercept_, atol=1e-12)
        self.assertEqual(model.alpha_, reference.alpha_)
        X = np.random.default_rng(4).normal(size=(7, 5))
        assert_allclose(model.transform(X), reference.predict(X), atol=1e-12)
        self.assertEqual((model.n_views_, model.n_samples_, model.n_pairs_), (3, 60, 180))
        self.assertEqual(model.coef_.shape, (5, 5))

    def test_matrix_fit_matches_views_and_ignores_row_order(self):
        X = np.concatenate(self.views)
        samples = np.tile(np.arange(60), 3)
        view_ids = np.repeat(np.arange(3), 60)
        order = np.random.default_rng(5).permutation(len(X))
        model = CrossViewRidge().fit(X[order], sample_ids=samples[order], view_ids=view_ids[order])
        direct = CrossViewRidge().fit_views(self.views)
        assert_allclose(model.coef_, direct.coef_, atol=1e-10)
        assert_allclose(model.intercept_, direct.intercept_, atol=1e-10)

    def test_missing_views_use_available_repetitions(self):
        X = np.concatenate(self.views)
        samples = np.tile(np.arange(60), 3)
        view_ids = np.repeat(np.arange(3), 60)
        # Sample 0 keeps one view (no target); sample 1 keeps two views.
        keep = ~(((samples == 0) & (view_ids > 0)) | ((samples == 1) & (view_ids == 2)))
        model = CrossViewRidge(alphas=[1.0]).fit(X[keep], sample_ids=samples[keep], view_ids=view_ids[keep])
        self.assertEqual((model.n_samples_, model.n_pairs_), (59, 3 * 58 + 2))
        inputs, targets = self.manual_pairs([view[2:] for view in self.views])
        pair = [self.views[0][1], self.views[1][1]]
        inputs = np.vstack([inputs, pair])
        targets = np.vstack([targets, pair[::-1]])
        reference = RidgeCV(alphas=[1.0]).fit(inputs, targets)
        assert_allclose(model.coef_, reference.coef_, atol=1e-10)

    def test_denoises_towards_shared_signal(self):
        model = CrossViewRidge().fit_views(self.views)
        raw_error = np.mean((self.views[0] - self.signal) ** 2)
        denoised_error = np.mean((model.transform(self.views[0]) - self.signal) ** 2)
        self.assertLess(denoised_error, raw_error)

    def test_no_intercept_and_float32_transform(self):
        model = CrossViewRidge(fit_intercept=False).fit_views(self.views)
        assert_array_equal(model.intercept_, np.zeros(5))
        result = model.transform(self.views[0].astype(np.float32))
        self.assertEqual(result.shape, (60, 5))

    def test_pipeline_and_clone(self):
        X = np.concatenate(self.views)
        samples = np.tile(np.arange(60), 3)
        template = CrossViewRidge()
        pipeline = Pipeline([("denoise", clone(template))])
        result = pipeline.fit_transform(X, denoise__sample_ids=samples)
        self.assertEqual(result.shape, X.shape)
        self.assertFalse(hasattr(template, "coef_"))

    def test_validation(self):
        with self.assertRaises(NotFittedError):
            CrossViewRidge().transform(self.views[0])
        for params in ({"alphas": []}, {"alphas": [0.0, 1.0]}, {"alphas": [np.inf]}, {"fit_intercept": "yes"}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                CrossViewRidge(**params).fit_views(self.views)
        for views in ([self.views[0]], [self.views[0], self.views[1][:-1]],
                      [self.views[0], np.full((60, 5), np.nan)]):
            with self.subTest(n_views=len(views)), self.assertRaises(ValueError):
                CrossViewRidge().fit_views(views)
        with self.assertRaises(ValueError):
            # Only one sample has two views.
            CrossViewRidge().fit(np.ones((3, 2)), sample_ids=[0, 0, 1])
        model = CrossViewRidge().fit_views(self.views)
        with self.assertRaises(ValueError):
            model.transform(np.ones((2, 4)))


if __name__ == "__main__":
    unittest.main()
