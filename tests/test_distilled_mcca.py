import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from neural_encoder.linear import DistilledMCCA


class TestDistilledMCCA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        signal = rng.normal(size=(20, 4))
        self.views = [signal + rng.normal(scale=0.2, size=signal.shape) + i for i in range(3)]

    def assert_manual_distillation(self, model, X):
        target = np.mean([
            model.mcca_.transform_view(X, i) for i in range(model.n_views_)
        ], axis=0)
        Xm = X.mean(axis=0) if model.fit_intercept else np.zeros(X.shape[1])
        Zm = target.mean(axis=0) if model.fit_intercept else np.zeros(target.shape[1])
        Xc, Zc = X - Xm, target - Zm
        W = np.linalg.solve(Xc.T @ Xc + model.distill_reg * np.eye(X.shape[1]), Xc.T @ Zc)
        assert_allclose(model.coef_, W.T, atol=1e-10)
        assert_allclose(model.intercept_, Zm - Xm @ W, atol=1e-10)
        assert_allclose(model.transform(X), X @ W + Zm - Xm @ W, atol=1e-10)

    def test_distillation_with_centering_and_no_intercept(self):
        X = np.concatenate(self.views)
        for intercept in (True, False):
            with self.subTest(intercept=intercept):
                model = DistilledMCCA(n_components=2, fit_intercept=intercept).fit_views(self.views)
                self.assert_manual_distillation(model, X)
                self.assertEqual(model.coef_.shape, (2, 4))
        assert_array_equal(X, np.concatenate(self.views))

    def test_missing_views_distill_only_original_measurements(self):
        X = np.concatenate(self.views)
        samples = np.tile(np.arange(20), 3)
        view_ids = np.repeat(np.arange(3), 20)
        X, samples, view_ids = X[:-1], samples[:-1], view_ids[:-1]
        for impute in ("mean", "zeros", "discard"):
            with self.subTest(impute=impute):
                model = DistilledMCCA(n_components=2, impute=impute)
                Z = model.fit_transform(X, sample_ids=samples, view_ids=view_ids)
                self.assertEqual(Z.shape, (59, 2))
                self.assertEqual(model.n_samples_, 19 if impute == "discard" else 20)
                self.assert_manual_distillation(model, X)

    def test_zero_penalty_reproduces_average_projector(self):
        model = DistilledMCCA(n_components=1, distill_reg=0).fit_views(self.views)
        X = np.random.default_rng(8).normal(size=(7, 4))
        expected = np.mean([model.mcca_.transform_view(X, i) for i in range(3)], axis=0)
        assert_allclose(model.transform(X), expected, atol=1e-10)
        self.assertEqual(model.coef_.shape, (1, 4))

    def test_matrix_inference_pipeline_and_clone(self):
        X = np.concatenate(self.views)
        samples = np.tile(np.arange(20), 3)
        template = DistilledMCCA(n_components=2)
        pipeline = Pipeline([("mcca", clone(template))])
        Z = pipeline.fit_transform(X, mcca__sample_ids=samples)
        self.assertEqual(Z.shape, (60, 2))
        self.assertFalse(hasattr(template, "mcca_"))
        direct = clone(template).fit_views(self.views)
        assert_allclose(Z, direct.transform(X), atol=1e-10)

    def test_validation(self):
        with self.assertRaises(NotFittedError):
            DistilledMCCA().transform(self.views[0])
        for params in ({"n_components": 0}, {"n_components": 5}, {"mcca_reg": 2},
                       {"distill_reg": -1}, {"impute": None}, {"fit_intercept": "yes"}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                DistilledMCCA(**params).fit_views(self.views)
        for views in ([self.views[0]], [self.views[0], self.views[1][:-1]],
                      [self.views[0], np.full((20, 4), np.nan)]):
            with self.assertRaises(ValueError):
                DistilledMCCA(n_components=2).fit_views(views)
        with self.assertRaisesRegex(ValueError, "aligned samples"):
            DistilledMCCA(n_components=1, impute="discard").fit(
                [[1], [2], [3]], sample_ids=[0, 0, 1],
            )
        with self.assertRaises(ValueError):
            DistilledMCCA(n_components=2).fit_views(self.views).transform([[1, 2]])


if __name__ == "__main__":
    unittest.main()
