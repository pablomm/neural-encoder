import itertools
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from neural_encoder.linear import CrossViewRidge, DistilledMCCA, FeatureReweighting, LinearEncoder


class TestLinearEncoder(unittest.TestCase):
    def test_transform_until(self):
        with self.assertRaises(NotFittedError):
            LinearEncoder().transform_until(self.test_X, stage="pca")
        model = LinearEncoder(
            pca_kwargs={"n_components": 4, "whiten": True},
            distilled_mcca_kwargs={"n_components": 2},
        ).fit(self.X, sample_ids=self.samples)
        weighted = model.feature_reweighting_.transform(self.test_X)
        assert_allclose(model.transform_until(self.test_X, stage="feature_reweighting"), weighted)
        scores = model.pca_.transform(weighted)
        assert_allclose(model.transform_until(self.test_X, stage="pca"), scores)
        assert_allclose(model.transform_until(self.test_X, stage="cross_view_ridge"),
                        model.cross_view_ridge_.transform(scores))
        assert_allclose(model.transform_until(self.test_X, stage="distilled_mcca"),
                        model.transform(self.test_X), atol=1e-12)
        with self.assertRaisesRegex(ValueError, "stage must be"):
            model.transform_until(self.test_X, stage="unknown")
        with self.assertRaises(ValueError):
            model.transform_until(self.test_X[:, :-1], stage="pca")

    def test_transform_until_skips_disabled_preceding_stages(self):
        model = LinearEncoder(
            feature_reweighting=None, pca_kwargs={"n_components": 4}, cross_view_ridge=None,
            distilled_mcca=None,
        ).fit(self.X)
        assert_allclose(model.transform_until(self.test_X, stage="pca"), model.pca_.transform(self.test_X))
        for stage in ("feature_reweighting", "cross_view_ridge", "distilled_mcca"):
            with self.assertRaisesRegex(ValueError, "disabled"):
                model.transform_until(self.test_X, stage=stage)

    def test_to_torch(self):
        import torch

        with self.assertRaises(NotFittedError):
            LinearEncoder().to_torch()
        model = LinearEncoder(
            pca_kwargs={"n_components": 4, "whiten": True},
            distilled_mcca_kwargs={"n_components": 2},
        ).fit(self.X, sample_ids=self.samples)
        layer = model.to_torch()
        self.assertIsInstance(layer, torch.nn.Linear)
        self.assertEqual(layer.weight.dtype, torch.float64)
        X = torch.tensor(self.test_X, requires_grad=True)
        result = layer(X)
        expected = model.transform(self.test_X)
        assert_allclose(result.detach().numpy(), expected, atol=1e-12)
        result.sum().backward()
        self.assertIsNotNone(X.grad)
        self.assertIsNotNone(layer.weight.grad)
        single = model.to_torch(dtype=torch.float32, device="cpu")
        assert_allclose(single(X.float()).detach().numpy(), expected, atol=1e-6)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
        assert_allclose(model.transform(self.test_X), expected)

    def test_get_projection(self):
        with self.assertRaises(NotFittedError):
            LinearEncoder().get_projection()
        model = LinearEncoder(
            pca_kwargs={"n_components": 4, "whiten": True},
            distilled_mcca_kwargs={"n_components": 2},
        ).fit(self.X, sample_ids=self.samples)
        W, b = model.get_projection()
        self.assertEqual(W.shape, (6, 2))
        self.assertEqual(b.shape, (2,))
        expected = model.transform(self.test_X)
        assert_allclose(self.test_X @ W + b, expected, atol=1e-12)
        W[:] = 0
        b[:] = 0
        assert_allclose(model.transform(self.test_X), expected)

    def setUp(self):
        rng = np.random.default_rng(12)
        signal = rng.normal(size=(30, 6))
        self.X = np.concatenate([signal + rng.normal(scale=0.4, size=signal.shape) + i
                                 for i in range(3)])
        self.samples = np.tile(np.arange(30), 3)
        self.views = np.repeat(np.arange(3), 30)
        self.test_X = rng.normal(size=(7, 6))

    def test_combined_projection_matches_all_stage_combinations(self):
        for weighted, reduced, denoised, distilled in itertools.product((False, True), repeat=4):
            for whiten in (False, True) if reduced else (False,):
                with self.subTest(weighted=weighted, reduced=reduced, denoised=denoised,
                                  distilled=distilled, whiten=whiten):
                    model = LinearEncoder(
                        feature_reweighting=FeatureReweighting(weighting="sqrt") if weighted else None,
                        pca=PCA(n_components=4, whiten=whiten) if reduced else None,
                        cross_view_ridge=CrossViewRidge(alphas=[0.5]) if denoised else None,
                        distilled_mcca=DistilledMCCA(n_components=2) if distilled else None,
                    )
                    original = self.X.copy()
                    model.fit(self.X, sample_ids=self.samples)
                    for X in (self.X, self.test_X):
                        sequential = X
                        stages = (model.feature_reweighting_, model.pca_, model.cross_view_ridge_,
                                  model.distilled_mcca_)
                        for stage in stages:
                            if stage is not None:
                                sequential = stage.transform(sequential)
                        assert_allclose(model.transform(X), sequential, atol=1e-10)
                    assert_array_equal(self.X, original)
                    self.assertEqual(model.coef_.shape, (model.n_components_, 6))

    def test_cross_view_ridge_stage(self):
        params = dict(pca_kwargs={"n_components": 4}, distilled_mcca_kwargs={"n_components": 2}, shuffle_views=False)
        model = LinearEncoder(**params, cross_view_ridge_kwargs={"alphas": [2.0]}).fit(
            self.X, sample_ids=self.samples, view_ids=self.views,
        )
        self.assertIsInstance(model.cross_view_ridge_, CrossViewRidge)
        self.assertEqual(model.cross_view_ridge_.alpha_, 2.0)
        scores = model.pca_.transform(model.feature_reweighting_.transform(self.X))
        expected = CrossViewRidge(alphas=[2.0]).fit(scores, sample_ids=self.samples, view_ids=self.views)
        assert_allclose(model.cross_view_ridge_.coef_, expected.coef_, atol=1e-12)
        # MCCA is fitted on the denoised scores.
        denoised = model.cross_view_ridge_.transform(scores)
        mcca = DistilledMCCA(n_components=2).fit(denoised, sample_ids=self.samples, view_ids=self.views)
        assert_allclose(model.distilled_mcca_.coef_, mcca.coef_, atol=1e-10)
        skipped = LinearEncoder(**params, cross_view_ridge=None).fit(
            self.X, sample_ids=self.samples, view_ids=self.views,
        )
        self.assertIsNone(skipped.cross_view_ridge_)
        mcca = DistilledMCCA(n_components=2).fit(scores, sample_ids=self.samples, view_ids=self.views)
        assert_allclose(skipped.distilled_mcca_.coef_, mcca.coef_, atol=1e-10)
        self.assertFalse(np.allclose(model.transform(self.X), skipped.transform(self.X)))

    def test_default_kwargs_clone_and_nested_parameters(self):
        kwargs = {"n_components": 4}
        model = LinearEncoder(pca_kwargs=kwargs, distilled_mcca_kwargs={"n_components": 2})
        cloned = clone(model)
        result = cloned.fit_transform(self.X, sample_ids=self.samples)
        self.assertEqual(result.shape, (90, 2))
        self.assertEqual(kwargs, {"n_components": 4})
        self.assertEqual(model.pca, "default")
        self.assertFalse(hasattr(model, "coef_"))
        pca = PCA(n_components=4)
        supplied = LinearEncoder(pca=pca, distilled_mcca=DistilledMCCA(n_components=2))
        supplied.set_params(pca__n_components=3)
        supplied.fit(self.X, sample_ids=self.samples)
        self.assertFalse(hasattr(pca, "components_"))
        self.assertEqual(supplied.pca_.n_components_, 3)

    def test_shuffle_assignment_shared_by_stages(self):
        params = dict(pca_kwargs={"n_components": 4}, distilled_mcca_kwargs={"n_components": 2})
        shuffled = LinearEncoder(**params, shuffle_views=True, random_state=42)
        Z = shuffled.fit_transform(self.X, sample_ids=self.samples, view_ids=self.views)
        repeated = clone(shuffled).fit_transform(self.X, sample_ids=self.samples, view_ids=self.views)
        assert_allclose(Z, repeated, atol=1e-12)
        assignments = np.broadcast_to(np.arange(3)[:, None], (3, 30))
        permutations = np.random.default_rng(42).permuted(assignments, axis=0)
        expected_ids = permutations[self.views, self.samples]
        manual = LinearEncoder(**params, shuffle_views=False).fit(
            self.X, sample_ids=self.samples, view_ids=expected_ids,
        )
        assert_allclose(Z, manual.transform(self.X), atol=1e-12)
        assert_allclose(shuffled.feature_reweighting_.weights_, manual.feature_reweighting_.weights_)
        self.assertEqual(shuffled.pca_.random_state, 42)

    def test_missing_views_and_inferred_ids(self):
        params = dict(pca_kwargs={"n_components": 4}, distilled_mcca_kwargs={"n_components": 2})
        inferred = LinearEncoder(**params, random_state=8).fit(
            self.X[:-1], sample_ids=self.samples[:-1],
        )
        explicit = LinearEncoder(**params, random_state=8).fit(
            self.X[:-1], sample_ids=self.samples[:-1], view_ids=self.views[:-1],
        )
        assert_allclose(inferred.transform(self.X), explicit.transform(self.X))
        self.assertEqual(inferred.feature_reweighting_.n_samples_, 29)
        self.assertEqual(inferred.pca_.n_samples_, 89)
        self.assertEqual(inferred.distilled_mcca_.n_samples_, 30)
        self.assertEqual(inferred.cross_view_ridge_.n_samples_, 30)
        self.assertEqual(inferred.cross_view_ridge_.n_pairs_, 89)

    def test_disabled_stages_without_ids_and_refit(self):
        model = LinearEncoder(feature_reweighting=None, pca=None, cross_view_ridge=None, distilled_mcca=None)
        assert_allclose(model.fit_transform(self.X), self.X)
        model.set_params(pca=PCA(n_components=3)).fit(self.X)
        self.assertEqual(model.transform(self.X).shape, (90, 3))
        model.set_params(pca=None).fit(self.X)
        self.assertIsNone(model.pca_)
        assert_allclose(model.transform(self.X), self.X)

    def test_validation(self):
        with self.assertRaises(NotFittedError):
            LinearEncoder().transform(self.X)
        with self.assertRaisesRegex(ValueError, "sample_ids"):
            LinearEncoder().fit(self.X)
        with self.assertRaisesRegex(ValueError, "sample_ids"):
            LinearEncoder(feature_reweighting=None, pca=None, distilled_mcca=None).fit(self.X)
        for kwargs in (
            {"pca": None, "pca_kwargs": {}},
            {"pca": PCA(), "pca_kwargs": {"n_components": 2}},
            {"pca": "other"}, {"pca_kwargs": []},
            {"cross_view_ridge": None, "cross_view_ridge_kwargs": {}},
            {"cross_view_ridge": CrossViewRidge(), "cross_view_ridge_kwargs": {"fit_intercept": False}},
            {"cross_view_ridge": "other"}, {"cross_view_ridge_kwargs": {"alphas": [-1.0]}},
            {"random_state": -1}, {"shuffle_views": "yes"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                LinearEncoder(**kwargs).fit(self.X, sample_ids=self.samples)
        with self.assertRaises(ValueError):
            LinearEncoder().fit(self.X, sample_ids=self.samples, view_ids=np.zeros(len(self.X)))


if __name__ == "__main__":
    unittest.main()
