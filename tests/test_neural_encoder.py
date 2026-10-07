import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from neural_encoder import NeuralEncoder
from neural_encoder.linear import CrossViewRidge, LinearEncoder


class TestNeuralEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        rng = np.random.default_rng(7)
        signal = rng.normal(size=(16, 8))
        self.X = np.concatenate([signal + rng.normal(scale=.2, size=signal.shape) for _ in range(3)])
        self.ids = np.tile(np.arange(16), 3)
        self.test_X = rng.normal(size=(5, 8))

    def make_encoder(self, **kwargs):
        return NeuralEncoder(**(dict(
            n_components_pca=4, n_components_mcca=2, random_state=7,
            refiner_kwargs={"steps": 3, "batch_size": 8, "device": "cpu",
                            "network_kwargs": {"hidden_dim": 8}, "dtype": torch.float64},
        ) | kwargs))

    def test_shared_pca_and_torch_equivalence(self):
        for dimension in (None, 4):
            model = self.make_encoder(
                refinement_input_stage="pca",
                n_components_pca_refinement=dimension,
            )
            embeddings = model.fit_transform(self.X, sample_ids=self.ids)
            self.assertEqual(embeddings.shape, (48, 2))
            self.assertIs(model.refinement_pca_, model.linear_encoder_.pca_)
            module = model.to_pytorch()
            self.assertIsNone(module.refinement_pca)
            self.assertIs(module.refiner, model.refiner_.model_)
            calls = []
            handle = module.pca.register_forward_hook(lambda *args: calls.append(1))
            x = torch.tensor(self.test_X, requires_grad=True)
            result = module(x)
            self.assertEqual(len(calls), 1)
            handle.remove()
            assert_allclose(result.detach(), model.transform(self.test_X), atol=1e-10)
            result.square().sum().backward()
            self.assertIsNotNone(x.grad)
            self.assertIsNotNone(module.pca.weight.grad)
            self.assertIsNotNone(module.refiner.alpha.grad)

    def test_separate_pca_whitening_and_clone(self):
        template = self.make_encoder(
            refinement_input_stage="pca", n_components_pca_refinement=6,
            pca_kwargs={"whiten": True},
        )
        model = clone(template).fit(self.X, sample_ids=self.ids)
        self.assertFalse(hasattr(template, "linear_encoder_"))
        self.assertIsNot(model.refinement_pca_, model.linear_encoder_.pca_)
        self.assertEqual(model.refinement_pca_.n_components_, 6)
        self.assertEqual(model.refiner_.network_input_features_, 6)
        module = model.to_pytorch()
        self.assertIsNot(module.pca, module.refinement_pca)
        output = module(torch.tensor(self.test_X))
        assert_allclose(output.detach(), model.transform(self.test_X), atol=1e-10)
        output.sum().backward()
        self.assertIsNotNone(module.refinement_pca.weight.grad)
        expected = model.transform(self.test_X)
        assert_allclose(clone(template).fit(self.X, sample_ids=self.ids).transform(self.test_X), expected)

    def test_custom_network_reused_and_float32_export(self):
        branch = torch.nn.Linear(6, 2)
        model = self.make_encoder(
            refinement_input_stage="pca", n_components_pca_refinement=6,
            refiner_kwargs={
                "steps": 2, "device": "cpu", "network": branch,
            },
        ).fit(self.X, sample_ids=self.ids)
        self.assertIs(model.refiner_.model_.network, branch)
        module = model.to_torch()
        assert_allclose(module(torch.tensor(self.test_X, dtype=torch.float32)).detach(),
                        model.transform(self.test_X), atol=1e-6)

    def test_default_distilled_mcca_input_and_torch_equivalence(self):
        model = self.make_encoder().fit(self.X, sample_ids=self.ids)
        self.assertEqual(model.refinement_input_stage, "distilled_mcca")
        self.assertIsNone(model.refinement_pca_)
        self.assertEqual(model.refiner_.network_input_features_, 2)
        self.assertFalse(model.refiner_.separate_network_input_)
        module = model.to_pytorch()
        self.assertEqual(module.refinement_input_stage, "distilled_mcca")
        self.assertIsNone(module.refinement_pca)
        result = module(torch.tensor(self.test_X))
        assert_allclose(result.detach(), model.transform(self.test_X), atol=1e-10)

    def test_cross_view_ridge_flag(self):
        default = self.make_encoder().fit(self.X, sample_ids=self.ids)
        self.assertTrue(default.cross_view_ridge)
        self.assertIsInstance(default.linear_encoder_.cross_view_ridge_, CrossViewRidge)
        tuned = self.make_encoder(cross_view_ridge_kwargs={"alphas": [3.0]}).fit(self.X, sample_ids=self.ids)
        self.assertEqual(tuned.linear_encoder_.cross_view_ridge_.alpha_, 3.0)
        for flag in (True, False):
            for stage in ("distilled_mcca", "pca"):
                with self.subTest(cross_view_ridge=flag, stage=stage):
                    model = self.make_encoder(cross_view_ridge=flag, refinement_input_stage=stage)
                    model.fit(self.X, sample_ids=self.ids)
                    linear = model.linear_encoder_
                    self.assertEqual(linear.cross_view_ridge_ is not None, flag)
                    # The embedding fed to the refiner is the combined linear encoding.
                    embedding, source = model._representations(self.test_X)
                    assert_allclose(embedding, linear.transform(self.test_X), atol=1e-10)
                    if stage == "pca":
                        assert_allclose(source, linear.transform_until(self.test_X, stage="pca"), atol=1e-10)
                    result = model.to_pytorch()(torch.tensor(self.test_X))
                    assert_allclose(result.detach(), model.transform(self.test_X), atol=1e-10)
        # Without the denoiser, the linear stage is reweighting, PCA, MCCA, and output reweighting.
        skipped = self.make_encoder(cross_view_ridge=False).fit(self.X, sample_ids=self.ids)
        original = LinearEncoder(
            pca_kwargs={"n_components": 4}, cross_view_ridge=None,
            distilled_mcca_kwargs={"n_components": 2}, random_state=7,
        ).fit(self.X, sample_ids=self.ids)
        assert_allclose(skipped.linear_encoder_.transform(self.test_X), original.transform(self.test_X), atol=1e-10)
        self.assertFalse(np.allclose(skipped.transform(self.test_X), default.transform(self.test_X)))
        with self.assertRaisesRegex(ValueError, "cross_view_ridge must be a boolean"):
            self.make_encoder(cross_view_ridge="yes").fit(self.X, sample_ids=self.ids)
        with self.assertRaisesRegex(ValueError, "requires cross_view_ridge=True"):
            self.make_encoder(cross_view_ridge=False, cross_view_ridge_kwargs={}).fit(self.X, sample_ids=self.ids)

    def test_output_reweighting_flag_and_refiner_output_scale(self):
        default = self.make_encoder().fit(self.X, sample_ids=self.ids)
        self.assertTrue(default.output_reweighting)
        weights = default.linear_encoder_.output_reweighting_.weights_
        for stage in ("distilled_mcca", "pca"):
            with self.subTest(stage=stage):
                plain = self.make_encoder(output_reweighting=False, refinement_input_stage=stage)
                plain.fit(self.X, sample_ids=self.ids)
                self.assertIsNone(plain.linear_encoder_.output_reweighting_)
                weighted = self.make_encoder(refinement_input_stage=stage).fit(self.X, sample_ids=self.ids)
                embedding, _ = weighted._representations(self.test_X)
                assert_allclose(embedding, plain._representations(self.test_X)[0] * weights, atol=1e-10)
                assert_allclose(weighted.to_pytorch()(torch.tensor(self.test_X)).detach(),
                                weighted.transform(self.test_X), atol=1e-10)
        # The refiner's own output reweighting is exported with the residual module.
        refiner_kwargs = dict(self.make_encoder().refiner_kwargs, output_reweighting="default")
        scaled = self.make_encoder(refiner_kwargs=refiner_kwargs).fit(self.X, sample_ids=self.ids)
        self.assertIsNotNone(scaled.refiner_.model_.output_scale)
        assert_allclose(scaled.to_pytorch()(torch.tensor(self.test_X)).detach(),
                        scaled.transform(self.test_X), atol=1e-10)
        cv = self.make_encoder(output_reweighting_cv=2, output_reweighting_matching="soft").fit(
            self.X, sample_ids=self.ids,
        )
        self.assertEqual(cv.linear_encoder_.output_reweighting_cv, 2)
        self.assertEqual(cv.linear_encoder_.output_reweighting_matching, "soft")
        assert_allclose(cv.to_pytorch()(torch.tensor(self.test_X)).detach(), cv.transform(self.test_X), atol=1e-10)
        with self.assertRaisesRegex(ValueError, "requires output reweighting"):
            self.make_encoder(output_reweighting=False, output_reweighting_cv=2).fit(self.X, sample_ids=self.ids)
        snr = self.make_encoder(output_reweighting_kwargs={"weighting": "snr"}).fit(self.X, sample_ids=self.ids)
        self.assertEqual(snr.linear_encoder_.output_reweighting_.weighting, "snr")
        with self.assertRaisesRegex(ValueError, "output_reweighting must be a boolean"):
            self.make_encoder(output_reweighting="yes").fit(self.X, sample_ids=self.ids)
        with self.assertRaisesRegex(ValueError, "requires output_reweighting=True"):
            self.make_encoder(output_reweighting=False, output_reweighting_kwargs={}).fit(self.X, sample_ids=self.ids)

    def test_shuffle_is_shared_by_linear_and_nonlinear_stages(self):
        shuffled = self.make_encoder(random_state=11)
        result = shuffled.fit_transform(self.X, sample_ids=self.ids)
        views = np.repeat(np.arange(3), 16)
        assignments = np.broadcast_to(np.arange(3)[:, None], (3, 16))
        permutations = np.random.default_rng(11).permuted(assignments, axis=0)
        expected_views = permutations[views, self.ids]
        manual = self.make_encoder(shuffle_views=False, random_state=11)
        expected = manual.fit_transform(
            self.X, sample_ids=self.ids, view_ids=expected_views,
        )
        assert_allclose(result, expected, atol=1e-10)

    def test_fit_views_matches_matrix_interface(self):
        views = np.split(self.X, 3)
        direct = self.make_encoder().fit(self.X, sample_ids=self.ids)
        aligned = self.make_encoder().fit_views(views)
        assert_allclose(
            aligned.transform(self.test_X), direct.transform(self.test_X), atol=1e-10,
        )
        with self.assertRaisesRegex(ValueError, "At least two views"):
            self.make_encoder().fit_views(views[:1])
        with self.assertRaisesRegex(ValueError, "equal shapes"):
            self.make_encoder().fit_views([views[0], views[1][:-1]])

    def test_validation_and_disabled_dimension_overrides(self):
        model = self.make_encoder()
        with self.assertRaises(NotFittedError):
            model.to_pytorch()
        with self.assertRaises(NotFittedError):
            model.transform(self.X)
        with self.assertRaisesRegex(ValueError, "dimension parameters"):
            self.make_encoder(pca_kwargs={"n_components": 3}).fit(self.X, sample_ids=self.ids)
        with self.assertRaisesRegex(ValueError, "refinement_input_stage"):
            self.make_encoder(refinement_input_stage="unknown").fit(
                self.X, sample_ids=self.ids,
            )
        with self.assertRaisesRegex(ValueError, "only valid"):
            self.make_encoder(n_components_pca_refinement=6).fit(
                self.X, sample_ids=self.ids,
            )
        model.fit(self.X[:-1], sample_ids=self.ids[:-1])
        self.assertEqual(model.refiner_.n_samples_, 15)
        with self.assertRaises(ValueError):
            model.transform(self.X[:, :-1])


if __name__ == "__main__":
    unittest.main()
