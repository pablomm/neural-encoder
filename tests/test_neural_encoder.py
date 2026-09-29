import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from neural_encoder import NeuralEncoder


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
            model = self.make_encoder(n_components_pca_refinement=dimension)
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
        template = self.make_encoder(n_components_pca_refinement=6, pca_kwargs={"whiten": True})
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
        model = self.make_encoder(n_components_pca_refinement=6, refiner_kwargs={
            "steps": 2, "device": "cpu", "network": branch,
        }).fit(self.X, sample_ids=self.ids)
        self.assertIs(model.refiner_.model_.network, branch)
        module = model.to_torch()
        assert_allclose(module(torch.tensor(self.test_X, dtype=torch.float32)).detach(),
                        model.transform(self.test_X), atol=1e-6)

    def test_validation_and_disabled_dimension_overrides(self):
        model = self.make_encoder()
        with self.assertRaises(NotFittedError):
            model.to_pytorch()
        with self.assertRaises(NotFittedError):
            model.transform(self.X)
        with self.assertRaisesRegex(ValueError, "dimension parameters"):
            self.make_encoder(pca_kwargs={"n_components": 3}).fit(self.X, sample_ids=self.ids)
        model.fit(self.X[:-1], sample_ids=self.ids[:-1])
        self.assertEqual(model.refiner_.n_samples_, 15)
        with self.assertRaises(ValueError):
            model.transform(self.X[:, :-1])


if __name__ == "__main__":
    unittest.main()
