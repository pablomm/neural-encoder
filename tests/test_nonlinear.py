import unittest
from io import StringIO
from contextlib import redirect_stdout

import numpy as np
import torch
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from torch import nn

from neural_encoder.nonlinear import (
    ResidualMLP, ResidualNetwork, MultiViewContrastiveLoss,
    NonlinearRefiner, symmetric_info_nce, cosine_pull,
)
from neural_encoder.utils import ConsoleLogger


def squared_residual(outputs, *, inputs=None):
    return torch.stack([(out - source).square().mean() for out, source in zip(outputs, inputs)]).mean()


class Recorder:
    def __init__(self):
        self.events = []

    def log(self, metrics, *, step):
        self.events.append((step, dict(metrics)))


class TestNonlinear(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        rng = np.random.default_rng(3)
        signal = rng.normal(size=(12, 4)).astype(np.float32)
        self.views = [signal + rng.normal(scale=0.3, size=signal.shape).astype(np.float32) for _ in range(3)]

    def make_refiner(self, **kwargs):
        config = dict(network_kwargs={"hidden_dim": 12, "dropout": 0.1},
                      steps=5, batch_size=8, device="cpu", random_state=7)
        return NonlinearRefiner(**(config | kwargs))

    def test_identity_initialization_and_direct_pytorch_gradients(self):
        model = ResidualMLP(4, hidden_dim=12)
        views = [torch.tensor(X) for X in self.views]
        assert_allclose(model(views[0]).detach(), views[0])
        criterion = MultiViewContrastiveLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = model.alpha.detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            loss = criterion([model(X) for X in views], inputs=views)
            loss.backward()
            self.assertIsNotNone(model.alpha.grad)
            optimizer.step()
        self.assertFalse(torch.equal(model.alpha, before))
        self.assertGreater(model.network[-1].weight.abs().sum().item(), 0)
        fixed = ResidualMLP(4, hidden_dim=12, trainable_alpha=False)
        self.assertIn("alpha", dict(fixed.named_buffers()))
        self.assertNotIn("alpha", dict(fixed.named_parameters()))

    def test_multiview_loss_matches_manual_and_has_gradients(self):
        views = [torch.tensor(X, requires_grad=True) for X in self.views]
        criterion = MultiViewContrastiveLoss(temperature=0.2, pull_weight=0.7)
        expected = sum(symmetric_info_nce(views[i], views[j], 0.2) + 0.7 * cosine_pull(views[i], views[j])
                       for i, j in ((0, 1), (0, 2), (1, 2))) / 3
        actual = criterion(views)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in views))

    def test_estimator_reproducibility_refit_clone_and_torch_interface(self):
        template = self.make_refiner()
        first = clone(template).fit_views(self.views)
        second = clone(template).fit_views(self.views)
        expected = first.transform(self.views[0])
        assert_array_equal(expected, second.transform(self.views[0]))
        first.fit_views(self.views)
        assert_array_equal(expected, first.transform(self.views[0]))
        self.assertFalse(hasattr(template, "model_"))
        self.assertFalse(first.model_.training)
        X = torch.tensor(self.views[0], requires_grad=True)
        assert_allclose(first.transform(X), first.model_(X).detach().numpy(), atol=1e-6)
        first.model_(X).sum().backward()
        self.assertIsNotNone(X.grad)
        self.assertEqual(first.n_steps_, 5)
        self.assertEqual(len(first.history_), 5)

    def test_matrix_fit_missing_views_and_frozen_input(self):
        X = np.concatenate(self.views)[:-1]
        samples = np.tile(np.arange(12), 3)[:-1]
        original = X.copy()
        model = self.make_refiner()
        result = model.fit_transform(X, sample_ids=samples)
        self.assertEqual(result.shape, X.shape)
        self.assertEqual(model.n_samples_, 11)
        assert_array_equal(X, original)
        model.set_params(impute="mean").fit(X, sample_ids=samples)
        self.assertEqual(model.n_samples_, 12)
        tensor = torch.tensor(X, requires_grad=True)
        model.fit(tensor, sample_ids=samples)
        self.assertIsNone(tensor.grad)

    def test_custom_network_loss_no_duplicate_samples_and_fixed_alpha(self):
        branch = nn.Linear(4, 4)
        before = branch.weight.detach().clone()
        calls = []
        def objective(outputs, *, inputs=None):
            self.assertEqual(len(torch.unique(inputs[0], dim=0)), len(inputs[0]))
            calls.append(len(inputs[0]))
            return squared_residual(outputs, inputs=inputs)
        model = self.make_refiner(network=branch, network_kwargs=None, loss=objective,
                                  trainable_alpha=False, batch_size=100, steps=12,
                                  optimizer_kwargs={"lr": 0.03, "weight_decay": 0.0})
        model.fit_views(self.views)
        self.assertEqual(calls, [12] * 12)
        self.assertIs(model.model_.network, branch)
        self.assertFalse(torch.equal(branch.weight, before))
        self.assertAlmostEqual(model.model_.alpha.item(), 0.25)
        self.assertLess(model.history_[-1]["loss"], model.history_[0]["loss"])
        factory = lambda n_features, bias: nn.Linear(n_features, n_features, bias=bias)
        self.make_refiner(network=factory, network_kwargs={"bias": False}).fit_views(self.views[:2])

    def test_supplied_loss_is_used_directly(self):
        loss = MultiViewContrastiveLoss()
        model = self.make_refiner(loss=loss).fit_views(self.views)
        self.assertIs(model.loss_, loss)

    def test_validation_logger_and_scheduler(self):
        recorder = Recorder()
        model = self.make_refiner(
            logger=recorder, eval_every=2, scheduler=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={"step_size": 1, "gamma": 0.5},
        ).fit_views(self.views, validation_views=[v[:5] for v in self.views])
        self.assertEqual([step for step, metrics in recorder.events if "validation_loss" in metrics], [2, 4, 5])
        self.assertIn("validation_rsa", model.history_[-1])
        self.assertAlmostEqual(model.history_[1]["lr"], model.history_[0]["lr"] * 0.5)
        stream = StringIO()
        with redirect_stdout(stream):
            ConsoleLogger().log({"loss": 1.2}, step=3)
        self.assertIn("step=3 loss=1.2", stream.getvalue())

    def test_rng_preservation_and_float64(self):
        state = torch.get_rng_state().clone()
        model = self.make_refiner(dtype=torch.float64).fit_views(self.views)
        torch.testing.assert_close(torch.get_rng_state(), state)
        self.assertEqual(model.transform(self.views[0]).dtype, np.float64)

    def test_to_torch_returns_internal_module(self):
        with self.assertRaises(NotFittedError):
            self.make_refiner().to_torch()
        refiner = self.make_refiner().fit_views(self.views)
        model = refiner.to_torch()
        self.assertIs(model, refiner.model_)
        with torch.no_grad():
            model.alpha.zero_()
        assert_allclose(refiner.transform(self.views[0]), self.views[0])
        model.train()
        self.assertTrue(refiner.to_torch().training)

    def test_validation_errors(self):
        with self.assertRaises(NotFittedError):
            self.make_refiner().transform(self.views[0])
        for kwargs in ({"steps": 0}, {"batch_size": 1}, {"impute": "zeros"},
                       {"loss": "other"}, {"random_state": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                self.make_refiner(**kwargs).fit_views(self.views)
        with self.assertRaises(ValueError):
            self.make_refiner().fit_views([self.views[0]])
        with self.assertRaises(ValueError):
            self.make_refiner(network=nn.Linear(4, 1), network_kwargs=None).fit_views(self.views)
        with self.assertRaises(ValueError):
            self.make_refiner(loss=lambda outputs, **kwargs: 0.0).fit_views(self.views)
        with self.assertRaises(TypeError):
            self.make_refiner(network=ResidualMLP(4), network_kwargs=None).fit_views(self.views)


if __name__ == "__main__":
    unittest.main()
