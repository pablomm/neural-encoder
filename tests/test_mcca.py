import unittest

import numpy as np
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from neural_encoder._mcca import MCCA
from neural_encoder._mcca.mcca import _construct_mcca_gevp


class TestMCCA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        latent = rng.normal(size=(40, 2))
        self.views = [latent @ rng.normal(size=(2, p)) + rng.normal(size=(40, p))
                      for p in (3, 4, 5)]

    def test_generalized_eigenproblem_and_transforms(self):
        for regs in (None, 0, 0.2, 1, "lw", "oas", [0.1, "lw", "oas"]):
            with self.subTest(regs=regs):
                model = MCCA(n_components=2, regs=regs)
                scores = model.fit_transform(self.views)
                assert_allclose(scores, model.transform(self.views), atol=1e-12)
                centered = [X - m for X, m in zip(self.views, model.means_)]
                lhs, rhs = _construct_mcca_gevp(centered, regs)
                W = np.concatenate(model.loadings_)
                assert_allclose(lhs @ W, (rhs @ W) * model.evals_, atol=1e-10)
                assert_allclose(W.T @ rhs @ W, np.eye(2), atol=1e-10)
                self.assertEqual(model.canon_corrs(scores).shape, (2, 3, 3))
                self.assertEqual(model.n_components_, 2)

    def test_common_output_inverse_and_score(self):
        model = MCCA(n_components=2, regs=0.1, multiview_output=False)
        common = model.fit_transform(self.views)
        assert_allclose(common, model.transform(self.views), atol=1e-12)
        assert_allclose(np.linalg.norm(common, axis=0), 1)
        scores = [model.transform_view(X, i) for i, X in enumerate(self.views)]
        reconstructions = model.inverse_transform(scores)
        errors = [np.sum((X - restored) ** 2) for X, restored in zip(self.views, reconstructions)]
        assert_allclose(model.score(self.views), np.mean(errors))
        for i in range(3):
            assert_allclose(model.score_view(self.views[i], i), errors[i])
        model.set_params(multiview_output=True)
        assert_allclose(model.score(self.views), errors)

    def test_informative_mcca_methods(self):
        for method, regs in (("auto", None), ("svd", None), ("gevp", None),
                             ("auto", 0.1), ("gevp", 0.1)):
            with self.subTest(method=method, regs=regs):
                model = MCCA(n_components=2, signal_ranks=[2, 3, 3], regs=regs,
                             i_mcca_method=method)
                scores = model.fit_transform(self.views)
                assert_allclose(scores, model.transform(self.views), atol=1e-10)
                self.assertTrue(np.isfinite(scores).all())
        svd = MCCA(n_components=2, signal_ranks=2, i_mcca_method="svd").fit(self.views)
        gevp = MCCA(n_components=2, signal_ranks=2, i_mcca_method="gevp").fit(self.views)
        assert_allclose(svd.evals_, gevp.evals_, atol=1e-10)

    def test_component_options_centering_and_clone(self):
        for components, expected in (("min", 3), ("max", 5), (None, 12)):
            with self.subTest(components=components):
                model = MCCA(n_components=components, regs=0.2, center=[True, False, True])
                fitted = clone(model).fit(self.views)
                self.assertEqual(fitted.n_components_, expected)
                self.assertIsNone(fitted.means_[1])
                self.assertFalse(hasattr(model, "loadings_"))
        two = MCCA(n_components=2, regs=0.1).fit(self.views[:2])
        self.assertEqual(two.canon_corrs(two.transform(self.views[:2])).shape, (2,))

    def test_validation_and_singular_input(self):
        with self.assertRaises(NotFittedError):
            MCCA().transform(self.views)
        with self.assertRaises(ValueError):
            MCCA().fit([self.views[0]])
        with self.assertRaises(ValueError):
            MCCA().fit([self.views[0], self.views[1][:-1]])
        with self.assertRaisesRegex(ValueError, "singular"):
            MCCA().fit([np.ones((5, 3)), np.ones((5, 3))])
        for method, regs in (("invalid", None), ("svd", 0.1)):
            with self.assertRaises(ValueError):
                MCCA(signal_ranks=2, i_mcca_method=method, regs=regs).fit(self.views)


if __name__ == "__main__":
    unittest.main()
