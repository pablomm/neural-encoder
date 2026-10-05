import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from neural_encoder.linear import GramPCA, LinearEncoder


class TestGramPCA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        latent = rng.normal(size=(60, 8)) @ rng.normal(size=(8, 500))
        self.X = latent + 0.5 * rng.normal(size=(60, 500)) + 2.0
        self.test_X = rng.normal(size=(7, 500))

    def assert_matches_sklearn(self, model, reference, atol=1e-10):
        assert_allclose(model.components_, reference.components_, atol=atol)
        assert_allclose(model.mean_, reference.mean_, atol=atol)
        assert_allclose(model.explained_variance_, reference.explained_variance_, rtol=1e-8)
        assert_allclose(model.explained_variance_ratio_, reference.explained_variance_ratio_, rtol=1e-8)
        assert_allclose(model.singular_values_, reference.singular_values_, rtol=1e-8)
        assert_allclose(model.transform(self.test_X), reference.transform(self.test_X), atol=atol)

    def test_matches_sklearn_pca(self):
        for whiten in (False, True):
            with self.subTest(whiten=whiten):
                reference = PCA(10, whiten=whiten, svd_solver="full").fit(self.X)
                model = GramPCA(10, whiten=whiten, chunk_size=77)
                scores = model.fit_transform(self.X)
                self.assert_matches_sklearn(model, reference)
                assert_allclose(scores, reference.transform(self.X), atol=1e-10)
                assert_allclose(scores, model.transform(self.X), atol=1e-10)

    def test_chunk_size_does_not_change_result(self):
        reference = GramPCA(6, chunk_size=500).fit(self.X)
        for chunk_size in (1, 64, 499, 10_000):
            model = GramPCA(6, chunk_size=chunk_size).fit(self.X)
            assert_allclose(model.components_, reference.components_, atol=1e-10)

    def test_more_samples_than_features(self):
        X = np.random.default_rng(1).normal(size=(80, 12))
        reference = PCA(12, svd_solver="full").fit(X)
        model = GramPCA(12, chunk_size=5).fit(X)
        assert_allclose(model.components_, reference.components_, atol=1e-10)
        assert_allclose(model.explained_variance_ratio_.sum(), 1.0)

    def test_memmap_input_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            X = np.lib.format.open_memmap(
                Path(directory) / "X.npy", mode="w+", dtype=np.float32,
                shape=self.X.shape, fortran_order=True,
            )
            X[:] = self.X
            original = np.array(X)
            model = GramPCA(5, chunk_size=128, dtype="float32").fit(X)
            assert_allclose(X, original)
            reference = PCA(5, svd_solver="full").fit(original.astype(np.float64))
            assert_allclose(model.components_, reference.components_, atol=1e-4)
            self.assertEqual(model.transform(X).dtype, np.float32)
            del X

    def test_float32(self):
        reference = PCA(10, svd_solver="full").fit(self.X)
        model = GramPCA(10, dtype="float32").fit(self.X)
        self.assertEqual(model.components_.dtype, np.float32)
        assert_allclose(model.components_, reference.components_, atol=1e-4)
        assert_allclose(model.explained_variance_, reference.explained_variance_, rtol=1e-4)

    def test_float32_refinement_with_partial_basis(self):
        # 200 samples and 10 components: the refined subspace has 74 vectors.
        rng = np.random.default_rng(4)
        X = rng.normal(size=(200, 600)) * np.arange(1, 601) ** -0.5
        reference = PCA(10, svd_solver="full").fit(X)
        model = GramPCA(10, dtype="float32", chunk_size=128).fit(X)
        assert_allclose(model.explained_variance_, reference.explained_variance_, rtol=1e-5)
        assert_allclose(model.components_, reference.components_, atol=1e-4)

    def test_rank_deficient_components_are_zero(self):
        X = np.random.default_rng(2).normal(size=(10, 3)) @ np.random.default_rng(3).normal(size=(3, 40))
        model = GramPCA(6).fit(X)
        assert_allclose(model.explained_variance_[3:], 0, atol=1e-10)
        assert_allclose(model.components_[3:], 0)
        assert np.isfinite(model.transform(X)).all()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_matches_cpu(self):
        cpu = GramPCA(10, chunk_size=100).fit(self.X)
        cuda = GramPCA(10, chunk_size=100, device="cuda").fit(self.X)
        assert_allclose(cuda.components_, cpu.components_, atol=1e-10)
        assert_allclose(cuda.transform(self.test_X), cpu.transform(self.test_X), atol=1e-10)

    def test_validation(self):
        with self.assertRaises(NotFittedError):
            GramPCA(2).transform(self.X)
        for params in ({"n_components": 0}, {"chunk_size": 0}, {"n_components": True},
                       {"whiten": 1}, {"dtype": "float16"}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                GramPCA(**({"n_components": 2} | params)).fit(self.X)
        with self.assertRaisesRegex(ValueError, "at most"):
            GramPCA(61).fit(self.X)
        with self.assertRaisesRegex(ValueError, "2D"):
            GramPCA(2).fit(self.X[0])
        X = self.X.copy()
        X[3, 4] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            GramPCA(2).fit(X)
        model = GramPCA(2).fit(self.X)
        with self.assertRaisesRegex(ValueError, "features"):
            model.transform(self.X[:, :-1])

    def test_clone_and_linear_encoder_stage(self):
        model = clone(GramPCA(4, chunk_size=50))
        self.assertEqual(model.get_params()["chunk_size"], 50)
        samples = np.repeat(np.arange(30), 2)
        reference = LinearEncoder(
            pca=PCA(4, svd_solver="full"), distilled_mcca_kwargs={"n_components": 2}, random_state=0,
        ).fit(self.X, sample_ids=samples)
        encoder = LinearEncoder(
            pca=GramPCA(4, chunk_size=50), distilled_mcca_kwargs={"n_components": 2}, random_state=0,
        ).fit(self.X, sample_ids=samples)
        self.assertIsInstance(encoder.pca_, GramPCA)
        assert_allclose(encoder.transform(self.test_X), reference.transform(self.test_X), atol=1e-8)
        assert_allclose(
            encoder.transform_until(self.test_X, stage="pca"),
            reference.transform_until(self.test_X, stage="pca"), atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
