import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from numpy.testing import assert_allclose

from neural_encoder.linear import ChunkedLinearEncoder, GramPCA, LinearEncoder
from neural_encoder.preprocessing import BetaPreprocessor, SessionStandardScaler

N_COMPONENTS = {"pca": 12, "mcca": 3}


class TestChunkedLinearEncoder(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n_samples, n_views, n_features = 50, 3, 300
        signal = rng.normal(size=(n_samples, 6)) @ rng.normal(size=(6, n_features))
        self.samples = np.tile(np.arange(n_samples), n_views)
        self.X = 300 * (signal[self.samples] + rng.normal(size=(len(self.samples), n_features)))
        self.X[:, :20] *= 5  # heavy-tailed features so that clipping matters
        self.sessions = np.repeat([0, 1, 2, 3, 4], [40, 40, 35, 30, 5])
        test_signal = rng.normal(size=(8, 6)) @ rng.normal(size=(6, n_features))
        self.test_X = 300 * (test_signal + rng.normal(size=(8, n_features)))
        self.test_sessions = np.array([0, 1, 2, 3, 4, 7, 7, 1])

    def fit_pair(self, *, scaler_kwargs=None, preprocessor_kwargs=None, sessions=None, **stages):
        """Fit ChunkedLinearEncoder and the equivalent in-memory pipeline."""
        chunked = ChunkedLinearEncoder(
            session_scaler=None if scaler_kwargs is None else "default",
            session_scaler_kwargs=None if scaler_kwargs is None else scaler_kwargs | {"dtype": np.float64},
            preprocessor_kwargs=preprocessor_kwargs,
            pca_kwargs={"n_components": N_COMPONENTS["pca"], "chunk_size": 37},
            distilled_mcca_kwargs={"n_components": N_COMPONENTS["mcca"]},
            random_state=0, **stages,
        )
        Z = chunked.fit_transform(self.X, sample_ids=self.samples, sessions=sessions)

        steps = []
        X = self.X
        if scaler_kwargs is not None:
            scaler = SessionStandardScaler(dtype=np.float64, **scaler_kwargs).fit(X, sessions=sessions)
            steps.append(lambda X, s: scaler.transform(X, sessions=s))
            X = steps[-1](X, sessions)
        if stages.get("preprocessor", "default") is not None:
            preprocessor = BetaPreprocessor(**({"dtype": np.float64} | (preprocessor_kwargs or {}))).fit(X)
            steps.append(lambda X, s: preprocessor.transform(X))
            X = steps[-1](X, sessions)
        linear_stages = {k: v for k, v in stages.items() if k != "preprocessor"}
        reference = LinearEncoder(
            pca=GramPCA(N_COMPONENTS["pca"]),
            distilled_mcca_kwargs={"n_components": N_COMPONENTS["mcca"]},
            random_state=0, **linear_stages,
        ).fit(X, sample_ids=self.samples)

        def reference_transform(X, sessions=None, stage="distilled_mcca"):
            for step in steps:
                X = step(X, sessions)
            return reference.transform_until(X, stage=stage)

        return chunked, Z, reference, reference_transform

    def assert_equivalent(self, chunked, Z, reference, reference_transform, sessions=None, test_sessions=None):
        assert_allclose(Z, reference_transform(self.X, sessions), atol=1e-8)
        assert_allclose(chunked.transform(self.X, sessions=sessions), Z, atol=1e-8)
        assert_allclose(
            chunked.transform(self.test_X, sessions=test_sessions),
            reference_transform(self.test_X, test_sessions), atol=1e-8,
        )
        for stage in ("pca", "cross_view_ridge"):
            if getattr(reference, f"{stage}_") is not None:
                assert_allclose(
                    chunked.transform_until(self.test_X, stage=stage, sessions=test_sessions),
                    reference_transform(self.test_X, test_sessions, stage=stage), atol=1e-8,
                )
        assert_allclose(chunked.coef_, reference.coef_, atol=1e-10)
        assert_allclose(chunked.intercept_, reference.intercept_, atol=1e-8)

    def test_default_preprocessing(self):
        chunked, Z, reference, transform = self.fit_pair()
        self.assert_equivalent(chunked, Z, reference, transform)
        expected = BetaPreprocessor(dtype=np.float64).fit(self.X)
        assert_allclose(chunked.preprocessor_.clip_bounds_, expected.clip_bounds_)
        assert_allclose(chunked.preprocessor_.feature_mean_, expected.feature_mean_)
        assert_allclose(chunked.feature_reweighting_.weights_, reference.feature_reweighting_.weights_)
        self.assertEqual(chunked.feature_reweighting_.n_samples_, 50)

    def test_all_preprocessing_options(self):
        preprocessing = {
            "sample_centering": True, "feature_scaling": True, "normalize": True,
            "quantile_clip": (0.01, 0.98),
        }
        for options in ({}, {"feature_centering": False}):
            with self.subTest(options=options):
                chunked, Z, reference, transform = self.fit_pair(
                    preprocessor_kwargs=preprocessing | options | {"dtype": np.float64},
                    feature_reweighting_kwargs={"weighting": "sqrt"},
                )
                self.assert_equivalent(chunked, Z, reference, transform)
                expected = BetaPreprocessor(dtype=np.float64, **preprocessing, **options).fit(self.X)
                assert_allclose(chunked.preprocessor_.feature_scale_, expected.feature_scale_)
                # The fitted preprocessor works as a regular BetaPreprocessor.
                assert_allclose(chunked.preprocessor_.transform(self.test_X), expected.transform(self.test_X))

    def test_session_standardization(self):
        for kwargs in ({"min_samples": 10}, {"min_samples": 10, "with_mean": False}):
            with self.subTest(kwargs=kwargs):
                chunked, Z, reference, transform = self.fit_pair(
                    scaler_kwargs=kwargs, sessions=self.sessions,
                    preprocessor_kwargs={"scaling": None},
                )
                self.assert_equivalent(
                    chunked, Z, reference, transform, self.sessions, self.test_sessions,
                )
                expected = SessionStandardScaler(dtype=np.float64, **kwargs).fit(self.X, sessions=self.sessions)
                assert_allclose(chunked.session_scaler_.sessions_, expected.sessions_)
                assert_allclose(chunked.session_scaler_.means_, expected.means_)
                assert_allclose(chunked.session_scaler_.scales_, expected.scales_)
                assert_allclose(chunked.session_scaler_.scale_, expected.scale_)

    def test_fixed_bounds_and_disabled_stages(self):
        chunked, Z, reference, transform = self.fit_pair(
            preprocessor_kwargs={"quantile_clip": None, "clip_bounds": (-2.0, 2.0)},
            cross_view_ridge=None,
        )
        self.assert_equivalent(chunked, Z, reference, transform)
        chunked, Z, reference, transform = self.fit_pair(
            preprocessor=None, feature_reweighting=None,
        )
        self.assert_equivalent(chunked, Z, reference, transform)

    def test_missing_values_are_filled(self):
        self.X[3, 5] = np.nan
        chunked, Z, reference, transform = self.fit_pair()
        self.assert_equivalent(chunked, Z, reference, transform)
        with self.assertRaisesRegex(ValueError, "finite"):
            ChunkedLinearEncoder(preprocessor_kwargs={"fill_value": None}).fit(self.X, sample_ids=self.samples)

    def test_rows_select_training_measurements_from_int16_memmap(self):
        rows = np.flatnonzero(self.samples < 40)
        with tempfile.TemporaryDirectory() as directory:
            X = np.lib.format.open_memmap(
                Path(directory) / "X.npy", mode="w+", dtype=np.int16,
                shape=self.X.shape, fortran_order=True,
            )
            X[:] = np.clip(self.X, -32768, 32767)
            values = np.asarray(X, dtype=np.float64)
            kwargs = {
                "pca_kwargs": {"n_components": 8, "chunk_size": 50},
                "distilled_mcca_kwargs": {"n_components": 2}, "random_state": 0,
            }
            chunked = ChunkedLinearEncoder(**kwargs).fit(X, sample_ids=self.samples[rows], rows=rows)
            expected = ChunkedLinearEncoder(**kwargs).fit(values[rows], sample_ids=self.samples[rows])
            test_rows = np.flatnonzero(self.samples >= 40)
            assert_allclose(chunked.transform(X, rows=test_rows), expected.transform(values[test_rows]), atol=1e-8)
            assert_allclose(np.asarray(X, dtype=np.float64), values)
            del X

    def test_lazy_array_like_matches_array(self):
        class LazyArray:
            def __init__(self, values):
                self.values, self.shape, self.dtype, self.reads = values, values.shape, values.dtype, []

            def __len__(self):
                return self.shape[0]

            def __getitem__(self, key):
                self.reads.append(key)
                return self.values[key].copy()

        rows = np.flatnonzero(self.samples < 40)
        test_rows = np.flatnonzero(self.samples >= 40)
        X = LazyArray(self.X)
        kwargs = {
            "session_scaler": "default", "session_scaler_kwargs": {"min_samples": 10},
            "pca_kwargs": {"n_components": 8, "chunk_size": 50},
            "distilled_mcca_kwargs": {"n_components": 2}, "random_state": 0,
        }
        chunked = ChunkedLinearEncoder(**kwargs).fit(
            X, sample_ids=self.samples[rows], sessions=self.sessions[rows], rows=rows,
        )
        expected = ChunkedLinearEncoder(**kwargs).fit(
            self.X, sample_ids=self.samples[rows], sessions=self.sessions[rows], rows=rows,
        )
        assert_allclose(
            chunked.transform(X, sessions=self.sessions[test_rows], rows=test_rows),
            expected.transform(self.X, sessions=self.sessions[test_rows], rows=test_rows), atol=1e-8,
        )
        self.assertTrue(all(isinstance(key[1], slice) and key[1].stop - key[1].start <= 50 for key in X.reads))

    def test_sampled_quantiles_approximate_exact_bounds(self):
        chunked = ChunkedLinearEncoder(
            preprocessor_kwargs={"quantile_clip": 0.01},
            pca_kwargs={"n_components": 8}, distilled_mcca_kwargs={"n_components": 2},
            quantile_samples=10_000, random_state=0,
        ).fit(self.X, sample_ids=self.samples)
        lower, upper = chunked.preprocessor_.clip_bounds_
        values = self.X / 300
        self.assertAlmostEqual(np.mean(values < lower), 0.01, delta=0.003)
        self.assertAlmostEqual(np.mean(values > upper), 0.01, delta=0.003)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_matches_cpu(self):
        kwargs = {
            "preprocessor_kwargs": {"sample_centering": True, "normalize": True},
            "distilled_mcca_kwargs": {"n_components": 2}, "random_state": 0,
        }
        cpu = ChunkedLinearEncoder(pca_kwargs={"n_components": 8}, **kwargs).fit(self.X, sample_ids=self.samples)
        cuda = ChunkedLinearEncoder(pca_kwargs={"n_components": 8, "device": "cuda"}, **kwargs).fit(
            self.X, sample_ids=self.samples,
        )
        assert_allclose(cuda.transform(self.test_X), cpu.transform(self.test_X), atol=1e-8)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "pca cannot be None"):
            ChunkedLinearEncoder(pca=None).fit(self.X, sample_ids=self.samples)
        with self.assertRaisesRegex(ValueError, "Callable"):
            ChunkedLinearEncoder(feature_reweighting_kwargs={"weighting": np.sqrt}).fit(
                self.X, sample_ids=self.samples,
            )
        with self.assertRaisesRegex(ValueError, "sessions is required"):
            ChunkedLinearEncoder(session_scaler="default").fit(self.X, sample_ids=self.samples)
        with self.assertRaisesRegex(ValueError, "requires a session_scaler"):
            ChunkedLinearEncoder().fit(self.X, sample_ids=self.samples, sessions=self.sessions)
        with self.assertRaisesRegex(ValueError, "quantile_samples"):
            ChunkedLinearEncoder(quantile_samples=0).fit(self.X, sample_ids=self.samples)
        with self.assertRaisesRegex(ValueError, "rows"):
            ChunkedLinearEncoder().fit(self.X, sample_ids=self.samples, rows=[0, 10_000])
        model = ChunkedLinearEncoder(
            pca_kwargs={"n_components": 4}, distilled_mcca_kwargs={"n_components": 2}, cross_view_ridge=None,
        ).fit(self.X, sample_ids=self.samples)
        with self.assertRaisesRegex(ValueError, "features"):
            model.transform(self.X[:, :-1])
        with self.assertRaisesRegex(ValueError, "disabled"):
            model.transform_until(self.X, stage="cross_view_ridge")


if __name__ == "__main__":
    unittest.main()
