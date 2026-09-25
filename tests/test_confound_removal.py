import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from neural_encoder.utils import ConfoundRemover


class TestConfoundRemover(unittest.TestCase):
    def test_ridge_equation_and_unseen_data(self):
        rng = np.random.default_rng(3)
        C = np.column_stack((np.ones(30), rng.normal(size=(30, 2))))
        X = rng.normal(size=(30, 4)) + C @ rng.normal(size=(3, 4))
        original_X, original_C = X.copy(), C.copy()
        model = ConfoundRemover(alpha=0.2, dtype=np.float64)
        residuals = model.fit_transform(X, C)
        B = np.linalg.solve(C.T @ C + 0.2 * np.eye(3), C.T @ X)
        assert_allclose(model.coef_, B.T, atol=1e-12)
        assert_allclose(residuals, X - C @ B, atol=1e-12)
        X_test, C_test = rng.normal(size=(5, 4)), rng.normal(size=(5, 3))
        assert_allclose(model.transform(X_test, C_test), X_test - C_test @ B, atol=1e-12)
        assert_allclose(model.coef_, B.T, atol=1e-12)
        assert_array_equal(X, original_X)
        assert_array_equal(C, original_C)

    def test_single_feature_and_rank_deficient_design(self):
        C = np.array([[1., 1.], [2., 2.], [3., 3.]])
        X = np.array([[2.], [4.], [6.]])
        model = ConfoundRemover(alpha=0, dtype=np.float64)
        assert_allclose(model.fit_transform(X, C), 0, atol=1e-12)
        self.assertEqual(model.coef_.shape, (1, 2))
        assert_allclose(model.coef_.T, np.linalg.lstsq(C, X, rcond=None)[0], atol=1e-12)

    def test_default_dtype_dummy_confounds_and_clone(self):
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        C = np.repeat(np.eye(2), 2, axis=0)
        template = ConfoundRemover()
        model = clone(template)
        residuals = model.fit_transform(X, C)
        expected_B = np.linalg.solve(C.T @ C + 1e-3 * np.eye(2), C.T @ X)
        assert_allclose(residuals, np.array(X) - C @ expected_B, atol=2e-6)
        self.assertEqual(residuals.dtype, np.float32)
        self.assertFalse(hasattr(template, "coef_"))
        assert_array_equal(model.get_feature_names_out(), ["x0", "x1"])
        model.fit([[1], [2]], [[1], [1]])
        self.assertEqual(model.n_confounds_in_, 1)
        self.assertEqual(model.coef_.shape, (1, 1))

    def test_validation(self):
        X, C = [[1, 2], [3, 4]], [[1], [1]]
        with self.assertRaises(NotFittedError):
            ConfoundRemover().transform(X, C)
        for kwargs in ({"alpha": -1}, {"alpha": np.inf}, {"dtype": int}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ConfoundRemover(**kwargs).fit(X, C)
        for invalid in (None, [1, 1], [[1]], [[np.nan], [1]]):
            with self.assertRaises(ValueError):
                ConfoundRemover().fit(X, invalid)
        model = ConfoundRemover().fit(X, C)
        for data, confounds in ((X, None), (X, [[1, 2], [3, 4]]), ([[1], [2]], C)):
            with self.assertRaises(ValueError):
                model.transform(data, confounds)


if __name__ == "__main__":
    unittest.main()
