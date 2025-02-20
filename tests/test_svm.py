"""
Test suite for the KernelSVM implementation.
Tests basic functionality, edge cases, and numerical stability.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dna_kernel_binding.models.svm import KernelSVM


class TestKernelSVM:
    @pytest.fixture
    def linear_separable_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a simple linearly separable dataset and its kernel matrix.
        Returns two clusters of points that can be perfectly separated.
        """
        # Create two clusters of points
        X1 = np.array([[1, 1], [2, 2]])
        X2 = np.array([[-1, -1], [-2, -2]])
        X = np.vstack([X1, X2])
        y = np.array([1, 1, 0, 0])

        # Compute linear kernel matrix
        K = np.dot(X, X.T)

        return K, y, X

    @pytest.fixture
    def nonseparable_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a dataset that is not linearly separable to test soft margin behavior.
        """
        # Create data with some overlap
        X = np.array([[1, 1], [0.5, 0.5], [-0.5, -0.5], [-1, -1]])
        y = np.array([1, 0, 1, 0])

        # Compute linear kernel matrix
        K = np.dot(X, X.T)

        return K, y

    def test_init_parameters(self) -> None:
        """Test that initialization parameters are correctly stored."""
        svm = KernelSVM(C=2.0, tol=1e-4)
        assert svm.C == 2.0
        assert svm.tol == 1e-4
        assert svm.dual_coef_ is None
        assert svm.support_vectors_ is None
        assert svm.bias_ is None

    def test_fit_separable(
        self, linear_separable_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test fitting on perfectly separable data."""
        K, y, X = linear_separable_data

        svm = KernelSVM(C=1.0)
        svm.fit(K, y)

        # Check that the model learned something
        assert svm.dual_coef_ is not None
        assert svm.support_vectors_ is not None
        assert svm.bias_ is not None

        # Verify perfect separation on training data
        pred = svm.predict(svm, X, X, "test", K)
        assert_array_equal(pred, y)

    def test_predict_before_fit(
        self, linear_separable_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that prediction before fitting raises an error."""
        K, _, _ = linear_separable_data
        svm = KernelSVM()

        with pytest.raises(RuntimeError):
            svm.predict(K)

    def test_soft_margin(
        self, nonseparable_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that soft margin SVM can handle non-separable data."""
        K, y = nonseparable_data

        # Test with different C values
        svm_hard = KernelSVM(C=1000.0)  # Large C: closer to hard margin
        svm_soft = KernelSVM(C=0.1)  # Small C: more tolerance for errors

        svm_hard.fit(K, y)
        svm_soft.fit(K, y)

        # Soft margin should have fewer support vectors
        assert np.sum(svm_soft.support_vectors_) <= np.sum(svm_hard.support_vectors_)

    def test_decision_function(
        self, linear_separable_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that decision function outputs align with predictions."""
        K, y, _ = linear_separable_data
        svm = KernelSVM()
        svm.fit(K, y)

        # Decision values should be positive for class 1 and negative for class 0
        decision_values = svm.decision_function(K)
        predictions = svm.predict(K)

        assert np.all((decision_values > 0) == (predictions == 1))
        assert np.all((decision_values <= 0) == (predictions == 0))

    def test_support_vector_identification(
        self, linear_separable_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that support vectors are correctly identified."""
        K, y, _ = linear_separable_data
        svm = KernelSVM(tol=1e-5)
        svm.fit(K, y)

        # Check that dual coefficients are zero for non-support vectors
        assert np.all(np.abs(svm.dual_coef_) > svm.tol)

        # Check that we have at least one support vector per class
        sv_labels = y[svm.support_vectors_]
        assert len(np.unique(sv_labels)) == len(np.unique(y))

    def test_numerical_stability(self) -> None:
        """Test numerical stability with poorly scaled data."""
        # Create badly scaled data
        X = np.array([[1000, 1000], [-1000, -1000]])
        y = np.array([1, 0])
        K = np.dot(X, X.T)

        svm = KernelSVM()
        svm.fit(K, y)

        # Should still get reasonable predictions
        pred = svm.predict(K)
        assert_array_equal(pred, y)

    def test_prediction_new_data(
        self, linear_separable_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test predictions on new test points."""
        K_train, y_train, X_train = linear_separable_data

        # Create new test points
        X_test = np.array([[1.5, 1.5], [-1.5, -1.5]])
        K_test = np.dot(X_test, X_train.T)

        svm = KernelSVM()
        svm.fit(K_train, y_train)

        # Test points should be classified correctly
        pred = svm.predict(K_test)
        assert_array_equal(pred, np.array([1, 0]))

    # def test_get_params(self) -> None:
    #     """Test that get_params returns the correct information."""
    #     svm = KernelSVM(C=2.0, tol=1e-4)
    #     params = svm.get_params()

    #     assert params["C"] == 2.0
    #     assert params["tol"] == 1e-4
    #     assert params["n_support"] is None  # Before fitting

    #     # After fitting
    #     K = np.array([[1, 0], [0, 1]])
    #     y = np.array([0, 1])
    #     svm.fit(K, y)

    #     params = svm.get_params()
    #     assert isinstance(params["n_support"], int | np.integer)
