"""
Test suite for the KernelSVM implementation.
Tests basic functionality, edge cases, and numerical stability.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dna_kernel_binding.kernels.kernels import LinearKernel
from dna_kernel_binding.models.svm import KernelSVM


class TestKernelSVM:
    @pytest.fixture
    def linear_kernel(self) -> LinearKernel:
        """Create a linear kernel instance."""
        return LinearKernel(center=True)

    @pytest.fixture
    def linear_separable_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create a simple linearly separable dataset.
        Returns two clusters of points that can be perfectly separated.
        """
        # Create two clusters of points
        X1 = np.array([[1, 1], [2, 2]])
        X2 = np.array([[-1, -1], [-2, -2]])
        X = np.vstack([X1, X2])
        X_df = pd.DataFrame(X, columns=["feature1", "feature2"])
        y = np.array([1, 1, 0, 0])

        # Compute linear kernel matrix with the kernel class
        kernel = LinearKernel()
        K_uncentered = kernel.compute_gram_matrix(X_df)
        K_centered = kernel._center_gram_matrix(K_uncentered)

        return K_uncentered, K_centered, y, X_df

    @pytest.fixture
    def nonseparable_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a dataset that is not linearly separable to test soft margin behavior.
        """
        # Create data with some overlap
        X = np.array([[1, 1], [0.5, 0.5], [-0.5, -0.5], [-1, -1]])
        X_df = pd.DataFrame(X, columns=["feature1", "feature2"])
        y = np.array([1, 0, 1, 0])

        # Compute linear kernel matrix with the kernel class
        kernel = LinearKernel()
        K = kernel.compute_gram_matrix(X_df)

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
        self,
        linear_kernel: LinearKernel,
        linear_separable_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        """Test fitting on perfectly separable data."""
        K_uncentered, K_centered, y, X_df = linear_separable_data

        svm = KernelSVM(C=1.0)
        svm.fit(K_centered, y)

        # Check that the model learned something
        assert svm.dual_coef_ is not None
        assert svm.support_vectors_ is not None
        assert svm.bias_ is not None

        # Verify perfect separation on training data
        pred = svm.predict(linear_kernel, X_df, X_df, "test", K_uncentered)
        assert_array_equal(pred, y)

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
        self,
        linear_kernel: LinearKernel,
        linear_separable_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        """Test that decision function outputs align with predictions."""
        K_uncentered, K_centered, y, X_df = linear_separable_data
        svm = KernelSVM()
        svm.fit(K_centered, y)

        # Decision values should be positive for class 1 and negative for class 0
        decision_values = svm.decision_function(
            linear_kernel, X_df, X_df, "test", K_uncentered
        )
        predictions = svm.predict(linear_kernel, X_df, X_df, "test", K_uncentered)

        assert np.all((decision_values > 0) == (predictions == 1))
        assert np.all((decision_values <= 0) == (predictions == 0))

    def test_support_vector_identification(
        self,
        linear_separable_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        """Test that support vectors are correctly identified."""
        K_uncentered, K_centered, y, _ = linear_separable_data
        svm = KernelSVM(tol=1e-5)
        svm.fit(K_centered, y)

        # Check that dual coefficients are zero for non-support vectors
        assert np.all(np.abs(svm.dual_coef_) > svm.tol)

        # Check that we have at least one support vector per class
        sv_labels = y[svm.support_vectors_]
        assert len(np.unique(sv_labels)) == len(np.unique(y))

    def test_numerical_stability(self, linear_kernel: LinearKernel) -> None:
        """Test numerical stability with poorly scaled data."""
        # Create badly scaled data
        X = np.array([[1000, 1000], [-1000, -1000]])
        X_df = pd.DataFrame(X, columns=["feature1", "feature2"])
        y = np.array([1, 0])
        K_uncentered = linear_kernel.compute_gram_matrix(X_df, center=False)
        K_centered = linear_kernel._center_gram_matrix(K_uncentered)
        svm = KernelSVM()
        svm.fit(K_centered, y)

        # Should still get reasonable predictions
        pred = svm.predict(linear_kernel, X_df, X_df, "test", K_uncentered)
        import pdb

        pdb.set_trace()
        assert_array_equal(pred, y)

    def test_prediction_new_data(
        self,
        linear_kernel: LinearKernel,
        linear_separable_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        """Test predictions on new test points."""
        K_uncentered, K_centered, y_train, X_train = linear_separable_data

        # Create new test points
        X_test = np.array([[1.5, 1.5], [-1.5, -1.5]])
        X_test_df = pd.DataFrame(X_test, columns=["feature1", "feature2"])

        svm = KernelSVM()
        svm.fit(K_centered, y_train)

        # Test points should be classified correctly
        pred = svm.predict(linear_kernel, X_train, X_test_df, "test", K_uncentered)
        assert_array_equal(pred, np.array([1, 0]))
