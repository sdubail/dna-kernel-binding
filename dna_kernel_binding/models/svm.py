"""
Implementation of Support Vector Machine using kernel methods.
Follows the representer theorem formulation where the decision function
is expressed as a linear combination of kernel functions.
"""

from typing import Literal

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

from dna_kernel_binding.kernels.kernels import BaseKernel

# Configure solver to be less verbose and more precise
solvers.options["show_progress"] = False
solvers.options["abstol"] = 1e-10
solvers.options["reltol"] = 1e-10
solvers.options["feastol"] = 1e-10


class KernelSVM:
    """
    Support Vector Machine classifier using kernel methods.

    This implementation solves the dual optimization problem:
    min ½αᵀKα - αᵀy subject to 0 ≤ yᵢαᵢ ≤ C

    The decision function follows the representer theorem:
    f(x) = ∑ᵢ αᵢK(xᵢ,x)
    """

    def __init__(self, C: float = 1.0, tol: float = 1e-3):
        """
        Initialize the SVM classifier.

        Args:
            C: Regularization parameter
            tol: Numerical tolerance for support vector detection
        """
        self.C = C
        self.tol = tol

        # These will be set during fitting
        self.dual_coef_: np.ndarray | None = None
        self.support_vectors_: np.ndarray | None = None
        self.bias_: float | None = None

    def _solve_dual_problem(self, K: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve the dual optimization problem using quadratic programming.

        Minimizes: ½αᵀKα - αᵀy
        Subject to: 0 ≤ yᵢαᵢ ≤ C

        Args:
            K: Gram matrix of shape (n_samples, n_samples)
            y: Labels of shape (n_samples,) with values in {-1, 1}

        Returns:
            Optimal values of dual variables α
        """
        n_samples = K.shape[0]

        # Construct QP matrices
        # P = K (the quadratic term)
        P = matrix(K, tc="d")

        # q = -y (the linear term)
        # Convert to double precision and ensure column vector shape
        q = matrix(-y.astype(np.float64).reshape(-1, 1), tc="d")

        # Construct constraints for 0 ≤ yᵢαᵢ ≤ C
        # This means -yᵢαᵢ ≤ 0 and yᵢαᵢ ≤ C
        # We can write this as:
        # [-diag(y)] [α] ≤ [0]
        # [diag(y) ] [α] ≤ [C]
        G = matrix(np.vstack((-np.diag(y), np.diag(y))), tc="d")
        h = matrix(
            np.hstack((np.zeros(n_samples), self.C * np.ones(n_samples))), tc="d"
        )

        # Solve the quadratic program
        solution = solvers.qp(P, q, G, h)

        if solution["status"] != "optimal":
            raise RuntimeError(
                f"Quadratic programming solver failed with status: {solution['status']}"
            )

        return np.array(solution["x"]).flatten()

    def _compute_bias(self, K: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
        """
        Compute bias term using KKT conditions.

        Args:
            K: Gram matrix
            y: Labels in {-1, 1}
            alpha: Optimal dual coefficients

        Returns:
            Bias term
        """
        # Find support vectors (points where 0 < yᵢαᵢ < C)
        ya = y * alpha
        sv_mask = (ya > self.tol) & (ya < self.C - self.tol)
        if not np.any(sv_mask):
            sv_mask = ya > self.tol

        # Compute decision function values for support vectors
        sv_decision = np.sum(alpha[sv_mask].reshape(-1, 1) * K[sv_mask], axis=0)

        # Average difference between target and decision value
        bias = np.mean(y[sv_mask] - sv_decision[sv_mask])
        return float(bias)

    def fit(self, K: np.ndarray, y: np.ndarray) -> "KernelSVM":
        """
        Fit the SVM model using the precomputed kernel matrix.

        Args:
            K: Kernel matrix of shape (n_samples, n_samples)
            y: Labels of shape (n_samples,)

        Returns:
            self
        """
        # Convert labels to {-1, 1}
        y = np.asarray(y)
        y_normalized = np.where(y <= 0, -1, 1)

        # Solve the dual optimization problem
        alpha = self._solve_dual_problem(K, y_normalized)

        # Identify support vectors
        sv_mask = np.abs(alpha) > self.tol
        self.support_vectors_ = sv_mask

        # Store the coefficients (α) for support vectors
        self.dual_coef_ = alpha[sv_mask]

        # Compute bias term
        self.bias_ = self._compute_bias(K, y_normalized, alpha)

        return self

    def decision_function(
        self,
        kernel: BaseKernel,
        X_train: pd.DataFrame | list[str],
        X_test_validation: pd.DataFrame | list[str],
        decision_type: Literal["validation", "test"] = "test",
        K_train: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute decision function values following the representer theorem:
        f(x) = ∑ᵢ αᵢK(xᵢ,x)

        Args:
            K_pred: Kernel matrix between test points and training points
                   shape: (n_test, n_train)

        Returns:
            Decision function values for test points
        """
        if self.dual_coef_ is None:
            raise RuntimeError("Model must be fitted before calling decision_function")

        # Compute only kernel values for support vectors
        K_sv = kernel.compute_gram_matrix(
            X1=X_train,
            X2=X_test_validation,
            x2_type=decision_type,
            support_vectors=self.support_vectors_,
            K_train=K_train,
        )
        K_sv = K_sv.T  # Transpose to match shape of (n_test, n_train)

        # Compute decision values using pure kernel expansion
        return np.dot(K_sv, self.dual_coef_) + self.bias_

    def predict(
        self,
        kernel: BaseKernel,
        X_train: pd.DataFrame | list[str],
        X_test: pd.DataFrame | list[str],
        decision_type: Literal["validation", "test"] = "test",
        K_train: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict class labels for test points.

        Args:
            K_pred: Kernel matrix between test points and training points

        Returns:
            Predicted labels (0 or 1)
        """
        decision = self.decision_function(
            kernel, X_train, X_test, decision_type, K_train
        )
        return np.where(decision <= 0, 0, 1)
