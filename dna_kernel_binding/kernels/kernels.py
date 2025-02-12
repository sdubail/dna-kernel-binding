"""
Module implementing kernel functions for DNA sequence analysis.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import pandas as pd


class BaseKernel(ABC):
    """
    Abstract base class for all kernel functions.
    Provides common functionality like Gram matrix computation and centering,
    while letting specific kernels define their own similarity measures.
    """

    def __init__(self, center: bool = True) -> None:
        """
        Initialize the kernel.

        Args:
            center: Whether to center the kernel matrix in feature space
        """
        self.is_preprocessed = False
        self.preprocessed_data = None
        self.center = center

    def _center_gram_matrix(self, K: np.ndarray, is_train: bool = True) -> np.ndarray:
        """
        Center the Gram matrix in feature space using the formula:
        K^c = (I - U)K(I - U)
        where U is a matrix with all entries = 1/n

        Args:
            K: Gram matrix to center
            is_train: Whether this is training data (if False, use stored U)

        Returns:
            Centered Gram matrix
        """
        if not self.center:
            return K

        n_samples = K.shape[0]

        U = np.ones((n_samples, n_samples)) / n_samples

        # Identity matrix minus U
        I_minus_U = np.eye(n_samples) - U

        # Compute centered matrix using the formula from the slides
        K_centered = I_minus_U @ K @ I_minus_U

        return K_centered

    def compute_gram_matrix(
        self,
        X1: pd.DataFrame | list[str],
        X2: pd.DataFrame | list[str] | None = None,
        center: bool | None = None,
    ) -> np.ndarray:
        """
        Compute the Gram matrix between X1 and X2.
        If X2 is None, compute the Gram matrix between X1 and itself.

        Args:
            X1: First set of instances
            X2: Optional second set of instances
            center: Whether to center the kernel matrix. If None, use the instance setting.

        Returns:
            Gram matrix of kernel values
        """
        if not self.is_preprocessed:
            self.preprocess_data(X1)
            if X2 is not None:
                self.preprocess_data(X2)

        # If X2 is None, we're computing K(X1, X1)
        if X2 is None:
            X2 = X1

        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._compute_similarity(X1.iloc[i], X2.iloc[j], x2_type=x2_type)

        # Center the kernel matrix if requested
        should_center = self.center if center is None else center
        if should_center:
            K = self._center_gram_matrix(K, is_train=(X2 is X1))

        return K

    @abstractmethod
    def preprocess_data(self, X: pd.DataFrame | list[str]) -> None:
        """
        Preprocess the input data for efficient kernel computation.
        Each kernel implementation should define its own preprocessing steps.
        """
        pass

    @abstractmethod
    def _compute_similarity(self, x1: str | np.ndarray, x2: str | np.ndarray, x2_type: Literal["train", "test"]) -> float:
        """
        Compute the kernel value between two instances.
        Must be implemented by each specific kernel.
        """
        pass


class SpectrumKernel(BaseKernel):
    """
    Implementation of the spectrum kernel for DNA sequences.

    The spectrum kernel computes similarity between sequences based on their shared k-mers.
    It uses efficient pre-indexing of k-mers for faster computation.
    """

    def __init__(self, k: int, center: bool = True) -> None:
        """
        Initialize the spectrum kernel.

        Args:
            k: Length of k-mers to consider
        """
        super().__init__(center=center)
        self.k = k
        self.kmer_indices = {}  # Maps sequences to their k-mer count dictionaries
        self.kmer_test_indices = {}

    def _extract_kmers(self, sequence: str) -> dict[str, int]:
        """
        Extract all k-mers from a sequence and count their occurrences.

        Args:
            sequence: Input DNA sequence

        Returns:
            Dictionary mapping k-mers to their counts
        """
        counts: defaultdict[str, int] = defaultdict(int)
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i : i + self.k]
            counts[kmer] += 1
        return dict(counts)

    def preprocess_data(self, X: pd.DataFrame | list[str], X_type: Literal["train", "test"]="train") -> None:
        """
        Preprocess sequences by computing their k-mer count dictionaries.

        Args:
            X: Input sequences
        """
        # Convert to list of strings if DataFrame
        sequences = X.loc[:, 'seq'].tolist() if isinstance(X, pd.DataFrame) else X

        # Create k-mer indices for each sequence
        for idx, seq in enumerate(sequences):
            if X_type == "train":
                self.kmer_indices[idx] = self._extract_kmers(seq)
            else:
                self.kmer_test_indices[idx] = self._extract_kmers(seq)
        self.is_preprocessed = True

    def _compute_similarity(self, x1: str | np.ndarray, x2: str | np.ndarray, x2_type: Literal["train", "test", "validation"]) -> float:
        """
        Compute the spectrum kernel value between two sequences.

        Args:
            x1: First sequence
            x2: Second sequence

        Returns:
            Kernel value based on shared k-mers
        """
        # Get indices for the sequences
        if not hasattr(x1, "name"):
            raise ValueError("A sequence indexation is required for the computation")
        idx1 = x1.name
        if not hasattr(x2, "name"):
            raise ValueError("A sequence indexation is required for the computation")
        idx2 = x2.name

        # Get their k-mer count dictionaries
        kmers1 = self.kmer_indices[idx1]
        if (x2_type == "train") | (x2_type == "validation"):
            kmers2 = self.kmer_indices[idx2]
        else:
            kmers2 = self.kmer_test_indices[idx2]

        # Compute dot product of k-mer counts
        similarity = 0.0
        for kmer, count1 in kmers1.items():
            if kmer in kmers2:
                similarity += count1 * kmers2[kmer]

        return similarity
