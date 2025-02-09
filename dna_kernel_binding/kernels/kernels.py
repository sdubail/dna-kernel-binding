"""
Module implementing kernel functions for DNA sequence analysis.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


class BaseKernel(ABC):
    """
    Abstract base class for all kernel functions.

    This class defines the interface that all kernel implementations must follow.
    It provides common functionality like Gram matrix computation while letting
    specific kernels define their own similarity measures.
    """

    def __init__(self) -> None:
        self.is_preprocessed = False
        self.preprocessed_data = None

    @abstractmethod
    def preprocess_data(self, X: pd.DataFrame | list[str]) -> None:
        """
        Preprocess the input data for efficient kernel computation.
        Each kernel implementation should define its own preprocessing steps.

        Args:
            X: Input data to preprocess
        """
        pass

    @abstractmethod
    def _compute_similarity(self, x1: str | np.ndarray, x2: str | np.ndarray) -> float:
        """
        Compute the kernel value between two instances.
        Must be implemented by each specific kernel.

        Args:
            x1: First instance
            x2: Second instance

        Returns:
            Kernel value (similarity) between x1 and x2
        """
        pass

    def compute_gram_matrix(
        self, X1: pd.DataFrame | list[str], X2: pd.DataFrame | list[str] | None = None
    ) -> np.ndarray:
        """
        Compute the Gram matrix between X1 and X2.
        If X2 is None, compute the Gram matrix between X1 and itself.

        Args:
            X1: First set of instances
            X2: Optional second set of instances

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
                K[i, j] = self._compute_similarity(X1.iloc[i], X2.iloc[j])

        return K


class SpectrumKernel(BaseKernel):
    """
    Implementation of the spectrum kernel for DNA sequences.

    The spectrum kernel computes similarity between sequences based on their shared k-mers.
    It uses efficient pre-indexing of k-mers for faster computation.
    """

    def __init__(self, k: int):
        """
        Initialize the spectrum kernel.

        Args:
            k: Length of k-mers to consider
        """
        super().__init__()
        self.k = k
        self.kmer_indices = {}  # Maps sequences to their k-mer count dictionaries

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

    def preprocess_data(self, X: pd.DataFrame | list[str]) -> None:
        """
        Preprocess sequences by computing their k-mer count dictionaries.

        Args:
            X: Input sequences
        """
        # Convert to list of strings if DataFrame
        sequences = X.iloc[:, 0].tolist() if isinstance(X, pd.DataFrame) else X

        # Create k-mer indices for each sequence
        for idx, seq in enumerate(sequences):
            self.kmer_indices[idx] = self._extract_kmers(seq)

        self.is_preprocessed = True

    def _compute_similarity(self, x1: str | np.ndarray, x2: str | np.ndarray) -> float:
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
        kmers2 = self.kmer_indices[idx2]

        # Compute dot product of k-mer counts
        similarity = 0.0
        for kmer, count1 in kmers1.items():
            if kmer in kmers2:
                similarity += count1 * kmers2[kmer]

        return similarity
