"""
Module implementing kernel functions for DNA sequence analysis.
This implementation includes proper handling of train/test data for gram matrix computation.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, List, Union

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
        self.n_train_samples = None  # Store number of training samples for centering
        self.K_train_uncentered = None

    def _center_gram_matrix(self, K: np.ndarray, is_train: bool = True) -> np.ndarray:
        if not self.center:
            return K

        if is_train:
            # Store the uncentered training matrix first
            self.K_train_uncentered = K.copy()  # Important: store before centering

            n_samples = K.shape[0]
            U = np.ones((n_samples, n_samples)) / n_samples
            I_minus_U = np.eye(n_samples) - U
            K_centered = I_minus_U @ K @ I_minus_U
            self.n_train_samples = n_samples
            return K_centered
        else:
            if self.n_train_samples is None or self.K_train_uncentered is None:
                raise ValueError(
                    "Must process training data before test data for centering"
                )

            n_test = K.shape[0]
            n_train = self.n_train_samples

            U_test = np.ones((n_test, n_train)) / n_train
            U_train = np.ones((n_train, n_train)) / n_train

            # Use the uncentered training matrix for test data centering
            K_centered = (
                K
                - U_test @ self.K_train_uncentered
                - K @ U_train
                + U_test @ self.K_train_uncentered @ U_train
            )
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
            X1: First set of instances (training data)
            X2: Optional second set of instances (test data)
            center: Whether to center the kernel matrix. If None, use the instance setting.

        Returns:
            Gram matrix of kernel values
        """
        if not self.is_preprocessed:
            self.preprocess_data(X1, is_train=True)
            if X2 is not None:
                self.preprocess_data(X2, is_train=False)

        # If X2 is None, we're computing K(X1, X1)
        if X2 is None:
            X2 = X1

        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._compute_similarity(X1.iloc[i], X2.iloc[j])

        if (
            X2 is not None
        ):  # we are computing K(X_test, X_train) so we must transpose it
            K = K.T

        # Center the kernel matrix if requested
        should_center = self.center if center is None else center
        if should_center:
            K = self._center_gram_matrix(K, is_train=(X2 is X1))

        return K

    @abstractmethod
    def preprocess_data(
        self, X: pd.DataFrame | list[str], is_train: bool = True
    ) -> None:
        """
        Preprocess the input data for efficient kernel computation.
        Each kernel implementation should define its own preprocessing steps.

        Args:
            X: Input data to preprocess
            is_train: Whether this is training data (X1) or test data (X2)
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
            Kernel similarity value
        """
        pass


class SpectrumKernel(BaseKernel):
    """
    Implementation of the spectrum kernel for DNA sequences.

    The spectrum kernel computes similarity between sequences based on their shared k-mers.
    It maintains separate indices for training and test data to avoid overwriting issues
    when computing cross-gram matrices.
    """

    def __init__(self, k: int, center: bool = True):
        """
        Initialize the spectrum kernel.

        Args:
            k: Length of k-mers to consider
            center: Whether to center the kernel matrix
        """
        super().__init__(center=center)
        self.k = k
        # Separate dictionaries for training and test data
        self.train_kmer_indices = {}  # Maps training sequences to k-mer counts
        self.test_kmer_indices = {}  # Maps test sequences to k-mer counts

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

    def preprocess_data(
        self, X: pd.DataFrame | list[str], is_train: bool = True
    ) -> None:
        """
        Preprocess sequences by computing their k-mer count dictionaries.
        Stores results in either train_kmer_indices or test_kmer_indices based on is_train.

        Args:
            X: Input sequences
            is_train: Whether this is training data (X1) or test data (X2)
        """
        # Convert to list of strings if DataFrame
        sequences = X.iloc[:, 0].tolist() if isinstance(X, pd.DataFrame) else X

        # Choose appropriate dictionary based on whether this is train or test data
        target_dict = self.train_kmer_indices if is_train else self.test_kmer_indices

        # Clear the target dictionary before adding new data
        target_dict.clear()

        # Create k-mer indices for each sequence
        for idx, seq in enumerate(sequences):
            target_dict[idx] = self._extract_kmers(seq)

        self.is_preprocessed = True

    def _compute_similarity(self, x1: str | np.ndarray, x2: str | np.ndarray) -> float:
        """
        Compute the spectrum kernel value between two sequences.

        Args:
            x1: First sequence (from training set)
            x2: Second sequence (from training or test set)

        Returns:
            Kernel value based on shared k-mers

        Raises:
            ValueError: If sequences lack required indexing information
        """
        # Validate sequence indices
        if not hasattr(x1, "name") or not hasattr(x2, "name"):
            raise ValueError(
                "Sequences must have index information for kernel computation"
            )

        idx1, idx2 = x1.name, x2.name

        # Get k-mer counts for first sequence (always from training set)
        kmers1 = self.train_kmer_indices[idx1]

        # Get k-mer counts for second sequence (from test set if available, else training)
        kmers2 = (
            self.test_kmer_indices
            if self.test_kmer_indices
            else self.train_kmer_indices
        )[idx2]

        # Compute dot product of k-mer counts
        similarity = 0.0
        for kmer, count1 in kmers1.items():
            if kmer in kmers2:
                similarity += count1 * kmers2[kmer]

        return similarity

    def clear_test_data(self) -> None:
        """
        Clear the test data k-mer indices.
        Useful when computing multiple different cross-gram matrices.
        """
        self.test_kmer_indices.clear()


class MismatchSpectrumKernel(BaseKernel):
    """
    Implementation of the spectrum kernel with mismatches for DNA sequences.
    Allows for efficient computation of similarity between sequences based on their k-mers
    with up to m mismatches permitted.
    """

    def __init__(self, k: int, m: int, center: bool = True):
        """
        Initialize the mismatch spectrum kernel.

        Args:
            k: Length of k-mers to consider
            m: Maximum number of mismatches allowed
            center: Whether to center the kernel matrix
        """
        super().__init__(center=center)
        self.k = k
        self.m = m
        self.alphabet = ["A", "C", "G", "T"]
        self.train_feature_map = {}  # Maps training sequences to their feature vectors
        self.test_feature_map = {}  # Maps test sequences to their feature vectors
        self._precomputed_neighborhoods = {}  # Cache for mismatch neighborhoods

    def _generate_mismatch_neighborhood(self, kmer: str) -> set[str]:
        """
        Generate all k-mers that differ from the input k-mer by at most m positions.
        Uses dynamic programming to avoid regenerating neighborhoods.

        Args:
            kmer: Input k-mer sequence

        Returns:
            Set of all k-mers within hamming distance m
        """
        if kmer in self._precomputed_neighborhoods:
            return self._precomputed_neighborhoods[kmer]

        def _recursive_mismatches(
            pattern: str, pos: int, mismatches_left: int
        ) -> set[str]:
            if pos == len(pattern):
                return {pattern}
            if mismatches_left == 0:
                return {pattern}

            results = set()
            # Keep original nucleotide
            results.update(_recursive_mismatches(pattern, pos + 1, mismatches_left))

            # Try substituting with other nucleotides
            if mismatches_left > 0:
                orig_char = pattern[pos]
                pattern_list = list(pattern)
                for nucleotide in self.alphabet:
                    if nucleotide != orig_char:
                        pattern_list[pos] = nucleotide
                        new_pattern = "".join(pattern_list)
                        results.update(
                            _recursive_mismatches(
                                new_pattern, pos + 1, mismatches_left - 1
                            )
                        )
                pattern_list[pos] = orig_char

            return results

        neighborhood = _recursive_mismatches(kmer, 0, self.m)
        # Verify that all generated k-mers have the same length as the input
        assert all(
            len(neighbor) == self.k for neighbor in neighborhood
        ), f"Generated k-mers must all have length {self.k}"
        self._precomputed_neighborhoods[kmer] = neighborhood
        return neighborhood

    def _compute_feature_vector(self, sequence: str) -> dict[str, float]:
        """
        Compute the feature vector for a sequence, considering mismatches.
        Uses memoization to avoid recomputing mismatch neighborhoods.

        Args:
            sequence: Input DNA sequence

        Returns:
            Dictionary mapping canonical k-mers to their weighted counts
        """
        features = defaultdict(float)

        # First, count exact k-mers in the sequence
        exact_kmers = defaultdict(int)
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i : i + self.k]
            exact_kmers[kmer] += 1

        # For each observed k-mer, distribute its count across its mismatch neighborhood
        # This precomputation is mathematically equivalent to checking neighborhoods during
        # similarity computation because:
        # 1. For k-mers within m mismatches, their neighborhoods overlap completely
        # 2. For k-mers beyond m mismatches, their neighborhoods don't overlap
        # Therefore, the dot product of these weighted features gives the same result
        # as counting matching k-mers within m mismatches
        for kmer, count in exact_kmers.items():
            neighborhood = self._generate_mismatch_neighborhood(kmer)
            weight = 1.0 / len(neighborhood)  # Distribute count evenly
            for neighbor in neighborhood:
                features[neighbor] += count * weight

        return dict(features)

    def preprocess_data(
        self, X: pd.DataFrame | list[str], is_train: bool = True
    ) -> None:
        """
        Preprocess sequences by computing their feature vectors with mismatch consideration.

        Args:
            X: Input sequences
            is_train: Whether this is training data (X1) or test data (X2)
        """
        sequences = X.iloc[:, 0].tolist() if isinstance(X, pd.DataFrame) else X
        target_map = self.train_feature_map if is_train else self.test_feature_map

        target_map.clear()
        for idx, seq in enumerate(sequences):
            target_map[idx] = self._compute_feature_vector(seq)

        self.is_preprocessed = True

    def _compute_similarity(self, x1: str | np.ndarray, x2: str | np.ndarray) -> float:
        """
        Compute the mismatch spectrum kernel value between two sequences.

        Args:
            x1: First sequence (from training set)
            x2: Second sequence (from training or test set)

        Returns:
            Kernel value based on shared k-mers considering mismatches
        """
        if not hasattr(x1, "name") or not hasattr(x2, "name"):
            raise ValueError(
                "Sequences must have index information for kernel computation"
            )

        idx1, idx2 = x1.name, x2.name

        # Get feature vectors
        features1 = self.train_feature_map[idx1]
        features2 = (
            self.test_feature_map if self.test_feature_map else self.train_feature_map
        )[idx2]

        # Compute dot product of feature vectors
        similarity = 0.0
        for kmer, value1 in features1.items():
            if kmer in features2:
                similarity += value1 * features2[kmer]

        return similarity

    def clear_test_data(self) -> None:
        """Clear the test data feature vectors and mismatch neighborhood cache."""
        self.test_feature_map.clear()
        self._precomputed_neighborhoods.clear()  # Optional: clear cache when done
