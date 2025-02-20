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

    def _center_gram_matrix(self, K: np.ndarray, is_train: bool = True, support_vectors: np.ndarray=None, K_train:np.ndarray=None) -> np.ndarray:
        """
        Center the Gram matrix in feature space using the formula:
        K^c = (I - U)K(I - U)
        where U is a matrix with all entries = 1/n

        Args:
            K: Gram matrix to center
            is_train: Whether this is training data
                if False, we adapt the formula using:
                    An m×m identity matrix on the left (for m test points)
                    An n×n identity matrix on the right (for n training points)
                    So the formula becomes: K_test_centered = (I_m - 1_m1_nᵀ/m)K_test(I_n - 1_n1_nᵀ/n)
                    

        Returns:
            Centered Gram matrix

        """
        if not self.center:
            return K

        
        if is_train:
            n_samples = K.shape[0]
            # Compute U matrix for training data
            U= np.ones((n_samples, n_samples)) / n_samples
            
            # Identity matrix minus U
            I_minus_U = np.eye(n_samples) - U
        
            # Compute centered matrix using the formula from the slides
            K_centered = I_minus_U @ K @ I_minus_U
            
            return K_centered
            
        else:
            #TODO: Check if this is correct

            # Here K = K_test_sv
            n_train =  K_train.shape[0]
            n_test = K.shape[1]
            n_sv = support_vectors.sum().item()
            
            # Compute required statistics from full training kernel
            mean_full =  K_train.mean()
            mean_cols_sv = K_train[:, support_vectors].mean(axis=0)
            mean_rows_train = K_train.mean(axis=1)
            
            # Center test kernel using these statistics
            K_test_centered = K - \
                            np.outer(np.ones(n_test), mean_cols_sv).T - \
                            np.outer(mean_rows_train[support_vectors], np.ones(n_test)) + \
                            mean_full
                            
            
            return K_test_centered   
        
            # TODO: Remove when the above is correct
            # 1st alternative  
            # n_train, n_test = K.shape
            # # Compute U matrix for test data
            # U_test = np.ones((n_test, n_test)) / n_test
            # U_train = np.ones((n_train, n_train)) / n_train
            # # Identity matrices minus U
            # I_m_minus_U_test = np.eye(n_test) - U_test
            # I_n_minus_U_train = np.eye(n_train) - U_train
            
            # # Compute centered matrix using the formula
            # K_centered = I_n_minus_U_train @ K @ I_m_minus_U_test
            
            # TODO: Remove when the above is correct
            # 2nd alternative    
            # n_train, n_test = K.shape
            # U_test = np.ones((n_test, n_test)) / n_train
            # K_centered = K - U_test @ self.K_train
             

    def compute_gram_matrix(
        self,
        X1: pd.DataFrame | list[str],
        X2: pd.DataFrame | list[str] | None = None,
        center: bool | None = None,
        x2_type: Literal["train", "test", "validation"] = "train",
        support_vectors: np.ndarray | None = None,
        K_train: np.ndarray | None = None
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
            # if x2_type == 'train':
            #     self.preprocess_data(X2, X_type=x2_type)

        # make sure to preprocess the test data
        if x2_type == 'test':
            self.preprocess_data(X2, X_type=x2_type) 
        
        if support_vectors is not None:
            X1 = X1[support_vectors]
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
            is_train = x2_type == "train"
            K = self._center_gram_matrix(K, is_train, support_vectors, K_train)

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

    def _compute_similarity(self, x1: str | np.ndarray, x2: str | np.ndarray, x2_type: Literal["train", "test", "validation"]="train") -> float:
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
