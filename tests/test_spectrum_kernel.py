"""
Test module for the improved kernel implementations.
Tests both standard functionality and the new train/test separation feature.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dna_kernel_binding.kernels.kernels import SpectrumKernel


@pytest.fixture
def spectrum_kernel() -> SpectrumKernel:
    """Fixture providing a spectrum kernel with k=2."""
    return SpectrumKernel(k=2)


@pytest.fixture
def test_sequences() -> pd.DataFrame:
    """Fixture providing a small dataset of DNA sequences for testing."""
    sequences = ["ACGT", "ACGT", "AAAA"]
    return pd.DataFrame(sequences, columns=["sequence"])


@pytest.fixture
def test_sequences_2() -> pd.DataFrame:
    """Fixture providing a second set of DNA sequences for cross-gram testing."""
    sequences = ["TTTT", "CGCG"]
    return pd.DataFrame(sequences, columns=["sequence"])


def test_spectrum_kernel_initialization(spectrum_kernel: SpectrumKernel) -> None:
    """Test if the spectrum kernel initializes correctly with new attributes."""
    assert spectrum_kernel.k == 2
    assert not spectrum_kernel.is_preprocessed
    assert len(spectrum_kernel.train_kmer_indices) == 0
    assert len(spectrum_kernel.test_kmer_indices) == 0
    assert spectrum_kernel.K_train_uncentered is None
    assert spectrum_kernel.n_train_samples is None


def test_kmer_extraction(spectrum_kernel: SpectrumKernel) -> None:
    """Test if k-mer extraction works correctly."""
    sequence = "ACGT"
    kmers = spectrum_kernel._extract_kmers(sequence)
    expected_kmers = {"AC": 1, "CG": 1, "GT": 1}
    assert kmers == expected_kmers


def test_preprocessing_train_data(
    spectrum_kernel: SpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test if preprocessing works correctly for training data."""
    spectrum_kernel.preprocess_data(test_sequences, is_train=True)

    assert spectrum_kernel.is_preprocessed
    assert len(spectrum_kernel.train_kmer_indices) == len(test_sequences)
    assert len(spectrum_kernel.test_kmer_indices) == 0

    # Check if first two sequences have same k-mer counts (they're identical)
    assert (
        spectrum_kernel.train_kmer_indices[0] == spectrum_kernel.train_kmer_indices[1]
    )

    # Check third sequence is different
    assert (
        spectrum_kernel.train_kmer_indices[0] != spectrum_kernel.train_kmer_indices[2]
    )


def test_preprocessing_test_data(
    spectrum_kernel: SpectrumKernel,
    test_sequences: pd.DataFrame,
    test_sequences_2: pd.DataFrame,
) -> None:
    """Test if preprocessing works correctly for both train and test data."""
    # First preprocess training data
    spectrum_kernel.preprocess_data(test_sequences, is_train=True)
    initial_train_indices = spectrum_kernel.train_kmer_indices.copy()

    # Then preprocess test data
    spectrum_kernel.preprocess_data(test_sequences_2, is_train=False)

    # Check that train indices weren't modified
    assert spectrum_kernel.train_kmer_indices == initial_train_indices
    assert len(spectrum_kernel.test_kmer_indices) == len(test_sequences_2)


def test_similarity_computation_train(
    spectrum_kernel: SpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test if similarity computation works correctly for training data."""
    spectrum_kernel.preprocess_data(test_sequences, is_train=True)

    # Test similarity between identical sequences (first two sequences)
    similarity = spectrum_kernel._compute_similarity(
        test_sequences.iloc[0], test_sequences.iloc[1]
    )
    assert similarity > 0

    # Test similarity between different sequences (first and last sequences)
    similarity = spectrum_kernel._compute_similarity(
        test_sequences.iloc[0], test_sequences.iloc[2]
    )
    assert similarity == 0  # No common 2-mers between ACGT and AAAA


def test_similarity_computation_cross(
    spectrum_kernel: SpectrumKernel,
    test_sequences: pd.DataFrame,
    test_sequences_2: pd.DataFrame,
) -> None:
    """Test if similarity computation works correctly between train and test sets."""
    spectrum_kernel.preprocess_data(test_sequences, is_train=True)
    spectrum_kernel.preprocess_data(test_sequences_2, is_train=False)

    # Test similarity between train and test sequences
    similarity = spectrum_kernel._compute_similarity(
        test_sequences.iloc[0],  # ACGT
        test_sequences_2.iloc[1],  # CGCG
    )
    assert similarity > 0  # Should share the CG k-mer


def test_train_gram_matrix_uncentered(
    spectrum_kernel: SpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test uncentered training gram matrix computation."""
    K = spectrum_kernel.compute_gram_matrix(test_sequences, center=False)

    # Check matrix properties
    assert K.shape == (len(test_sequences), len(test_sequences))
    assert np.allclose(K, K.T)  # Should be symmetric
    assert np.all(np.diag(K) >= 0)  # Diagonal elements should be non-negative

    # First two sequences are identical, so their similarities should be equal
    assert np.isclose(K[0, 0], K[1, 1])
    assert np.isclose(K[0, 1], K[1, 0])

    # Third sequence (AAAA) should have zero similarity with others
    assert np.isclose(K[0, 2], 0)
    assert np.isclose(K[1, 2], 0)


def test_train_kernel_centering(
    spectrum_kernel: SpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test if training kernel matrix is properly centered."""
    # First compute uncentered matrix
    K_uncentered = spectrum_kernel.compute_gram_matrix(test_sequences, center=False)

    # Then compute centered matrix
    K_centered = spectrum_kernel.compute_gram_matrix(test_sequences, center=True)

    # Verify that original training matrix was stored before centering
    assert spectrum_kernel.K_train_uncentered is not None
    assert np.allclose(spectrum_kernel.K_train_uncentered, K_uncentered)
    assert spectrum_kernel.n_train_samples == len(test_sequences)

    # Check that centered matrix is different from uncentered
    assert not np.allclose(K_centered, K_uncentered)

    # Verify centering properties
    assert np.allclose(K_centered.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(K_centered.mean(axis=1), 0, atol=1e-10)


def test_cross_gram_matrix_uncentered(
    spectrum_kernel: SpectrumKernel,
    test_sequences: pd.DataFrame,
    test_sequences_2: pd.DataFrame,
) -> None:
    """Test uncentered cross-gram matrix computation."""
    # First compute training gram matrix (needed for proper initialization)
    K_train = spectrum_kernel.compute_gram_matrix(test_sequences, center=False)

    # Then compute cross gram matrix
    K_cross = spectrum_kernel.compute_gram_matrix(
        test_sequences, test_sequences_2, center=False
    )

    # Check matrix dimensions
    assert K_cross.shape == (len(test_sequences_2), len(test_sequences))

    # Verify specific values based on our test sequences
    # ACGT (from train) and CGCG (from test) share a "CG" k-mer
    assert K_cross[1, 0] > 0  # ACGT vs CGCG
    assert K_cross[1, 1] > 0  # ACGT vs CGCG
    assert K_cross[1, 2] == 0  # AAAA vs CGCG (no common k-mers)


def test_cross_kernel_centering(
    spectrum_kernel: SpectrumKernel,
    test_sequences: pd.DataFrame,
    test_sequences_2: pd.DataFrame,
) -> None:
    """Test if test kernel matrix is properly centered using training statistics."""
    # First compute centered training kernel matrix
    K_train = spectrum_kernel.compute_gram_matrix(test_sequences, center=True)
    original_K_train_uncentered = spectrum_kernel.K_train_uncentered.copy()

    # Then compute centered test kernel matrix
    K_test = spectrum_kernel.compute_gram_matrix(
        test_sequences, test_sequences_2, center=True
    )

    # Verify that K_train_uncentered wasn't modified
    assert np.allclose(original_K_train_uncentered, spectrum_kernel.K_train_uncentered)

    # Manually compute centering to verify correctness
    n_train = len(test_sequences)
    n_test = len(test_sequences_2)

    U_test = np.ones((n_test, n_train)) / n_train
    U_train = np.ones((n_train, n_train)) / n_train

    # Get the uncentered kernel matrix
    K_test_uncentered = spectrum_kernel.compute_gram_matrix(
        test_sequences, test_sequences_2, center=False
    )

    # Apply centering formula manually
    K_test_centered_manual = (
        K_test_uncentered
        - U_test @ spectrum_kernel.K_train_uncentered
        - K_test_uncentered @ U_train
        + U_test @ spectrum_kernel.K_train_uncentered @ U_train
    )

    # Verify that our implementation matches the manual computation
    assert np.allclose(K_test, K_test_centered_manual)


def test_centering_disabled(
    spectrum_kernel: SpectrumKernel,
    test_sequences: pd.DataFrame,
    test_sequences_2: pd.DataFrame,
) -> None:
    """Test that matrices are unchanged when centering is disabled."""
    # Create kernel with centering disabled
    kernel_no_center = SpectrumKernel(k=2, center=False)

    # Compute matrices with centering disabled
    K_train = kernel_no_center.compute_gram_matrix(test_sequences)
    K_test = kernel_no_center.compute_gram_matrix(test_sequences, test_sequences_2)

    # Verify that K_train_uncentered wasn't created
    assert kernel_no_center.K_train_uncentered is None

    # Compare with explicitly uncentered matrices from regular kernel
    K_train_raw = spectrum_kernel.compute_gram_matrix(test_sequences, center=False)
    K_test_raw = spectrum_kernel.compute_gram_matrix(
        test_sequences, test_sequences_2, center=False
    )

    assert np.allclose(K_train, K_train_raw)
    assert np.allclose(K_test, K_test_raw)


def test_centering_error_handling(
    spectrum_kernel: SpectrumKernel,
    test_sequences: pd.DataFrame,
    test_sequences_2: pd.DataFrame,
) -> None:
    """Test error handling when trying to center test data without training data."""
    # Try to compute centered test kernel matrix without first computing training matrix
    spectrum_kernel.K_train_uncentered = None
    spectrum_kernel.n_train_samples = None

    with pytest.raises(
        ValueError, match="Must process training data before test data for centering"
    ):
        spectrum_kernel.compute_gram_matrix(
            test_sequences, test_sequences_2, center=True
        )


def test_multiple_test_sets(
    spectrum_kernel: SpectrumKernel,
    test_sequences: pd.DataFrame,
    test_sequences_2: pd.DataFrame,
) -> None:
    """Test handling of multiple test sets in sequence."""
    # First compute training gram matrix
    K_train = spectrum_kernel.compute_gram_matrix(test_sequences, center=True)

    # Compute first cross-gram matrix
    K1 = spectrum_kernel.compute_gram_matrix(
        test_sequences, test_sequences_2, center=True
    )

    # Create a third set of sequences
    test_sequences_3 = pd.DataFrame(["GGCC", "AATT"], columns=["sequence"])

    # Clear test data and compute second cross-gram matrix
    spectrum_kernel.clear_test_data()
    K2 = spectrum_kernel.compute_gram_matrix(
        test_sequences, test_sequences_3, center=True
    )

    # Check that matrices have correct shapes
    assert K1.shape == (len(test_sequences_2), len(test_sequences))
    assert K2.shape == (len(test_sequences_3), len(test_sequences))


def test_error_no_name(spectrum_kernel: SpectrumKernel) -> None:
    """Test if error is raised when sequence has no name attribute."""
    with pytest.raises(
        ValueError, match="Sequences must have index information for kernel computation"
    ):
        spectrum_kernel._compute_similarity("ACGT", "ACGT")


def test_spectrum_kernel_edge_cases(spectrum_kernel: SpectrumKernel) -> None:
    """Test spectrum kernel with edge cases."""
    # Test with empty sequences
    df_empty = pd.DataFrame(["", ""], columns=["sequence"])
    spectrum_kernel.preprocess_data(df_empty, is_train=True)
    assert all(len(kmers) == 0 for kmers in spectrum_kernel.train_kmer_indices.values())

    # Test with sequences shorter than k
    kernel_k3 = SpectrumKernel(k=3)
    df_short = pd.DataFrame(["AC", "GT"], columns=["sequence"])
    kernel_k3.preprocess_data(df_short, is_train=True)
    assert all(len(kmers) == 0 for kmers in kernel_k3.train_kmer_indices.values())
