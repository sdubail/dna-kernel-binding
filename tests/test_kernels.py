"""
Test module for kernel implementations.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os
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
    return pd.DataFrame(sequences, columns=["seq"])


def test_spectrum_kernel_initialization(spectrum_kernel: SpectrumKernel) -> None:
    """Test if the spectrum kernel initializes correctly."""
    assert spectrum_kernel.k == 2
    assert not spectrum_kernel.is_preprocessed
    assert len(spectrum_kernel.kmer_indices) == 0


def test_kmer_extraction(spectrum_kernel: SpectrumKernel) -> None:
    """Test if k-mer extraction works correctly."""
    sequence = "ACGT"
    kmers = spectrum_kernel._extract_kmers(sequence)

    expected_kmers = {"AC": 1, "CG": 1, "GT": 1}
    assert kmers == expected_kmers


def test_preprocessing(
    spectrum_kernel: SpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test if preprocessing works correctly."""
    spectrum_kernel.preprocess_data(test_sequences)

    assert spectrum_kernel.is_preprocessed
    assert len(spectrum_kernel.kmer_indices) == len(test_sequences)

    # Check if first two sequences have same k-mer counts
    assert spectrum_kernel.kmer_indices[0] == spectrum_kernel.kmer_indices[1]


def test_similarity_computation(
    spectrum_kernel: SpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test if similarity computation works correctly."""
    spectrum_kernel.preprocess_data(test_sequences)

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


def test_gram_matrix_computation(
    spectrum_kernel: SpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test if Gram matrix computation works correctly."""
    K = spectrum_kernel.compute_gram_matrix(test_sequences)

    # Check matrix properties
    assert K.shape == (len(test_sequences), len(test_sequences))
    assert np.allclose(K, K.T)  # Gram matrix should be symmetric
    assert np.all(np.diag(K) >= 0)  # Diagonal elements should be non-negative


def test_gram_matrix_different_sets(spectrum_kernel: SpectrumKernel) -> None:
    """Test Gram matrix computation between different sets."""
    df1 = pd.DataFrame(["ACGT", "AAAA"], columns=["seq"])
    df2 = pd.DataFrame(["ACTT", "AGGA"], columns=["seq"])

    K = spectrum_kernel.compute_gram_matrix(df1, df2, "test")
    assert K.shape == (2, 2)


def test_error_no_name(spectrum_kernel: SpectrumKernel) -> None:
    """Test if error is raised when sequence has no name attribute."""
    with pytest.raises(
        ValueError, match="A sequence indexation is required for the computation"
    ):
        spectrum_kernel._compute_similarity("ACGT", "ACGT")


def test_spectrum_kernel_edge_cases(spectrum_kernel: SpectrumKernel) -> None:
    """Test spectrum kernel with edge cases."""
    # Test with empty sequences
    df_empty = pd.DataFrame(["", ""], columns=["seq"])
    spectrum_kernel.preprocess_data(df_empty)
    assert all(len(kmers) == 0 for kmers in spectrum_kernel.kmer_indices.values())

    # Test with sequences shorter than k
    kernel_k3 = SpectrumKernel(k=3)
    df_short = pd.DataFrame(["AC", "GT"], columns=["seq"])
    kernel_k3.preprocess_data(df_short)
    assert all(len(kmers) == 0 for kmers in kernel_k3.kmer_indices.values())


def test_spectrum_kernel_normalization(
    spectrum_kernel: SpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test if self-similarity scores make sense."""
    K = spectrum_kernel.compute_gram_matrix(test_sequences)

    # Self-similarity should be largest
    for i in range(len(test_sequences)):
        row = K[i]
        assert np.all(row[i] >= row)  # Diagonal element should be largest in row
