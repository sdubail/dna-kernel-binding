"""
Test module for the mismatch spectrum kernel implementation.
Tests both standard functionality and mismatch-specific features.
"""

import numpy as np
import pandas as pd
import pytest

from dna_kernel_binding.kernels.kernels import MismatchSpectrumKernel


@pytest.fixture
def mismatch_kernel() -> MismatchSpectrumKernel:
    """Fixture providing a mismatch spectrum kernel with k=2 and m=1."""
    return MismatchSpectrumKernel(k=2, m=1)


@pytest.fixture
def test_sequences() -> pd.DataFrame:
    """Fixture providing a small dataset of DNA sequences for testing."""
    sequences = [
        "ACGT",
        "AGGT",
        "AAAA",
    ]  # Note: first two sequences differ by 1 mismatch
    return pd.DataFrame(sequences, columns=["sequence"])


@pytest.fixture
def test_sequences_2() -> pd.DataFrame:
    """Fixture providing a second set of DNA sequences for cross-gram testing."""
    sequences = ["TTTT", "CGCG"]
    return pd.DataFrame(sequences, columns=["sequence"])


def test_mismatch_kernel_initialization(
    mismatch_kernel: MismatchSpectrumKernel,
) -> None:
    """Test if the mismatch spectrum kernel initializes correctly with new attributes."""
    assert mismatch_kernel.k == 2
    assert mismatch_kernel.m == 1
    assert not mismatch_kernel.is_preprocessed
    assert len(mismatch_kernel.train_feature_map) == 0
    assert len(mismatch_kernel.test_feature_map) == 0
    assert isinstance(mismatch_kernel._precomputed_neighborhoods, dict)


def test_mismatch_neighborhood_generation(
    mismatch_kernel: MismatchSpectrumKernel,
) -> None:
    """Test if mismatch neighborhood generation works correctly."""
    # Test neighborhood of "AC"
    neighborhood = mismatch_kernel._generate_mismatch_neighborhood("AC")

    # Expected neighbors with 1 mismatch: AA, AG, AT, CC, GC, TC
    expected_neighbors = {"AC", "AA", "AG", "AT", "CC", "GC", "TC"}
    assert neighborhood == expected_neighbors

    # Verify all generated k-mers have the same length
    assert all(len(kmer) == mismatch_kernel.k for kmer in neighborhood)


def test_feature_vector_computation(
    mismatch_kernel: MismatchSpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test if feature vector computation works correctly."""
    sequence = test_sequences.iloc[0]["sequence"]  # "ACGT"
    features = mismatch_kernel._compute_feature_vector(sequence)

    # Verify some properties of the feature vector
    assert isinstance(features, dict)
    # All values should be non-negative
    assert all(v >= 0 for v in features.values())
    # All keys should be k-mers of length k
    assert all(len(kmer) == mismatch_kernel.k for kmer in features.keys())


def test_preprocessing_with_mismatches(
    mismatch_kernel: MismatchSpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test if preprocessing works correctly with mismatches."""
    mismatch_kernel.preprocess_data(test_sequences, X_type="train")

    assert mismatch_kernel.is_preprocessed
    assert len(mismatch_kernel.train_feature_map) == len(test_sequences)
    assert len(mismatch_kernel.test_feature_map) == 0

    # First two sequences are similar (ACGT vs AGGT), should have overlapping features
    features1 = mismatch_kernel.train_feature_map[0]
    features2 = mismatch_kernel.train_feature_map[1]

    # Find common features between the two sequences
    common_features = set(features1.keys()) & set(features2.keys())
    assert len(common_features) > 0


def test_similarity_computation_with_mismatches(
    mismatch_kernel: MismatchSpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test if similarity computation works correctly with mismatches."""
    mismatch_kernel.preprocess_data(test_sequences, X_type="train")

    # Test similarity between sequences differing by one mismatch
    similarity = mismatch_kernel._compute_similarity(
        test_sequences.iloc[0],  # ACGT
        test_sequences.iloc[1],  # AGGT
    )
    assert similarity > 0

    # Test similarity with sequence having no similar k-mers
    similarity = mismatch_kernel._compute_similarity(
        test_sequences.iloc[0],  # ACGT
        test_sequences.iloc[2],  # AAAA
    )
    assert similarity >= 0  # Should be non-negative


def test_gram_matrix_with_mismatches(
    mismatch_kernel: MismatchSpectrumKernel, test_sequences: pd.DataFrame
) -> None:
    """Test if gram matrix computation works correctly with mismatches."""
    K = mismatch_kernel.compute_gram_matrix(test_sequences, center=False)

    # Check matrix properties
    assert K.shape == (len(test_sequences), len(test_sequences))
    assert np.allclose(K, K.T)  # Should be symmetric
    assert np.all(np.diag(K) >= 0)  # Diagonal elements should be non-negative

    # Similar sequences should have higher similarity than dissimilar ones
    assert K[0, 1] > K[0, 2]  # ACGT vs AGGT should be more similar than ACGT vs AAAA


def test_cross_gram_matrix_with_mismatches(
    mismatch_kernel: MismatchSpectrumKernel,
    test_sequences: pd.DataFrame,
    test_sequences_2: pd.DataFrame,
) -> None:
    """Test if cross-gram matrix computation works correctly with mismatches."""
    # Create training data with some similar sequences
    df_train = pd.DataFrame(["ACGT", "ACTT", "CCCC", "GGGG"], columns=["sequence"])

    # Create test data with sequences that have some mismatches with training data
    df_test = pd.DataFrame(
        ["AGGT", "ACTA"],  # First sequence has 1 mismatch with ACGT
        columns=["sequence"],
    )

    # Mock support vectors (pretend first and third training examples are support vectors)
    support_vectors = np.array([True, False, True, False])

    # Compute training gram matrix first
    K_train = mismatch_kernel.compute_gram_matrix(df_train)

    # Compute test gram matrix with support vectors and centering
    K_test = mismatch_kernel.compute_gram_matrix(
        df_train,
        df_test,
        center=True,
        x2_type="test",
        support_vectors=support_vectors,
        K_train=K_train,
    )

    # Check dimensions
    assert K_test.shape == (2, 2)  # Only using 2 support vectors
    # Check that similar sequences have higher similarity scores
    # AGGT should be more similar to ACGT than ACTA is
    assert K_test[0, 0] > K_test[0, 1]


def test_edge_cases_with_mismatches(mismatch_kernel: MismatchSpectrumKernel) -> None:
    """Test mismatch kernel with edge cases."""
    # Test with empty sequences
    df_empty = pd.DataFrame(["", ""], columns=["sequence"])
    mismatch_kernel.preprocess_data(df_empty, X_type="train")
    assert all(
        len(features) == 0 for features in mismatch_kernel.train_feature_map.values()
    )

    # Test with sequences shorter than k
    kernel_k3 = MismatchSpectrumKernel(k=3, m=1)
    df_short = pd.DataFrame(["AC", "GT"], columns=["sequence"])
    kernel_k3.preprocess_data(df_short, X_type="train")
    assert all(len(features) == 0 for features in kernel_k3.train_feature_map.values())


def test_neighborhood_caching(mismatch_kernel: MismatchSpectrumKernel) -> None:
    """Test if neighborhood caching works correctly."""
    # Generate neighborhood for the first time
    neighborhood1 = mismatch_kernel._generate_mismatch_neighborhood("AC")

    # Should use cached result
    neighborhood2 = mismatch_kernel._generate_mismatch_neighborhood("AC")

    assert neighborhood1 == neighborhood2
    assert "AC" in mismatch_kernel._precomputed_neighborhoods

    # Clear cache
    mismatch_kernel.clear_test_data()
    assert len(mismatch_kernel._precomputed_neighborhoods) == 0
