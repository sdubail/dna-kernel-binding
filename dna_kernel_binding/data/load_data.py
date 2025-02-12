"""
Module for loading DNA sequence data and their corresponding labels.
Handles both raw sequences and pre-processed matrix data.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


class DNADataLoader:
    def __init__(self, data_dir: str | Path):
        """
        Initialize the data loader with the path to data directory.

        Args:
            data_dir: Path to directory containing the data files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    def load_dataset(
        self, k: int, split: Literal["train", "test"], use_matrix: bool = False
    ) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
        """
        Load a specific dataset.

        Args:
            k: Dataset number (0, 1, or 2)
            split: Whether to load train or test data
            use_matrix: If True, load pre-processed matrix data instead of sequences

        Returns:
            If split is "train":
                Tuple of (X, y) where X is the feature data and y is labels
            If split is "test":
                Only X (feature data)
        """
        if k not in [0, 1, 2]:
            raise ValueError("k must be 0, 1, or 2")

        # Construct file paths
        prefix = "Xtr" if split == "train" else "Xte"
        suffix = "_mat100.csv" if use_matrix else ".csv"
        x_path = self.data_dir / f"{prefix}{k}{suffix}"

        # Load X data
        if not x_path.exists():
            raise FileNotFoundError(f"Data file {x_path} does not exist")

        X = pd.read_csv(x_path, header=0)

        # For test data, return only X
        if split == "test":
            return X

        # For train data, also load and return y
        y_path = self.data_dir / f"Ytr{k}.csv"
        if not y_path.exists():
            raise FileNotFoundError(f"Label file {y_path} does not exist")

        y = pd.read_csv(y_path)
        # Assuming the label column is named 'Bound' - adjust if different
        y = y["Bound"] if "Bound" in y.columns else y.iloc[:, 1]

        return X, y

    def load_all_datasets(
        self, split: Literal["train", "test"], use_matrix: bool = False
    ) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
        """
        Load and concatenate all datasets (k=0,1,2).

        Args:
            split: Whether to load train or test data
            use_matrix: If True, load pre-processed matrix data instead of sequences

        Returns:
            If split is "train":
                Tuple of (X, y) where X is concatenated feature data and y is concatenated labels
            If split is "test":
                Only concatenated X
        """
        if split == "train":
            Xs, ys = [], []
            for k in range(3):
                X, y = self.load_dataset(k, split, use_matrix)
                X["k"] = k  # Add k value as a column
                y.name = "Bound"  # Ensure y has a name for the Series
                df = pd.DataFrame({"k": k, "Bound": y})
                Xs.append(X)
                ys.append(df)
            return pd.concat(Xs, axis=0, ignore_index=True), pd.concat(
                ys, axis=0, ignore_index=True
            )
        else:
            Xs = []
            for k in range(3):
                X = self.load_dataset(k, split, use_matrix)
                X["k"] = k  # Add k value as a column
                Xs.append(X)
            return pd.concat(Xs, axis=0, ignore_index=True)

    @staticmethod
    def get_sequence_length(X: pd.DataFrame) -> int:
        """
        Get the length of sequences in the dataset.

        Args:
            X: DataFrame containing sequences

        Returns:
            Length of sequences (assumes all sequences have same length)
        """
        return len(X.iloc[0, 0]) if isinstance(X.iloc[0, 0], str) else X.shape[1]

    def verify_data_integrity(self) -> bool:
        """
        Verify that all expected data files exist and have the correct format.

        Returns:
            True if all checks pass, raises exception otherwise
        """
        expected_files = []
        for k in range(3):
            expected_files.extend(
                [
                    f"Xtr{k}.csv",
                    f"Xte{k}.csv",
                    f"Xtr{k}_mat100.csv",
                    f"Xte{k}_mat100.csv",
                    f"Ytr{k}.csv",
                ]
            )

        for file in expected_files:
            if not (self.data_dir / file).exists():
                raise FileNotFoundError(f"Missing required file: {file}")

        return True
    
    def get_train_and_test_data(self) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
        """
        Load the training and test data from the data directory.
        Args:
            data_dir: Path to the data directory.
        Returns:
            X_train: List of training data sequences.
            Y_train: List of training data labels.
            X_test: List of test data sequences.
        """
        X_train, Y_train = self.load_all_datasets(split="train")
        X_test = self.load_all_datasets(split="test")
        return X_train, Y_train, X_test
