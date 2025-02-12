import numpy as np
from typing import Tuple, List

def create_k_folds(X: np.ndarray, 
                   y: np.ndarray, 
                   n_splits: int = 5, 
                   shuffle: bool = True, 
                   random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create K-fold cross-validation splits from data.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
        n_splits (int): Number of folds (default: 5)
        shuffle (bool): Whether to shuffle the data before splitting (default: True)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        List[Tuple]: List of (X_train_fold, X_val_fold, y_train_fold, y_val_fold) for each fold
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get total number of samples
    n_samples = len(X)
    
    # Create array of indices
    indices = X.index.to_numpy()
    
    # Shuffle indices if requested
    if shuffle:
        np.random.shuffle(indices)
    
    # Calculate fold size
    fold_size = n_samples // n_splits
    remainder = n_samples % n_splits
    
    # List to store all fold combinations
    fold_data = []
    
    # Create folds
    start = 0
    for fold in range(n_splits):
        # Calculate end index for current fold
        # Add one extra sample to some folds if data size isn't perfectly divisible
        end = start + fold_size + (1 if fold < remainder else 0)
        
        # Get validation indices for current fold
        val_indices = indices[start:end]
        
        # Get training indices (all indices except validation)
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        # Split the data for this fold
        X_train_fold = X.loc[train_indices]
        X_val_fold = X.loc[val_indices]
        y_train_fold = y.loc[train_indices, "Bound"].to_numpy()
        y_val_fold = y.loc[val_indices, "Bound"].to_numpy()
        
        # Add the fold data to our list
        fold_data.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))
        
        # Update start index for next fold
        start = end
    
    return fold_data


def compute_accuracy(y_pred, y_true):
    """
    Calculate accuracy using the formula: (N - sum(|pred - true|)) / N
    where N is the number of predictions
    
    Args:
        y_pred: Predicted values (numpy array)
        y_true: True values (numpy array)
    
    Returns:
        float: Accuracy score between 0 and 1
    """
    import numpy as np
    
    # Number of predictions
    N = len(y_pred)
    
    # Calculate absolute differences between predictions and true values
    absolute_differences = np.abs(y_pred - y_true).sum()
    
    # Calculate accuracy
    accuracy = (N - absolute_differences) / N
    
    return accuracy
