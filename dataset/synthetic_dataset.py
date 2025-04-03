import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_dataset(n_samples:int=1000, n_features:int=5, noise:float=.5, n_groups:int=2, sensitive_proportions:list=None) -> tuple[np.array, np.array, np.array, np.array]:
    """Generate a synthetic dataset for a fairness problem

    Args:
        n_samples (int, optional): Number of sample in the dataset. Defaults to 1000.
        n_features (int, optional): Number of features in the dataset. Defaults to 5.
        noise (float, optional): Noise level. Defaults to .5.
        sensitive_proportion (float, optional): Proportion of elements in one of the sensitive group. Keep to .5 for balanced sensitive classes. Defaults to .5.

    Returns:
        tuple[np.array, np.array, np.array, np.array]: _description_
    """

    # Generate synthetic dataset
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)

    # Generate sensitive attribute S with multiple values
    if sensitive_proportions is None:
        sensitive_proportions = [1 / n_groups] * n_groups  # Uniform distribution

    # Add group-dependent bias
    S = np.random.choice(range(n_groups), size=n_samples, p=sensitive_proportions)
    
    # Assign different biases to each group
    group_biases = np.linspace(-1, 1, n_groups)  # Bias values spread between -1 and 1
    bias = np.array([group_biases[s] for s in S])
    
    y += bias

    # Split into train/test
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, S, test_size=0.3)

    print(f"Synthetic data shapes: X_train={X_train.shape}, s_train={s_train.shape}, y_train={y_train.shape}")

    return X_train, s_train, y_train, X_test, s_test, y_test