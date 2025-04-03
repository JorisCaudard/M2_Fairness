import numpy as np

def evaluate_performance(y_pred:np.array, y_true:np.array, S_test:np.array, weight_type):
    
    # Unique groups and their counts
    groups = np.unique(S_test)
    n_groups = len(groups)

    assert weight_type in ["Equals", "Proportional", "Inverse"], "Weight types is not supported" 
    if weight_type == "Equals":
        weights = np.array([1/n_groups for s in groups])
    elif weight_type == "Proportional":
        weights = np.array([np.mean(S_test==s) for s in groups])
    elif weight_type == "Inverse":
        weights = np.array([1/np.mean(S_test==s) for s in groups])

    # Compute Risk (MSE)
    risk = np.mean((y_true - y_pred) ** 2)
    
    # Compute unfairness (Wasserstein-2 variance of group-wise predictions)
    group_preds = [y_pred[S_test == s] for s in groups]
    group_means = [np.mean(preds) for preds in group_preds]
    
    # Wasserstein-2 distance between each group and the barycenter (weighted mean)
    barycenter = np.sum(weights * group_means)
    unfairness = np.sum(weights * (group_means - barycenter) ** 2)  # Simplified for 1D (equiv. to W₂²)
    
    return risk, unfairness