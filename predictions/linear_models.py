from sklearn.linear_model import LinearRegression

import numpy as np


def fit_models(X_train, s_train, y_train, X_test, s_test, alpha):

    # Compute unconstrained prediction (f1)
    X_train_f1 = np.column_stack([X_train, s_train])
    model_f1 = LinearRegression().fit(X_train_f1, y_train)
    f1_pred_train = model_f1.predict(np.column_stack([X_train, s_train]))
    f1_pred = model_f1.predict(np.column_stack([X_test, s_test]))

    # Compute fair prediction (f0)
    # Get group means from training predictions
    mask0 = (s_train == 0)
    mask1 = (s_train == 1)
    mean0 = np.mean(f1_pred_train[mask0])
    mean1 = np.mean(f1_pred_train[mask1])
    
    # Compute Wasserstein barycenter (weighted average)
    w1 = np.mean(mask1)  # Proportion of group 1
    barycenter = w1 * mean1 + (1-w1) * mean0

    # For test set, apply same group-specific adjustments
    f0_pred = f1_pred.copy()
    f0_pred[s_test == 0] += (barycenter - mean0)
    f0_pred[s_test == 1] += (barycenter - mean1)

    # Compute α-RI oracle predictions (fα)
    f_alpha_pred = np.sqrt(alpha) * f1_pred + (1 - np.sqrt(alpha)) * f0_pred

    return f_alpha_pred

def fit_linear_model(X_train, S_train, y_train, X_test, s_test, y_test, alpha, weight_type):

    # Unique groups ans their counts
    groups = np.unique(S_train)
    n_groups = len(groups)

    # Compute group weights

    assert weight_type in ["Equals", "Proportional", "Inverse"], "Weight types is not supported" 
    if weight_type == "Equals":
        weights = np.array([1/n_groups for s in groups])
    elif weight_type == "Proportional":
        weights = np.array([np.mean(S_train==s) for s in groups])
    elif weight_type == "Inverse":
        weights = np.array([1/np.mean(S_train==s) for s in groups])

    # Fit group specific group model
    beta_hat = np.zeros(X_train.shape[1])
    b_hat = np.zeros(n_groups)

    for s in groups:
        mask = (S_train == s)
        X_s = X_train[mask]
        y_s = y_train[mask]

        # Fit linear regression model
        model = LinearRegression(fit_intercept=True).fit(X_s, y_s)
        beta_hat += model.coef_ * weights[s]
        b_hat[s] = model.intercept_

    # Compute weigthed mean of biases
    b_bar = np.sum(weights * b_hat)

    # Compute predictor
    def predictor(x, s):
        return np.dot(x, beta_hat) + np.sqrt(alpha) * b_hat[s] + (1 - np.sqrt(alpha)) * b_bar
    
    # Compute unfairness
    unfairness = alpha * np.sum(weights * (b_hat - b_bar)**2)

    # Compute risk
    y_pred = np.array([predictor(X_test[i], s_test[i]) for i in range(X_test.shape[0])])
    risk = np.mean((y_test - y_pred)**2)

    return risk, unfairness
