import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import generate_synthetic_dataset
from predictions import fit_models, fit_linear_model
from evaluation import evaluate_performance

def main():
    # Generate data
    X_train, s_train, y_train, X_test, s_test, y_test = generate_synthetic_dataset()


    # Evaluate for α ∈ [0, 1]
    alphas = np.linspace(0, 1, 11)
    results = []
    for alpha in alphas:
        #f_alpha_pred = fit_models(X_train, s_train, y_train, X_test, s_test, alpha)
        #risk, unfairness = evaluate_performance(f_alpha_pred, y_test, s_test, weight_type="Proportional")

        risk_, unfairness_ = fit_linear_model(X_train, s_train, y_train, X_test, s_test, y_test, alpha, weight_type="Proportional")
        #results.append((alpha, risk, unfairness, risk_, unfairness_))
        results.append((alpha, risk_, unfairness_))


    # Convert to DataFrame
    #results_df = pd.DataFrame(results, columns=["alpha", "risk", "unfairness", "risk_", "unfairness_"])
    results_df = pd.DataFrame(results, columns=["alpha", "risk_", "unfairness_"])
    print(results_df)

#    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#
#    ax1 = axes[0]
#    ax1_twin = ax1.twinx()
#    ax1.plot(results_df["alpha"], results_df["risk"], 'o-', color='green', label="Risk")
#    ax1_twin.plot(results_df["alpha"], results_df["unfairness"], 'o-', color='orange', label="Unfairness")
#    ax1.set_xlabel("α")
#    ax1.set_ylabel("Risk (MSE)", color='green')
#    ax1_twin.set_ylabel("Unfairness", color='orange')
#
#    ax2 = axes[1]
#    ax2_twin = ax2.twinx()
#    ax2.plot(results_df["alpha"], results_df["risk_"], 'o-', color='green', label="Risk")
#    ax2_twin.plot(results_df["alpha"], results_df["unfairness_"], 'o-', color='orange', label="Unfairness")
#    ax2.set_xlabel("α")
#    ax2.set_ylabel("Risk (MSE)", color='green')
#    ax2_twin.set_ylabel("Unfairness", color='orange')
#
#    plt.tight_layout()
#    plt.show()

    fig, ax = plt.subplots(figsize=(10,4))
    ax_twin = ax.twinx()
    ax.plot(results_df["alpha"], results_df["risk_"], 'o-', color='green', label="Risk")
    ax_twin.plot(results_df["alpha"], results_df["unfairness_"], 'o-', color='orange', label="Unfairness")
    ax.set_xlabel("α")
    ax.set_ylabel("Risk (MSE)", color='green')
    ax_twin.set_ylabel("Unfairness", color='orange')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()