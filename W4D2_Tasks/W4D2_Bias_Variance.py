
"""
W4D2: Bias-Variance Tradeoff & Regularisation
Compare underfitting, overfitting, Ridge, and Lasso regression.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

OUTPUT_DIR = "W4D2_Tasks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Load regression dataset
    dataset = load_diabetes(as_frame=True)
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Polynomial Degree 10": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=10, include_bias=False)),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=1.0, max_iter=10000)),
        ]),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Train RMSE": mean_squared_error(
                y_train, train_pred
            ) ** 0.5,
            "Test RMSE": mean_squared_error(
                y_test, test_pred
            ) ** 0.5,
            "Train R2": r2_score(y_train, train_pred),
            "Test R2": r2_score(y_test, test_pred),
        })

    results_df = pd.DataFrame(results)
    print("\n=== Bias-Variance Model Comparison ===")
    print(results_df.round(4).to_string(index=False))
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False
    )

    # Plot train vs test RMSE
    results_df.set_index("Model")[["Train RMSE", "Test RMSE"]].plot(
        kind="bar", figsize=(10, 6)
    )
    plt.ylabel("RMSE (lower is better)")
    plt.title("Bias-Variance and Regularisation Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plt.close()

    # Learning curve for Ridge regression
    ridge_model = models["Ridge Regression"]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    sizes, train_scores, validation_scores = learning_curve(
        ridge_model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_jobs=-1,
    )

    train_rmse = -train_scores.mean(axis=1)
    validation_rmse = -validation_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_rmse, marker="o", label="Training RMSE")
    plt.plot(sizes, validation_rmse, marker="o", label="Validation RMSE")
    plt.xlabel("Training Samples")
    plt.ylabel("RMSE (lower is better)")
    plt.title("Ridge Regression Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"))
    plt.close()

    print("\nInterpretation:")
    print("- High train and test errors may indicate underfitting.")
    print("- Low train error with much higher test error may indicate overfitting.")
    print("- Ridge and Lasso regularisation penalise large coefficients.")
    print("- Compare both train and test metrics before selecting a model.")
    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
