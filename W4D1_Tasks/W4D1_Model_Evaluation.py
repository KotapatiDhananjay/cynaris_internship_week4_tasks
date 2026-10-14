
"""
W4D1: Model Evaluation Metrics — Precision, Recall, AUC
Covers classification metrics, stratified cross-validation,
overfitting/underfitting diagnosis, and learning curves.
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

OUTPUT_DIR = "W4D1_Tasks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Load binary classification dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Stratified train/test split preserves class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Pipeline prevents scaling leakage across CV folds
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000)),
    ])

    # Train model and measure training time
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    # Predict labels and positive-class probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Training Time (seconds)": training_time,
    }

    print("\n=== Test Set Evaluation ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Save metrics and classification report
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "classification_metrics.csv"),
        index=False,
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=data.target_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "classification_report.csv")
    )

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=data.target_names,
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close()

    # Stratified 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="roc_auc"
    )

    print("\n=== Stratified 5-Fold Cross-Validation ===")
    print("Fold ROC-AUC scores:", cv_scores.round(4))
    print(f"Mean ROC-AUC: {cv_scores.mean():.4f}")
    print(f"Standard deviation: {cv_scores.std():.4f}")

    pd.DataFrame({
        "Fold": range(1, 6),
        "ROC-AUC": cv_scores,
    }).to_csv(os.path.join(OUTPUT_DIR, "cross_validation_scores.csv"),
              index=False)

    # Learning curve: compare training and validation performance
    sizes, train_scores, validation_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_mean, marker="o", label="Training Accuracy")
    plt.plot(sizes, validation_mean, marker="o",
             label="Cross-Validation Accuracy")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"))
    plt.close()

    # Basic diagnosis using final learning-curve scores
    gap = train_mean[-1] - validation_mean[-1]
    print("\n=== Learning Curve Diagnosis ===")
    print(f"Final training accuracy: {train_mean[-1]:.4f}")
    print(f"Final validation accuracy: {validation_mean[-1]:.4f}")

    if train_mean[-1] < 0.80 and validation_mean[-1] < 0.80:
        print("Possible underfitting: both scores are low.")
    elif gap > 0.10:
        print("Possible overfitting: training score exceeds validation score.")
    else:
        print("No strong overfitting signal from the final learning-curve scores.")

    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
