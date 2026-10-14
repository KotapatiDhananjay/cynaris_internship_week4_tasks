
"""
W4D3: Model Serialisation — joblib & pickle
Train a model, evaluate it, save it, reload it, and verify predictions.
"""

import os
import time
import pickle
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
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
    PrecisionRecallDisplay,
)
import matplotlib.pyplot as plt

OUTPUT_DIR = "W4D3_Tasks/outputs"
MODEL_DIR = "W4D3_Tasks/models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Build preprocessing and classification pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000)),
    ])

    # Train the model
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    # Evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Training Time (seconds)": training_time,
    }

    print("\n=== Model Evaluation ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"),
        index=False,
    )

    # Save and reload using joblib
    joblib_path = os.path.join(MODEL_DIR, "breast_cancer_model.joblib")
    joblib.dump(model, joblib_path)
    loaded_joblib_model = joblib.load(joblib_path)
    joblib_predictions = loaded_joblib_model.predict(X_test)

    # Save and reload using pickle
    pickle_path = os.path.join(MODEL_DIR, "breast_cancer_model.pkl")
    with open(pickle_path, "wb") as file:
        pickle.dump(model, file)

    with open(pickle_path, "rb") as file:
        loaded_pickle_model = pickle.load(file)

    pickle_predictions = loaded_pickle_model.predict(X_test)

    # Confirm saved models preserve predictions
    print("\n=== Serialisation Verification ===")
    print("Joblib predictions match:",
          (y_pred == joblib_predictions).all())
    print("Pickle predictions match:",
          (y_pred == pickle_predictions).all())

    # Classification report
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
        y_test, y_pred, display_labels=data.target_names, cmap="Blues"
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

    # Precision-recall curve
    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_curve.png"))
    plt.close()

    print(f"\nModel files saved in: {MODEL_DIR}")
    print(f"Metrics and plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
