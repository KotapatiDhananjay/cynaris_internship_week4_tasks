
"""
W4D5: 1M Capstone — Sentiment Classifier
Compare Logistic Regression and Random Forest on binary sentiment data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score,
)

OUTPUT_DIR = "W4D5_Tasks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sentiment_data():
    """
    Load IMDb-style text data from:
    W4D5_Tasks/data/pos/
    W4D5_Tasks/data/neg/

    Each folder should contain one text file per review.
    """
    data_dir = "W4D5_Tasks/data"

    if os.path.isdir(os.path.join(data_dir, "pos")) and os.path.isdir(
        os.path.join(data_dir, "neg")
    ):
        dataset = load_files(
            data_dir,
            categories=["neg", "pos"],
            encoding="utf-8",
            decode_error="replace",
            shuffle=True,
            random_state=42,
        )
        # load_files sorts categories alphabetically: neg=0, pos=1
        return dataset.data, dataset.target

    # Fallback: built-in binary dataset, so the project runs immediately.
    # This is a breast cancer classification fallback, NOT sentiment data.
    from sklearn.datasets import load_breast_cancer

    print(
        "WARNING: Sentiment folders not found. Using the built-in "
        "Breast Cancer dataset for a classification demo instead."
    )
    dataset = load_breast_cancer()
    return dataset.data, dataset.target


def main():
    X, y = load_sentiment_data()

    # Use stratification to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    is_text = isinstance(X_train[0], str)

    if is_text:
        models = {
            "Logistic Regression": Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=10000,
                    stop_words="english",
                )),
                ("classifier", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                )),
            ]),
            "Random Forest": Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=10000,
                    stop_words="english",
                )),
                ("classifier", RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                )),
            ]),
        }
        class_names = ["Negative", "Positive"]
    else:
        from sklearn.preprocessing import StandardScaler

        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                )),
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        }
        class_names = ["Class 0", "Class 1"]

    results = []

    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        auc = roc_auc_score(y_test, probabilities)

        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=class_names,
            zero_division=0,
        ))

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "ROC-AUC": auc,
        })

        # Save confusion matrix
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=[0, 1],
            display_labels=class_names,
            cmap="Blues",
        )
        plt.title(f"{name} — Confusion Matrix")
        plt.tight_layout()
        filename = name.lower().replace(" ", "_")
        plt.savefig(os.path.join(
            OUTPUT_DIR, f"{filename}_confusion_matrix.png"
        ))
        plt.close()

        # Save ROC curve
        RocCurveDisplay.from_predictions(
            y_test, probabilities, name=name
        )
        plt.title(f"{name} — ROC Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTPUT_DIR, f"{filename}_roc_curve.png"
        ))
        plt.close()

    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df.round(4).to_string(index=False))
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison.csv"),
        index=False,
    )

    results_df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "ROC-AUC"]
    ].plot(kind="bar", figsize=(10, 6), ylim=(0, 1.05))
    plt.title("Logistic Regression vs Random Forest")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plt.close()

    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
