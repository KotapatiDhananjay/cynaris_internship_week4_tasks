
"""
W4D4: FastAPI Model Serving Endpoint
Train a classification model and serve predictions through an API.
"""

import os
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "breast_cancer_model.joblib")

app = FastAPI(
    title="W4D4 Model Serving API",
    description="Breast cancer classification model serving endpoint",
    version="1.0.0",
)

# Load the saved model, or train and save it on first startup.
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000)),
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"Validation accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Validation ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    return model


model = load_or_train_model()
FEATURE_COUNT = 30


class PredictionRequest(BaseModel):
    # Supply exactly 30 numeric features from the Breast Cancer dataset.
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 numerical input features",
    )


class PredictionResponse(BaseModel):
    prediction: int
    diagnosis: str
    probability_malignant: float
    probability_benign: float


@app.get("/")
def home():
    return {
        "message": "Model Serving API is running",
        "status": "healthy",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    values = np.asarray(request.features, dtype=float)

    if not np.all(np.isfinite(values)):
        raise HTTPException(
            status_code=422,
            detail="Features must contain only finite numeric values",
        )

    probabilities = model.predict_proba(values.reshape(1, -1))[0]
    prediction = int(model.predict(values.reshape(1, -1))[0])

    # Dataset class 0 = malignant; class 1 = benign.
    diagnosis = "malignant" if prediction == 0 else "benign"

    return PredictionResponse(
        prediction=prediction,
        diagnosis=diagnosis,
        probability_malignant=float(probabilities[0]),
        probability_benign=float(probabilities[1]),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
