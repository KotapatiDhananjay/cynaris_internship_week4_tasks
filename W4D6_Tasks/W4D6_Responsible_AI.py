from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

cols = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income"
]

# Load Adult dataset
data_dir = Path(
    r"C:\Users\hp\AppData\Local\Programs\Python\Python312"
    r"\Lib\site-packages\aif360\data\raw\adult"
)

train_file = data_dir / "adult.data"
test_file = data_dir / "adult.test"

if not train_file.exists() or not test_file.exists():
    raise FileNotFoundError(
        "Place adult.data and adult.test in "
        "W4D6_Tasks/data/adult and update data_dir accordingly."
    )

df1 = pd.read_csv(train_file, names=cols, skipinitialspace=True, na_values="?")
df2 = pd.read_csv(
    test_file, names=cols, skipinitialspace=True,
    na_values="?", comment="|", skiprows=1
)
df = pd.concat([df1, df2], ignore_index=True)

for c in df.select_dtypes(include="object"):
    df[c] = df[c].str.strip()

df["income"] = df["income"].str.rstrip(".").map({"<=50K": 0, ">50K": 1})
df["sex"] = df["sex"].map({"Female": 0, "Male": 1})
df = df.dropna().copy()

df = df.drop(columns=["fnlwgt"])
df = pd.get_dummies(
    df, columns=df.select_dtypes(include="object").columns,
    dtype=float
)
df = df.astype(float)

# AIF360 dataset
dataset = BinaryLabelDataset(
    df=df, label_names=["income"],
    protected_attribute_names=["sex"],
    favorable_label=1.0, unfavorable_label=0.0
)

X = df.drop(columns="income")
y = df["income"].astype(int)

# Remove protected attribute from model inputs
X = X.drop(columns="sex")

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("Baseline accuracy:", accuracy_score(y_test, pred))
print("Baseline F1:", f1_score(y_test, pred))
print("Baseline ROC-AUC:", roc_auc_score(y_test, prob))

# Build aligned AIF360 test dataset and predictions
test_df = df.loc[idx_test].copy()
test_ds = BinaryLabelDataset(
    df=test_df, label_names=["income"],
    protected_attribute_names=["sex"],
    favorable_label=1.0, unfavorable_label=0.0
)

pred_df = test_df.copy()
pred_df["income"] = pred
pred_ds = BinaryLabelDataset(
    df=pred_df, label_names=["income"],
    protected_attribute_names=["sex"],
    favorable_label=1.0, unfavorable_label=0.0
)

def fairness(ds, pds):
    m = ClassificationMetric(
        ds, pds,
        unprivileged_groups=[{"sex": 0.0}],
        privileged_groups=[{"sex": 1.0}]
    )
    return {
        "Disparate Impact": m.disparate_impact(),
        "Statistical Parity Difference": m.statistical_parity_difference(),
        "Equal Opportunity Difference": m.equal_opportunity_difference(),
        "Average Odds Difference": m.average_odds_difference()
    }

before = fairness(test_ds, pred_ds)
print("Before mitigation:", before)

# Reweigh training data
train_ds = BinaryLabelDataset(
    df=df.loc[idx_train], label_names=["income"],
    protected_attribute_names=["sex"],
    favorable_label=1.0, unfavorable_label=0.0
)
rw = Reweighing(
    unprivileged_groups=[{"sex": 0.0}],
    privileged_groups=[{"sex": 1.0}]
)
weighted_ds = rw.fit_transform(train_ds)

Xw = pd.DataFrame(
    weighted_ds.features, columns=weighted_ds.feature_names
).drop(columns="sex")
yw = weighted_ds.labels.ravel().astype(int)

model_rw = LogisticRegression(max_iter=500)
model_rw.fit(Xw, yw, sample_weight=weighted_ds.instance_weights)

pred_rw = model_rw.predict(X_test)
pred_df["income"] = pred_rw
pred_rw_ds = BinaryLabelDataset(
    df=pred_df, label_names=["income"],
    protected_attribute_names=["sex"],
    favorable_label=1.0, unfavorable_label=0.0
)

after = fairness(test_ds, pred_rw_ds)
print("After mitigation:", after)

pd.DataFrame([before, after], index=["Before", "After"]).to_csv(
    OUT / "fairness_comparison.csv"
)

pd.DataFrame([
    {"Model": "Baseline", "Accuracy": accuracy_score(y_test, pred),
     "F1": f1_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, prob)},
    {"Model": "Reweighed", "Accuracy": accuracy_score(y_test, pred_rw),
     "F1": f1_score(y_test, pred_rw),
     "ROC-AUC": roc_auc_score(y_test, model_rw.predict_proba(X_test)[:, 1])}
]).to_csv(OUT / "model_performance.csv", index=False)

# SHAP summary plot
explainer = shap.LinearExplainer(model, X_train)
values = explainer.shap_values(X_test.iloc[:200])
shap.summary_plot(values, X_test.iloc[:200], show=False, max_display=12)
plt.savefig(OUT / "shap_summary.png", bbox_inches="tight")
plt.close()

# Simple model card PDF
styles = getSampleStyleSheet()
pdf = SimpleDocTemplate(str(OUT / "W4D6_Model_Card.pdf"))
pdf.build([
    Paragraph("Responsible AI Model Card", styles["Title"]),
    Spacer(1, 12),
    Paragraph("Model: Logistic Regression on the UCI Adult Income dataset.",
              styles["Normal"]),
    Paragraph("Purpose: Educational income classification and fairness audit.",
              styles["Normal"]),
    Paragraph(f"Baseline accuracy: {accuracy_score(y_test, pred):.3f}",
              styles["Normal"]),
    Paragraph(f"Baseline fairness metrics: {before}", styles["Normal"]),
    Paragraph(f"After reweighing: {after}", styles["Normal"]),
    Paragraph(
        "Limitations: Historical US data may encode social bias. "
        "Removing gender does not remove proxy bias. This model is not "
        "intended for real-world eligibility or employment decisions.",
        styles["Normal"]
    )
])

(BASE / "W4D6_India_Bias_Case.md").write_text(
    "# India AI Bias Case: Facial Recognition\n\n"
    "Facial recognition systems can perform unevenly across demographic "
    "groups when training data is not representative. In India, differences "
    "in skin tone, age, gender, and image conditions may affect accuracy. "
    "Incorrect matches can cause privacy violations, wrongful suspicion, "
    "or unnecessary police questioning. Overall accuracy alone cannot show "
    "whether a system treats groups fairly. Developers should evaluate "
    "false-positive and false-negative rates across relevant groups using "
    "representative Indian data. High-impact decisions should include "
    "human review, a meaningful appeal process, strict limits on data "
    "retention, and independent audits. Facial recognition should not be "
    "treated as conclusive evidence. Responsible AI requires transparency, "
    "accountability, privacy protection, and safeguards against "
    "disproportionate harm.",
    encoding="utf-8"
)

print("Completed. Outputs saved in:", OUT)
