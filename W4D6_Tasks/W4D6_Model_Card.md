
# Model Card: Adult Income Classifier

## Model Details
- Model: Logistic Regression
- Dataset: UCI Adult Income
- Fairness mitigation: AIF360 Reweighing
- Explainability: SHAP LinearExplainer

## Intended Use
Educational research on classification, fairness evaluation, and model explainability. Not intended for actual hiring, lending, or other high-impact decisions.

## Training Data
The UCI Adult dataset contains demographic, employment, and education attributes with an income classification label.

## Factors and Fairness
Gender is audited as a protected attribute. The gender feature is excluded from model inputs, but correlated features may still act as proxies. Reweighing adjusts training sample weights.

## Evaluation Metrics
Accuracy, precision, recall, F1-score, ROC-AUC, disparate impact, statistical parity difference, equal opportunity difference, and average odds difference.

Actual results are recorded in the generated model card PDF and CSV outputs.

## Explainability
SHAP is used to estimate how input features contribute to Logistic Regression predictions. These explanations do not establish causation.

## Limitations
The dataset reflects historical US census data and may contain representation and measurement biases. Results may not generalize to India or other populations. Removing gender alone does not eliminate proxy discrimination.

## Ethical Considerations
Audit subgroup performance, protect sensitive data, document limitations, obtain stakeholder feedback, and require human oversight before any consequential deployment.

## Monitoring
Reassess fairness and performance when data distributions or intended uses change.
