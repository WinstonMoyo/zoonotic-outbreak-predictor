import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model1 = joblib.load('model1.pkl')
model2 = joblib.load('model2.pkl')
model3 = joblib.load('model3.pkl')

X_test_1 = pd.read_csv('Model 1 Dataset Prepared.csv')
X_test_2 = pd.read_csv('Model 2 Dataset Prepared.csv')
X_test_3 = pd.read_csv('Model 3 Dataset Prepared.csv')
y_test = pd.read_csv('Ensemble Output Values.csv').values.ravel()

min_len = min(len(X_test_1), len(X_test_2), len(X_test_3))

X_test_1 = X_test_1.iloc[:min_len].reset_index(drop=True)
X_test_2 = X_test_2.iloc[:min_len].reset_index(drop=True)
X_test_3 = X_test_3.iloc[:min_len].reset_index(drop=True)

y_pred1 = model1.predict_proba(X_test_1)[:, 1]
y_pred2 = model2.predict_proba(X_test_2)[:, 1]
y_pred3 = model3.predict_proba(X_test_3)[:, 1]

y_pred_ensemble_prob = (y_pred1 + y_pred2 + y_pred3) / 3
y_pred_ensemble = (y_pred_ensemble_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_ensemble)
precision = precision_score(y_test, y_pred_ensemble)
conf_matrix = confusion_matrix(y_test, y_pred_ensemble)

print(f"Ensemble Accuracy: {accuracy:.4f}")
print(f"Ensemble Precision: {precision:.4f}")

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Ensemble Model')
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(y_pred_ensemble_prob, label='Predicted Probability', marker='o')
plt.plot(y_test, label='True Label', linestyle='--')
plt.title('Ensemble Model Predictions vs True Labels')
plt.xlabel('Sample Index')
plt.ylabel('Probability / Class')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
