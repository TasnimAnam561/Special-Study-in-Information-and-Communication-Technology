import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# 1. Phase A: Load & Preprocess Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Phase B: Model Implementation
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM (Support Vector Machine)": SVC(kernel='rbf'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

# 3. Phase C: Evaluation Metrics Function
def evaluate_model(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return acc, prec, rec, f1, spec


# 4. Train, Evaluate & Store Results
results = []
accuracies = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc, prec, rec, f1, spec = evaluate_model(y_test, y_pred)
    accuracies[name] = acc

    results.append({
        "Algorithm": name,
        "Accuracy": f"{acc:.4f}",
        "Precision": f"{prec:.4f}",
        "Recall": f"{rec:.4f}",
        "F1-Score": f"{f1:.4f}",
        "Specificity": f"{spec:.4f}"
    })

# Print Results Table
results_df = pd.DataFrame(results)
print("\n" + "="*85)
print("DISEASE PREDICTION MODEL EVALUATION METRICS")
print("="*85)
print(results_df.to_string(index=False))


# 5. Generate and Save Plot to 'results/' Folder
plt.figure(figsize=(10, 5))
plt.barh(list(accuracies.keys()), list(accuracies.values()), color='skyblue')
plt.xlabel("Accuracy Score")
plt.title("Disease Prediction - Model Accuracy Comparison")
plt.xlim(0.8, 1.0)
plt.tight_layout()

# Saves the figure directly into the results directory for GitHub submission
plt.savefig("results/accuracy_comparison.png")
print("\n[INFO] Accuracy comparison plot saved to 'results/accuracy_comparison.png'")
