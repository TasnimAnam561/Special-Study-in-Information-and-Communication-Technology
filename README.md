# Special-Study-in-Information-and-Communication-Technology
Assignments


## Project Overview
This project implements machine learning classification models to predict disease outcomes using real-world medical data. The objective is to preprocess patient metrics, train multiple algorithms, and evaluate their comparative performance using standardized evaluation metrics.

---

## Dataset Description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** `scikit-learn.datasets`
- **Instances:** 569 samples
- **Features:** 30 numeric, real-valued features computed from digitized images of fine needle aspirate (FNA) of breast masses (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry).
- **Target Variable:** Binary Classification
  - `0`: Malignant (Disease Present)
  - `1`: Benign (Disease Absent)

---

## Data Preprocessing
1. **Train-Test Split:** The dataset was split into an **80% training set** and a **20% testing set** (`random_state=42`) to ensure reproducible evaluation on unseen data.
2. **Feature Scaling:** Feature values were standardized using `StandardScaler` ($\mu = 0, \sigma = 1$). Scaling ensures distance-sensitive models (e.g., SVM, K-NN, Logistic Regression) perform optimally without feature dominance.

---

## Machine Learning Algorithms
The following classification algorithms were implemented and compared:
- **Logistic Regression:** Probabilistic linear model used as a baseline.
- **Support Vector Machine (SVM):** Non-linear classification using the Radial Basis Function (RBF) kernel.
- **Decision Tree:** Intuitive rule-based model for binary classification.
- **Random Forest:** Ensemble of decision trees to minimize variance and prevent overfitting.
- **K-Nearest Neighbors (KNN):** Distance-based instance learning ($k=5$).
- **Naive Bayes:** Probabilistic model based on Bayes' theorem with feature independence assumption.

---

## Evaluation Metrics & Results
Models were evaluated on the test set using standard binary classification metrics:
- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** True Positives relative to total predicted positives.
- **Recall (Sensitivity):** Ability of the model to correctly identify positive disease cases.
- **F1-Score:** Harmonic mean of Precision and Recall.
- **Specificity:** Ability of the model to correctly identify negative (benign) cases.

### Model Performance Summary
Visual comparisons of model accuracies are stored in the `results/` directory as `accuracy_comparison.png`.

---

## Repository Structure
