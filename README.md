# 🩺 Breast Cancer Detection using Support Vector Machine (SVM)

This project applies **Support Vector Machine (SVM)** to predict whether a tumor is **malignant (M)** or **benign (B)** based on diagnostic features. 
We use the Breast Cancer Wisconsin dataset to demonstrate preprocessing, visualization, classification, and model evaluation techniques.

---

## 📊 Dataset Overview

- **File**: `breast-cancer.csv`
- **Target Variable**: `diagnosis`
  - `M`: Malignant
  - `B`: Benign
- **Features**: Radius, texture, perimeter, area, smoothness, etc.
- **Missing Values**: Handled appropriately
- **Source**: UCI Machine Learning Repository

---

## 🧪 Project Objectives

1. Load and clean the dataset.
2. Preprocess and normalize features.
3. Train SVM with both linear and RBF kernels.
4. Tune hyperparameters (`C`, `gamma`) using GridSearchCV.
5. Visualize decision boundaries using PCA.
6. Evaluate model performance using cross-validation.

---

## 🛠️ Tools & Libraries

- `pandas`, `numpy` – Data manipulation
- `scikit-learn` – Modeling and evaluation
- `matplotlib`, `seaborn` – Visualization
- `StandardScaler`, `PCA` – Preprocessing
- `SVC`, `GridSearchCV`, `cross_val_score` – Classification & tuning

📈 Workflow Summary
 1. Data Preprocessing
- Dropped non-informative columns (`id`, `Unnamed: 32`)
- Encoded `diagnosis` into binary (M=1, B=0)
- Scaled features using `StandardScaler`

 2. Model Training
🔹 Linear SVM
```python
model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)
# 3. RBF SVM
model_rbf = SVC(kernel='rbf')
model_rbf.fit(X_train, y_train)

# 4. Evaluation
Used accuracy_score, confusion_matrix, classification_report

Performed 5-fold cross-validation

5. Visualization
Reduced data to 2D using PCA

Plotted decision boundaries for linear and RBF kernels

📌 Results
Model	Accuracy (CV Avg.)
Linear SVM	~97%
RBF SVM	~98%

✅ RBF kernel performed slightly better
✅ PCA visualization confirmed separation between classes

💡 Key Insights
SVM models benefit greatly from feature scaling.
RBF kernel adapts well to non-linear data.
PCA is effective for visualizing high-dimensional decision boundaries.
Cross-validation prevents overfitting and confirms robustness.




