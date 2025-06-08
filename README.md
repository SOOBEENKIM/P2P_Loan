# 💳 P2P Loan Default Prediction

This project analyzes P2P lending data to predict whether a loan will be approved or defaulted using advanced classification techniques.

### 🎯 Objective
To accurately predict the default status of P2P loans using various sampling and modeling strategies that address data imbalance and nonlinearity in the financial domain.

### 📂 Dataset
- Real-world P2P lending data with ~13% positive class (imbalanced).
- Preprocessing included:
  - Removing or imputing missing values
  - Outlier detection (e.g., revol_util > 1 removed)
  - Label and ordinal encoding (e.g., `sub_grade`, `emp_length`)

## ⚙️ Modeling & Results Summary

### 🔍 Feature Engineering
- Label encoding for `emp_length`, `verification_status`
- Ordinal encoding for `sub_grade`, `purpose`, and `home_ownership`
- Outliers removed based on domain logic (e.g., revol_util capped at 1.0)

### 🧪 Sampling Methods
To address the data imbalance, we used:
- **Under Sampling**
- **SMOTE (Synthetic Minority Over-sampling Technique)**
- **ADASYN (Adaptive Synthetic Sampling)**

### 🧠 Models Trained
- Soft SVM (linear & RBF kernel)
- Entropy Fuzzy SVM (linear & RBF)
- Decision Tree (benchmark)

Hyperparameter tuning was conducted using Grid Search.

---

### 📈 Evaluation Metric
Main metric: **F1-Score**  
→ Chosen for its balanced consideration of precision and recall, suitable for imbalanced classification.

### 🚀 Key Results
- Models trained with **RBF kernel + SMOTE** showed better generalization
- Ensemble soft voting improved F1-score across multiple configurations
- **Best F1 Score**: 0.3248 using Soft Voting with RBF kernel + Under Sampling

---

> 📌 Note: A single sampling and training instance may not fully reflect generalizability; future improvements should include multiple sampling iterations and Bayesian hyperparameter tuning (e.g., Optuna).

## 🔍 Key Technical Feedback

This summary highlights the core insights and feedback from model evaluation during the P2P loan project:

### ✅ Why Ensemble Learning Worked Better
- **Model Independence Matters**: When base learners are statistically independent, ensemble models significantly outperform single models.
- **Voting Strategy**: Odd-number ensemble voting with different models showed strong results.
- **Random Forest Limitation**: Trees in a single model share the same structure; diversity in base models (heterogeneous ensemble) performs better.
- **Ensemble over Tuning**: Using ensemble learning was more effective than extensive hyperparameter tuning (especially for EFSVM).

---

### ⚠️ Limitations of EFSVM
- **Highly Sensitive to Hyperparameters**: Requires precise tuning.
- **Statistical Assumption Risk**: Relies heavily on past vs. future data distribution being similar.
- **Sampling Mismatch**: When test and train sets are sampled independently, differing distributions may reduce performance.

---

### 🧠 Practical Considerations
- **Voting on Test Sets**:
  - Soft voting can be ambiguous when outputs like [0, 0, 1] occur.
  - Use `sklearn` voting modules with caution; inspect parameters or implement your own for better control.
- **Feature Distribution**:
  - Even after using a scaler, imbalanced feature distances across samples caused uneven data distribution.
  - Data was not clearly separable even after feature engineering.

---

> 💡 *In future services, determine the ensemble level based on required response time and system constraints.*
