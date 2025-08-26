# Loan Approval Classification Project

## 📌 Overview
This project predicts loan approval decisions using machine learning models.  
We analyzed a dataset of **45,000 loan applications with 14 features** (demographics, financials, credit history, etc.) and applied multiple classifiers to build a robust predictive system.

## 🔍 Key Highlights
- Baseline logistic regression achieved **78.04% accuracy**
- Advanced models tested: XGBoost, Random Forest, Gradient Boosting, SVM, Voting, and Stacking
- **XGBoost with preprocessing** achieved the best performance at **94.12% accuracy**
- Preprocessing included: outlier capping, categorical encoding, and SMOTE
- Ensembles provided more stable predictions across folds, though only marginal improvements over XGBoost

## 📂 Files
- `loan_approval_classification.ipynb` → Notebook with full analysis, modeling, and evaluation  
- `group_project_report.pdf` → Detailed project report with methodology, results, and recommendations  
- `loan_data.csv` → Dataset containing **45,000 anonymized loan applications with 14 features**  

### Dataset Features
- **Demographics**: Age, Gender, Education, Employment Experience, Home Ownership  
- **Financials**: Income, Loan Amount, Loan Interest Rate, Loan-to-Income Ratio  
- **Credit History**: Credit Score, Credit History Length, Previous Loan Defaults  
- **Target Variable**: `loan_status` → 1 = Approved, 0 = Rejected  

---

# 📝 Project Details & Inferences

## What I Did
- Performed **EDA** on the dataset and identified heavy right-skew and extreme outliers (e.g., age 144, income > $7M).  
- Applied **preprocessing**:
  - Outlier capping using IQR filtering  
  - Combined sparse categories (e.g., "Master" + "Doctorate")  
  - Encoded categorical features (ordinal and one-hot)  
  - Resampled data using **SMOTE** to handle class imbalance  
- Built, compared, and tuned machine learning models:
  - Logistic Regression (baseline), Random Forest, Gradient Boosting, XGBoost, SVM  
  - Ensemble strategies: Voting and Stacking  
- Conducted **ablation experiments** to measure the contribution of each preprocessing step.  
- Tuned hyperparameters with **GridSearchCV** for XGBoost & Random Forest.  
- Validated fairness across gender and home-ownership demographics.

---

## 📊 Results
- **Benchmark Models**:  
  - Logistic Regression: 78.04% accuracy  
  - Majority-class predictor: 65.36%  

- **Advanced Models (Raw Data)**:  
  - XGBoost: 92.86%  
  - Random Forest: 92.64%  
  - Voting: 92.50%  
  - Stacking: 92.22%  
  - Gradient Boosting: 92.17%  
  - SVM: 91.53%  

- **With Preprocessing**:  
  - XGBoost: 94.12% (best overall)  
  - Random Forest: 93.90%  
  - SVM: 92.85%  

- **Ablation Insights**:  
  - SMOTE had the largest effect (+0.94 pp for XGBoost).  
  - Outlier capping contributed +0.59 pp.  
  - Encoding improved results by +0.26 pp.  
  - All steps combined gave the best gains:contentReference[oaicite:0]{index=0}.  

---

## 📌 Inferences & Insights

### Data & Features
- **Previous defaults, loan-to-income ratio, income, loan amount, and credit score** were the strongest predictors of approval.  
- Outliers heavily distorted raw data and needed treatment for stable results.  

### Modeling Insights
- **Tree-based ensembles (XGBoost, Random Forest)** consistently outperformed linear models and SVMs.  
- **Ensemble learning** (Voting & Stacking) only marginally improved over XGBoost but provided more stability across folds.  
- **Iterative experimentation** proved that no single preprocessing step was sufficient — combining SMOTE, capping, and encoding yielded the best improvements.  

### Business & Strategic Takeaways
1. **Risk-Based Pricing & Approval**  
   Use model probabilities to classify applicants into low/medium/high risk tiers for differentiated interest rates or collateral requirements.  

2. **Proactive Risk Intervention**  
   Flag medium-risk applications for manual review to reduce false negatives.  

3. **Portfolio Monitoring**  
   Retrain models quarterly to capture evolving applicant behavior and changing macroeconomic conditions.  

4. **Feature Enrichment**  
   Consider alternative data (utility payments, social media signals) for applicants with thin credit files to boost recall.  

### Ethical & Compliance
- **Fairness audit** showed precision and recall remained within ±2% across gender and home-ownership groups, indicating no major bias.  
- Continued monitoring recommended to ensure compliance and equity.  

---

## 🚀 Deployment Notes
- Final XGBoost model size: **~6 MB**.  
- **Inference speed**: <20 ms per application (real-time capable).  
- Deployment options:  
  - REST API with Flask/FastAPI  
  - Cloud ML platforms like **AWS SageMaker**, **Azure ML**, or **GCP AI Platform**.  

---

## 👨‍💻 Team Members
- Samira Saleh  
- Nirmit Pradip Patel  
- Gaurang Damjibhai Vora  
- Javier Arguelles Badillo  
- Thriksha Giriraju  
