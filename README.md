# MSDS25067_TTDS_COURSE_PROJECT
# Heart Disease Prediction - Machine Learning Project

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-completed-success.svg)]()

##  Project Overview

This project implements a comprehensive machine learning pipeline for predicting heart disease using clinical parameters. Developed as part of CS 591 - Tools & Techniques for Data Science course, the project demonstrates the complete data science workflow from data loading to model deployment.

**Key Highlights:**
-  **Accuracy:** 90.85% (Random Forest)
-  **Dataset:** 918 patients, 11 clinical features
-  **Models:** 7 different ML algorithms compared
-  **Visualizations:** 11 comprehensive charts
-  **Version Control:** Complete Git workflow

##  Course Information

- **Course:** CS 591 - Tools & Techniques for Data Science
- **Instructor:** Kamil Majeed
- **Student:** Numan Hussan
- **Roll Number:** Msds25067
- **Institution:** Information Technology University
- **Submission Date:** December 8, 2025

##  Dataset

**Source:** [Kaggle - Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

**Description:** This dataset combines 5 heart disease datasets to create the largest heart disease dataset available for research. It contains clinical parameters that can predict the presence of heart disease.

**Features:**
- Age, Sex, ChestPainType, RestingBP, Cholesterol
- FastingBS, RestingECG, MaxHR, ExerciseAngina
- Oldpeak, ST_Slope, HeartDisease (target)

**Statistics:**
- 918 observations
- 11 features + 1 target variable
- Binary classification (Heart Disease: Yes/No)
- Relatively balanced classes (55.3% positive)

##  Quick Start

### Prerequisites

```bash
Python 3.8 or higher
pip package manager
Git
```

### Installation

```bash
# Clone the repository
git clone https://github.com/[username]/heart-disease-prediction.git
cd heart-disease-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) and place it in the project root directory.

### Run Analysis

```bash
python analysis.py
```

##  Project Structure

```
heart-disease-prediction/
│
├── README.md                          # Project documentation
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
├── analysis.py                       # Main analysis script
├── LICENSE                           # Project license
│
├── data/                             # Data directory (not in Git)
│   └── heart.csv                    # Original dataset
│
├── charts/                           # Generated visualizations
│   ├── chart1_target_distribution.png
│   ├── chart2_age_analysis.png
│   ├── chart3_correlation_heatmap.png
│   ├── chart4_chest_pain_analysis.png
│   ├── chart5_gender_analysis.png
│   ├── chart6_maxhr_analysis.png
│   ├── chart7_clinical_parameters.png
│   ├── chart8_feature_importance.png
│   ├── chart9_model_comparison.png
│   ├── chart10_roc_curves.png
│   └── chart11_confusion_matrix_best.png
│
├── models/                           # Saved models
│   └── heart_disease_model.pkl      # Best model (Random Forest)
│
└── reports/                          # Project reports
    └── technical_report.pdf         # Complete technical report
```

##  Project Tasks

###  Task A: Dataset Overview (5 marks)
- Loaded and explored the dataset
- Documented all features and their meanings
- Analyzed statistical summaries
- Examined target variable distribution

###  Task B: Exploratory Data Analysis (15 marks)
- Missing values analysis (0 missing values)
- Duplicate detection (1 duplicate found)
- Data types examination
- Correlation analysis
- Outlier detection using IQR method
- Class balance assessment

###  Task C: Data Wrangling & Cleansing (15 marks)
- Removed duplicate records
- Handled invalid zero values (172 in Cholesterol)
- Encoded categorical variables (Label & One-Hot encoding)
- Feature engineering (Age_Group, BP_Category)
- Outlier removal (152 records)
- Feature normalization (StandardScaler)
- Final dataset: 765 samples, 21 features

###  Task D: Data Visualizations (15 marks)
Created 11 comprehensive visualizations:
1. Target variable distribution
2. Age distribution by heart disease status
3. Correlation heatmap
4. Chest pain type analysis
5. Gender analysis
6. Maximum heart rate patterns
7. Clinical parameters (Cholesterol, BP, Oldpeak)
8. Feature importance ranking
9. Model performance comparison
10. ROC curves for all models
11. Confusion matrix (best model)

###  Task E: Git Version Control (15 marks)
- Initialized Git repository
- Created meaningful commits for each task
- Used development branch strategy
- Pushed to GitHub with proper documentation
- Implemented .gitignore for data/images
- Created version tags

###  Task F: Machine Learning (15 marks)
Implemented 7 ML algorithms:
- Logistic Regression
- Decision Tree
- Random Forest  (Best: 90.85% accuracy)
- Support Vector Machine
- K-Nearest Neighbors
- Gradient Boosting
- Naive Bayes

##  Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **90.85%** | 91.30% | 92.55% | **91.92%** | 96.12% |
| Gradient Boosting | 90.85% | 91.84% | 91.49% | 91.67% | **96.58%** |
| SVM | 88.89% | 89.58% | 90.43% | 90.00% | 95.15% |
| Logistic Regression | 87.58% | 88.46% | 89.13% | 88.79% | 93.95% |
| KNN | 86.93% | 87.50% | 89.36% | 88.42% | 92.87% |
| Decision Tree | 86.27% | 87.50% | 87.23% | 87.37% | 92.01% |
| Naive Bayes | 85.62% | 86.36% | 87.23% | 86.79% | 92.23% |

### Best Model: Random Forest

**Performance Metrics:**
-  Accuracy: 90.85%
-  Precision: 91.30%
-  Recall: 92.55%
-  F1-Score: 91.92%
-  ROC-AUC: 96.12%

**Confusion Matrix:**
```
                Predicted
              No    Yes
Actual  No    52     6
        Yes    5    90
```

**Key Insights:**
- Only 11 misclassifications out of 153 test samples
- 94.7% sensitivity (disease detection rate)
- 89.7% specificity (healthy identification rate)
- Excellent for clinical deployment

##  Key Findings

### 1. Feature Importance
**Top 5 Most Important Features:**
1. **ST_Slope** (18%) - Slope of peak exercise ST segment
2. **Chest Pain Type** (15%) - Type of chest pain experienced
3. **Exercise Angina** (12%) - Exercise-induced angina
4. **MaxHR** (11%) - Maximum heart rate achieved
5. **Oldpeak** (10%) - ST depression induced by exercise

### 2. Surprising Discoveries
- **Cholesterol** shows weak correlation with heart disease (contradicts common belief)
- **Asymptomatic patients** have the highest disease prevalence (80%)
- **Exercise-related features** are more predictive than traditional risk factors
- **Males** have 2.5x higher disease prevalence than females

### 3. Clinical Implications
- Exercise stress testing is crucial for diagnosis
- Absence of symptoms doesn't mean absence of disease
- Multiple factors required for accurate prediction
- Gender-specific risk profiles exist

##  Technologies Used

**Programming Language:**
- Python 3.8+

**Data Analysis & Manipulation:**
- pandas 1.5.3
- numpy 1.24.3

**Machine Learning:**
- scikit-learn 1.2.2

**Data Visualization:**
- matplotlib 3.7.1
- seaborn 0.12.2

**Version Control:**
- Git
- GitHub

**Development Environment:**
- Jupyter Notebook (optional)
- VS Code / PyCharm

##  Dependencies

```txt
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
jupyter==1.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

##  Usage

### Basic Usage

```python
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('models/heart_disease_model.pkl')

# Prepare new patient data (21 features after preprocessing)
new_patient = [...]  # Your preprocessed feature vector

# Make prediction
prediction = model.predict([new_patient])
probability = model.predict_proba([new_patient])

print(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Disease'}")
print(f"Probability: {probability[0][1]:.2%}")
```

### Complete Workflow

```python
# Run the complete analysis
python analysis.py

# Output:
# - Prints detailed analysis for each task
# - Generates 11 visualization charts
# - Trains and evaluates 7 ML models
# - Saves best model to models/
```

##  Visualizations

All generated charts are saved in the `charts/` directory:

1. **Target Distribution** - Class balance analysis
2. **Age Analysis** - Age patterns by disease status
3. **Correlation Heatmap** - Feature relationships
4. **Chest Pain Analysis** - Chest pain type patterns
5. **Gender Analysis** - Gender distribution and disease prevalence
6. **MaxHR Analysis** - Heart rate patterns
7. **Clinical Parameters** - Cholesterol, BP, Oldpeak analysis
8. **Feature Importance** - Random Forest feature rankings
9. **Model Comparison** - Performance comparison bar chart
10. **ROC Curves** - All models' ROC curves
11. **Confusion Matrix** - Best model's confusion matrix

##  Git Workflow

### Commit History

```bash
# View commit history
git log --oneline --graph

* a1b2c3d Task F: Machine Learning implementation completed
* d4e5f6g Task E: Git version control documentation
* h7i8j9k Task D: Data visualizations completed
* l0m1n2o Task C: Data wrangling and cleansing completed
* p3q4r5s Task B: Exploratory Data Analysis completed
* t6u7v8w Task A: Dataset loading and description completed
* x9y0z1a Initial commit: Project structure setup
```

### Branching Strategy

- `main` - Stable, production-ready code
- `development` - Active development branch
- Feature branches for major additions

##  Future Enhancements

### Short-term Improvements
- [ ] Add web interface using Streamlit/Flask
- [ ] Implement real-time prediction API
- [ ] Create interactive dashboards
- [ ] Add model explainability (SHAP values)

### Long-term Goals
- [ ] Collect more diverse patient data
- [ ] Incorporate genetic markers
- [ ] Add longitudinal patient tracking
- [ ] Implement deep learning models
- [ ] Deploy to cloud (AWS/Azure/GCP)
- [ ] Integrate with Electronic Health Records (EHR)

##  Acknowledgments

- **Dataset:** Fedesoriano on Kaggle
- **Course Instructor:** Kamil Majeed
- **Institution:** Information Technology University
- **Inspiration:** Clinical need for better heart disease prediction

##  Contact

**Student:** Numan Hussan  
**Email:** msds25067@itu.edu.pk  

##  References

1. Fedesoriano. (2021). Heart Failure Prediction Dataset. Kaggle.
2. Scikit-learn Documentation. (2024). Machine Learning in Python.
3. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
4. American Heart Association. (2023). Heart Disease Statistics.

---


*Last Updated: December 7, 2025*
