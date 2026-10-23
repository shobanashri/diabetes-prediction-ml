# Diabetes Prediction Using Machine Learning

An educational machine learning project that uses a Support Vector Machine (SVM) classifier to predict diabetes outcomes from patient measurements. The project includes data analysis, preprocessing, model evaluation, and an interactive Gradio application.

> **Disclaimer:** This project is for educational purposes only. Its predictions are not medical diagnoses and must not be used to make healthcare decisions.

## Project Overview

The goal is to explore how supervised machine learning can classify diabetes outcomes using numerical health-related features.

### Objectives

* Understand and analyze the dataset.
* Explore relationships between input features and the target variable.
* Handle missing or invalid measurements.
* Train an SVM classification model.
* Evaluate the model on unseen test data.
* Build an interactive application for experimental predictions.

## Dataset

The dataset contains patient-related measurements and a binary target variable named `Outcome`.

| Feature                  | Description                                 |
| ------------------------ | ------------------------------------------- |
| Pregnancies              | Number of pregnancies                       |
| Glucose                  | Glucose measurement                         |
| BloodPressure            | Blood pressure measurement                  |
| SkinThickness            | Skin thickness measurement                  |
| Insulin                  | Insulin measurement                         |
| BMI                      | Body mass index                             |
| DiabetesPedigreeFunction | Dataset-provided diabetes pedigree function |
| Age                      | Age                                         |
| Outcome                  | Target class: 0 or 1                        |

Verify the dataset's source and license before redistributing it with this repository.

## Technologies Used

* Python
* Pandas and NumPy
* Matplotlib and Seaborn
* Scikit-learn
* Support Vector Machine (SVM)
* Gradio
* Joblib

## Project Workflow

1. Data loading and understanding
2. Exploratory data analysis (EDA)
3. Data preprocessing
4. Train-test split
5. Feature scaling and SVM model training
6. Model evaluation
7. Saving the trained pipeline
8. Interactive predictions using Gradio

Preprocessing should be fitted on the training data only to avoid data leakage.

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/shobanashri/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Start the application

```bash
python app.py
```

Open the local URL displayed in the terminal.

Ensure that the trained model file is available at the path expected by `app.py`.

## Model Evaluation

The model should be evaluated using held-out test data. Relevant metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* ROC-AUC, when calculated appropriately

Actual evaluation results should be added after verifying them against the final notebook. Accuracy alone does not establish clinical usefulness.

## Limitations

* The dataset has a limited number of records and may not represent all populations.
* Model performance depends on data quality and preprocessing.
* Predictions are experimental and may be unreliable for individual patients.
* The model has not been validated for clinical use.

## Future Improvements

* Improve input validation in the application.
* Investigate model calibration and class-specific performance.
* Improve the user interface.
* Evaluate generalization using suitable validation techniques.

## Author

**Shobana Shri**

GitHub: [@shobanashri](https://github.com/shobanashri)
