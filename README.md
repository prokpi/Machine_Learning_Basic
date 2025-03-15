# Machine Learning: Water Quality Prediction

## 1. Project Description

This project aims to predict the **potability** of water based on various physiochemical properties. It involves:

- Data preprocessing
- Exploratory data analysis (EDA)
- Feature engineering
- Model training and evaluation

The goal is to classify water as **potable** (safe to drink) or **not potable**.

## 2. Objectives

- **Analyze** water quality parameters and their relationship with potability.
- **Develop** and **compare** machine learning models for binary classification.
- **Evaluate** model performance using appropriate metrics to identify the most predictive model.

## 3. Dataset

- **Name**: Water Potability Dataset
- **Source**: [Water Potability Dataset on Kaggle](https://www.kaggle.com/datasets)

### Dataset Features:
The dataset contains **10 attributes** related to water quality:

- **pH**: pH value of water
- **Hardness**: Hardness of water
- **Solids**: Solids in ppm
- **Chloramines**: Amount of chloramines in ppm
- **Sulfates**: Sulfates in mg/L
- **Conductivity**: Conductivity of water in μS/cm
- **Organic_carbon**: Amount of organic carbon in ppm
- **Trihalomethanes**: Amount of trihalomethanes in μg/L
- **Turbidity**: Measure of water cloudiness in NTU

#### Target Variable:
- **Potability** (0 = Not Potable, 1 = Potable)

## 4. Technologies and Tools

### Programming Language:
- **Python**

### Libraries:
- **Data Handling**: `pandas`, `numpy`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Metrics**: `classification_report`, `confusion_matrix`, `roc_auc_score`

