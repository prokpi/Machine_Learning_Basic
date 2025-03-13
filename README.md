# Machine_Learning

Machine Learning Project on Water Quality Prediction

1. Project description
This project aims to predict the potability of water based on various physiochemical properties. It involves data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation to classify water as potable (safe to drink) or not.

2. Objective:
   - Analyse water quality parameters and their relationship with potability
   - Develop and compare machine learning models for binary classification
   - Evaluate model performance using appropriate metrics to identify the most predictive model
  
3. Dataset
  - Name: Water Potability Dataset
  - Source: Water Potability Dataset on Kaggle
    
  The dataset contains 10 attributes related to water quality:
  - pH: pH value of water.
  - Hardness: Hardness of water.
  - Solids: Solids in ppm.
  - Chloramines: Amount of chloramines in ppm.
  - Sulfates: Sulfates in mg/L.
  - Conductivity: Conductivity of water in μS/cm.
  - Organic_carbon: Amount of organic carbon in ppm.
  - Trihalomethanes: Amount of trihalomethanes in μg/L.
  - Turbidity: Measure of water cloudiness in NTU.
  - Target Variable: Potability (0 = Not Potable, 1 = Potable)

4. Mechnologies and Tools
  - Programming Language: Python
  - Libraries:
  - Data Handling: pandas, numpy
  - Data Visualization: matplotlib, seaborn
  - Machine Learning: scikit-learn
  - Metrics: classification_report, confusion_matrix, roc_auc_score
