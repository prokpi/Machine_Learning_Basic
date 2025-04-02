# Water Quality Prediction (Machine Learning Basic)

## Project Structure

```
Machine_Learning/
│
├── README.md                    # Project description 
├── data/                        
│   └── water_quality.csv        # The dataset
├── notebook/                   
│   └── water_quality_analysis.ipynb   # The code that was ran in Google Colab
└── requirements.txt             # List of dependencies (for installing via pip)
```

## Project Overview
This project analyzes a **water quality dataset** to predict **potability** (whether water is safe to drink). The dataset contains **3,276 entries** with 10 variables, including:
  - **pH:** pH value of water.
  - **Hardness:** Hardness of water.
  - **Solids:** Solids in ppm.
  - **Chloramines:** Amount of chloramines in ppm.
  - **Sulfates:** Sulfates in mg/L.
  - **Conductivity:** Conductivity of water in μS/cm.
  - **Organic_carbon:** Amount of organic carbon in ppm.
  - **Trihalomethanes:** Amount of trihalomethanes in μg/L.
  - **Turbidity:** Measure of water cloudiness in NTU.
  - **Target Variable:** Potability (0 = Not Potable, 1 = Potable)

## Objectives
- Perform data preprocessing and exploratory data analysis (EDA)
- Train machine learning models using **Logistic Regression** and **Random Forest** to predict water potability
- Evaluate model performance and discuss findings

## Dataset
The dataset is sourced from **Water Potability Dataset** on Kaggle: [Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability). It contains missing values, which required preprocessing before training models.

## Methodology
1. **Data Preprocessing**
   - Handled missing values using [Imputation Strategy]
   - Scaled numerical features using [Scaling Method]
2. **Exploratory Data Analysis (EDA)**
   - Visualized feature distributions and correlations
   - Checked class imbalance in potability
3. **Model Selection & Training**
   - Tried models:
     - Logistic Regression
     - Random Forest
     - [Any other models used]
   - Tuned hyperparameters using GridSearchCV
4. **Evaluation**
   - Assessed models using Accuracy, Precision, Recall, and F1-score

## Results
[Summarize model performance, key observations, and potential issues]

- Best model: **[Insert Best Model]**
- Accuracy: **[Insert Accuracy]**
- Key challenges:
  - **Imbalanced data?** (If yes, mention any handling techniques used)
  - **Model overfitting/underfitting?**
  - **Feature importance insights?**

## Challenges & Future Improvements
- Improve data preprocessing (e.g., advanced imputation techniques)
- Try feature engineering to extract more meaningful variables
- Experiment with ensemble methods or deep learning models
- Address class imbalance using techniques like SMOTE

## How to Run the Code
1. Clone this repository:
   ```bash
   git clone https://github.com/prokpi/water_quality_prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python water_try.py
   ```

## Technologies and Tools
  - **Programming Language:** Python
  - **Libraries:**
    - Data Handling: pandas, numpy
    - Data Visualization: matplotlib, seaborn
    - Machine Learning: scikit-learn
    - Metrics: classification_report, confusion_matrix, roc_auc_score

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib/Seaborn (for visualization)

## Acknowledgments
- **Dataset:** [Water Potability Dataset on Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

## Contact
For questions, contact Kornelija Prokopimaite at korneli.prokopimaite@studio.unibo.it.




