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
   - Handled missing values using **Imputation Strategy**
   - Scaled numerical features using **Min-Max Normalization** 
   - Visualized feature distributions and correlations
   - Checked class imbalance in potability
3. **Model Selection & Training**
   - Tried models:
     - Logistic Regression
     - Random Forest
   - Tuned hyperparameters using GridSearchCV
4. **Evaluation**
   - Assessed models using Accuracy, Precision, Recall, and F1-score

## Results

- Best model: **Grid Search with Random Forest**
- Accuracy: **66.31%**
- Key observations:
  - The Random Forest model shows reasonable precision and recall for both classes but struggles with predicting **class 1 (potable)** with lower **recall (33%)**
  - The Grid Search seems to **improve precision and recall slightly for class 0**, but the performance for **class 1** is still lacking in **recall (43%)**
  - **Logistic Regression** has the **lowest accuracy (53.81%)** and also the least effective performance on class 1. The precision for class 1 is lower (43%), and recall for class 0 is decent (65%) but significantly lower than in Random Forest models
 
## Conclusions
- Random Forest seems to be the best-performing model in this scenario, although improvements are needed, especially in recall for class 1 (non-potable water)
- Logistic Regression is underperforming and likely not suitable for this dataset due to assumption of linearity

## Challenges & Future Improvements
- Try feature engineering to extract more meaningful variables
- Experiment with ensemble methods or deep learning models

## How to Run the Code
1. Clone this repository:
   ```bash
   !git clone https://github.com/prokpi/Machine_Learning_Basic.git
)
   ```
2. Navigate to the project folder:
   ```bash
   %cd Machine_Learning_Basic
   ```
3. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
4. Run the notebook:
   ```bash
   In Google Colab, you can open the notebook
   Paste the GitHub URL https://github.com/prokpi/Machine_Learning_Basic and select water_potability_analysis.ipynb
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




