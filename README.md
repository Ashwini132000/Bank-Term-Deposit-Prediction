# 📈 Predicting Bank Term Deposit Subscriptions Using Machine Learning Techniques  

A retail bank has conducted a series of direct marketing campaigns via phone calls to promote term deposit subscriptions. The bank wants to use machine learning to **predict whether a customer will subscribe to a term deposit** based on demographic, financial, and campaign-related attributes.  

![Screenshot 2025-06-05 145744](https://github.com/user-attachments/assets/a482082c-05f2-4539-9194-66802c52f508)

---

## 🎯 Objective  

To develop a classification model that accurately predicts the **likelihood of a customer subscribing to a term deposit** using historical campaign data. This can help the bank:  
1. Improve targeting for future campaigns.
2. Reduce marketing costs.
3. Increase campaign success rates.

--- 

## 🗃️ Dataset 

![Screenshot 2025-05-28 203949](https://github.com/user-attachments/assets/91e7f177-d6a5-49bc-a62b-e9f0de6adaab)  

---  

## 🔧 Tools & Libraries

This project was built using the following Python tools and libraries:

- **Python** – Programming Language
- **NumPy, Pandas** – Data manipulation and analysis
- **Matplotlib, Seaborn, Plotly** – Data visualization
- **Scikit-learn** – Machine Learning models and utilities
- **XGBoost** – Advanced gradient boosting framework
- **Imbalanced-learn (SMOTE)** – Handling class imbalance
- **Warnings** – To suppress unnecessary warnings

---

## 📚 Project Workflow  

### 1️⃣ Import Necessary libraries  
```
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")  
```

---

### 2️⃣ Data Loading and Preview  

![Screenshot 2025-07-06 234351](https://github.com/user-attachments/assets/bf561b4c-08a3-47df-bda8-2400dacc1776)  

---

### 3️⃣ Data Overview & Cleaning  

**🔹Dataset Summary**  
Dataset contains both categorical (object) and numerical (int64) features.  

**🔹Missing Values Check**  
No missing/null values found in the dataset. 

**🔹Feature Classification**
- Categorical Features: [`'job'`, `'marital'`, `'education'`, `'default'`, `'housing'`, `'loan'`, 
 `'contact'`, `'month'`, `'poutcome'`]
- Numerical Features: [`'age'`, `'balance'`, `'day'`, `'duration'`, `'campaign'`, `'pdays'`, `'previous'`]

---

### 4️⃣ Exploratory Data Analysis (EDA)    

**🔹Univariate Analysis:**  

![Univariate_Cat](https://github.com/user-attachments/assets/572d178d-a14f-4777-80b2-70752ab2ebfe)  

![Univariate_Num](https://github.com/user-attachments/assets/0e82cb0b-4cd9-42c2-87b7-9d366e79e9b1)

**🔹Bivariate Analysis:**

![Bivariate_Cat](https://github.com/user-attachments/assets/53ec4ed0-e71f-405d-9c8a-9de75cd31e33)  

![Bivariate_Num](https://github.com/user-attachments/assets/ca459123-1423-4c5e-976e-44e9fcc24bc4)

**🔹Multivariate Analysis:**  

![Pairplot](https://github.com/user-attachments/assets/2dd39959-33b6-4884-945c-182d44adfeb2)

**🔹Correlation Matrix:**  

![Correlation_Matrix](https://github.com/user-attachments/assets/212b86cb-e0df-48cd-a809-77182ced6ecf)  

---

### 5️⃣ Outlier Detection    

![Outlier](https://github.com/user-attachments/assets/cbd5262a-58d8-4dfc-9c2b-d89f3c9e7b88)  

---

### 6️⃣ Feature Scaling
**🔹Binary Label Encoding**  
Converted binary categorical variables (`default`, `housing`, and `loan`) into numerical format using `LabelEncoder`.  

**🔹Target Variable Encoding**  
The target column `y` was encoded as:  
- `'yes'` → 1
- `'no'` → 0
  
**🔹Ordinal Encoding**  
The `month` column was ordinally encoded to reflect the natural calendar sequence.

**🔹One-Hot Encoding**  
Used `pd.get_dummies()` for nominal features like `job`, `marital`, `education`, `contact`, and `poutcome`.
`drop_first=True` was used to avoid multicollinearity. 

---

### 7️⃣ Train-Test Split  

**🔹Splitting the Dataset**  
- We separate features and the target variable.
- The dataset is then split into 80% training and 20% testing data using stratified sampling to maintain class distribution.  

**🔹Checking Class Imbalance**  
- We analyze the target variable to identify any class imbalance.  
- `Class 0`: 88.30%
- `Class 1`: 11.70%
   
**🔹Handling Imbalance with SMOTE**  
- To balance the training set, we apply SMOTE (Synthetic Minority Oversampling Technique).
  
**🔹Feature Scaling**
- We use StandardScaler to scale the features so that the model treats all features equally.

---

### 8️⃣ Model Building  
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- XGBoost Classifier
- Gradient Boosting Classifier

---  

**📊 Model Performance Comparison**  

| Model                        | Train Accuracy | Test Accuracy | Train AUC | Test AUC | Precision | Recall | F1-Score |
|-----------------------------|----------------|---------------|-----------|----------|-----------|--------|----------|
| Logistic Regression         | 0.9149         | 0.8812        | 0.9711    | 0.8705   | 0.49      | 0.54   | 0.51     |
| K-Nearest Neighbors (KNN)   | 1.0000         | 0.8905        | 1.0000    | 0.8833   | 0.54      | 0.41   | 0.47     |
| Decision Tree               | 0.9154         | 0.8746        | 0.9779    | 0.8783   | 0.47      | 0.61   | 0.53     |
| Random Forest               | 1.0000         | 0.8964        | 1.0000    | 0.9163   | 0.57      | 0.48   | 0.53     |
| Support Vector Machine (SVM)| 0.9358         | 0.8941        | 0.9835    | 0.8711   | 0.56      | 0.45   | 0.50     |
| Naive Bayes                 | 0.8264         | 0.8106        | 0.9030    | 0.7995   | 0.33      | 0.58   | 0.42     |
| XGBoost Classifier          | 0.9713         | 0.9024        | 0.9968    | 0.9268   | 0.59      | 0.53   | 0.56     |
| Gradient Boosting Classifier| 0.9678         | 0.9025        | 0.9961    | 0.9279   | 0.59      | 0.57   | **0.57** |

---  

## 🔚 Conclusion

The goal of this project was to build a robust classification model to predict whether a client will subscribe to a term deposit, based on their demographic and campaign-related features.

### 📌 Summary of Key Steps & Findings:

- **Data Quality:**  
  - The dataset was clean with no missing values.  
  - Outliers were handled using clipping and log transformation.  
  - Numerical features were standardized using **StandardScaler**.

- **Class Imbalance Handling:**  
  - Applied **SMOTE** to balance the minority class and improve model sensitivity to term deposit subscriptions.

- **Model Selection & Performance:**  
  - Among various models, the **Gradient Boosting Classifier** performed best:
    - **F1-Score:** 0.57 (balanced precision and recall)  
    - **Test Accuracy:** 90.25%  
    - **AUC-ROC:** 0.9279 (excellent class separation)

- **Most Influential Features:**  
  - `duration` (call duration) – **43%**  
  - `poutcome` (previous campaign outcome) – **14%**  
  - `housing` (loan status) – **10%**  
  - `previous` (number of contacts) – moderate impact

### 🎯 Final Outcome:
The final model shows strong predictive power and can be reliably used to identify potential clients who are likely to subscribe to a term deposit. It balances class performance through SMOTE and achieves high AUC, making it suitable for targeted marketing or customer retention strategies.

---  

## 🧑‍💻 Author

**Ashwini Bawankar**  
*Data Science Intern | Passionate about Machine Learning*

---

## 📬 Contact

📧 Email: [abawankar13@gmail.com]  
🔗 LinkedIn: [https://www.linkedin.com/in/ashwini-bawankar/]  


