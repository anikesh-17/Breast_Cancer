# 🧠 Breast Cancer Prediction using Logistic Regression

## 🩺 Understanding the Target Labels: Malignant (0) vs Benign (1)

### 🔴 Malignant (0)
- Cancerous tumor  
- Grows quickly and can spread  
- Requires immediate treatment  

### 🟢 Benign (1)
- Non-cancerous tumor  
- Does not spread  
- Usually less dangerous  

| Label | Meaning      | Type           | Severity |
|-------|--------------|----------------|----------|
| **0** | Malignant    | Cancerous      | 🔴 High  |
| **1** | Benign       | Non-cancerous  | 🟢 Low   |

---

## 📌 Project Overview
This project demonstrates how to:
- Load the Breast Cancer dataset  
- Explore and preprocess data  
- Train a Logistic Regression model  
- Evaluate accuracy  
- Build a predictive system  

---

## 🛠 Dependencies

```python
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## 📥 Data Loading

```python
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data,
                          columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target
```

---

## 🤖 Model Training

```python
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)
```

---

## 📊 Evaluation

Typical Accuracy:
- Training: **~95%**
- Testing: **~93%**

---

## 🟢 Prediction Example

```python
prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("The Breast Cancer is Malignant")
else:
    print("The Breast Cancer is Benign")
```

---

## 🏁 Conclusion
Logistic Regression effectively predicts tumor type using clinical features.
