# elevate-labs-task1-data-cleaning

This repository contains Task 1 of my AIML Internship by Elevate Labs.  
The focus of this task is on cleaning, preprocessing, and preparing the Titanic dataset for machine learning models, including training baseline ML models.

## Repository Structure

├── tit-dataset.csv                # Raw dataset (uploaded from Kaggle)  
├── processed_titanic.csv          # Fully cleaned dataset generated after preprocessing  
├── preprocessing.py               # Script version of preprocessing
├── training.py                    # Script version of training  
├── outputs/                       # Folder storing all saved trained models  
│   ├── model_lr.joblib            # Logistic Regression model  
│   ├── model_rf.joblib            # Random Forest model  
│   ├── model_nn.h5                # Neural Network model  
│   └── scaler.joblib              # StandardScaler used during preprocessing  
└── README.md                      # Project documentation  


## 🧹 Task 1 — Data Cleaning & Preprocessing

### ✔ Steps Performed

### 1. Handling Missing Values
- Filled missing **Embarked** with mode  
- Filled missing **Fare** with median  
- Filled missing **Age** using Title-wise median strategy  
  (Mr, Mrs, Miss each get separate medians)

### 2. Feature Engineering
- Extracted **Title** from Name  
- Created new feature: **HasCabin**  
- Encoded **Sex**, **Embarked**, and **Title**

### 3. Removed Irrelevant Columns
- Dropped unnecessary fields:  
  - `Name`  
  - `Ticket`

### 4. One-hot Encoding
Converted categorical columns to numeric:  
- Title  
- Embarked  

### 5. Scaling
Normalized numerical columns using **StandardScaler**:
- Age  
- Fare  

### 6. Final Clean Dataset
Saved as:

``processed_titanic.csv``


## 🤖 BONUS — Baseline Model Training

Although model training is usually Task 2 or 3, baseline ML models were trained:

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | ~81%     |
| Random Forest        | ~84%     |
| Neural Network (Keras) | ~82–85% |

All model files are stored inside the `outputs/` directory.


## 🚀 How to Run This Project

### Preprocessing:
python preprocessing.py

### Training:
python training.py

Run all cells sequentially.


## 📌 Tools & Libraries Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Google Colab  


## ✨ Author
**Thrishool M S**  
Elevate Labs — Task 1
