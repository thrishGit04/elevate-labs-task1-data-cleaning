import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load raw CSV
RAW_CSV_PATH = "tit-dataset.csv"   # change if needed
df = pd.read_csv(RAW_CSV_PATH)

# -----------------------------
# 1. Feature Engineering
# -----------------------------

# Create HasCabin (1 or 0)
if 'Cabin' in df.columns:
    df['HasCabin'] = df['Cabin'].notnull().astype(int)
    df.drop(columns=['Cabin'], inplace=True)

# Extract Title from Name
if 'Name' in df.columns:
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')[0]

    # Combine common titles
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace(['Mme'], 'Mrs')

    # Replace rare titles
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# -----------------------------
# 2. Handle Missing Values
# -----------------------------

# Fill missing Embarked with mode
if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Fill missing Fare with median
if 'Fare' in df.columns:
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Fix Age using Title-wise median (correct version)
if 'Age' in df.columns:
    df['Age'] = df.groupby('Title')['Age'].transform(lambda s: s.fillna(s.median()))
    df['Age'].fillna(df['Age'].median(), inplace=True)

# -----------------------------
# 3. Drop Unnecessary Columns
# -----------------------------

for col in ['Ticket', 'Name']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# -----------------------------
# 4. Encode Categorical Columns
# -----------------------------

# Encode Male/Female to 0/1
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)

# One-hot encode Title + Embarked
for col in ['Embarked', 'Title']:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# -----------------------------
# 5. Final NA Handling
# -----------------------------

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in [np.int64, np.float64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# -----------------------------
# 6. Scale Numeric Columns
# -----------------------------

scaler = StandardScaler()

for col in ['Age', 'Fare']:
    if col in df.columns:
        df[[col]] = scaler.fit_transform(df[[col]])

# -----------------------------
# 7. Save Output
# -----------------------------

df.to_csv("processed_titanic.csv", index=False)
print("✔ Preprocessing complete! Saved as processed_titanic.csv")
