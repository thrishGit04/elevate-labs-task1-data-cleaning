#!/usr/bin/env python3
"""
training.py

Usage:
    python training.py
This script expects a `processed_titanic.csv` in the current working directory.
It trains:
  - Logistic Regression
  - Random Forest
  - Simple Keras Neural Network """


import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------
# Config / Paths
# ----------------------
RAW_PATH = "processed_titanic.csv"   # input (already preprocessed)
OUT_DIR = "outputs"
RANDOM_STATE = 42
NN_EPOCHS = 20
NN_BATCH_SIZE = 32

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------
# Helper functions
# ----------------------
def evaluate_print(name, y_true, y_pred):
    """Print basic classification metrics (precision, recall, f1 explained)."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}   (precision = how many selected items are relevant)")
    print(f"Recall: {rec:.4f}      (recall = how many relevant items are selected)")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm.tolist()}

# ----------------------
# Load data
# ----------------------
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"{RAW_PATH} not found. Run preprocessing first or place the file here.")

df = pd.read_csv(RAW_PATH)
print("Loaded", RAW_PATH, "shape =", df.shape)

# Drop PassengerId (not a feature)
if "PassengerId" in df.columns:
    df = df.drop(columns=["PassengerId"])

# Ensure label exists
if "Survived" not in df.columns:
    raise ValueError("Label column 'Survived' not found in the processed dataset.")

X = df.drop(columns=["Survived"])
y = df["Survived"]

# Ensure all features are numeric; raise if any non-numeric found
for col in X.columns:
    try:
        X[col] = pd.to_numeric(X[col], errors='raise')
    except Exception as e:
        raise ValueError(f"Feature column '{col}' is not numeric. Convert it before training. Error: {e}")

# ----------------------
# Train/Val/Test split
# ----------------------
# We do: Train 70%, Val 15%, Test 15% (stratify keeps class ratio similar) 
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.17647, random_state=RANDOM_STATE, stratify=y_temp
)  # 0.17647*0.85 ~= 0.15 overall

print("Splits -> train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

# ----------------------
# Scaling numeric features
# ----------------------
# If preprocessing already scaled Age/Fare, this still safely fits a new scaler to the training split.
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
print("Saved scaler to", os.path.join(OUT_DIR, "scaler.joblib"))

# Convert to numpy arrays for Keras (NN) and sklearn
X_train_np = X_train.values.astype("float32")
X_val_np = X_val.values.astype("float32")
X_test_np = X_test.values.astype("float32")
y_train_np = y_train.values
y_val_np = y_val.values
y_test_np = y_test.values

# ----------------------
# Train classical models (scikit-learn)
# ----------------------
test_summary = {}

# Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_np, y_train_np)
val_pred_lr = lr.predict(X_val_np)
test_summary['lr_val'] = evaluate_print("Logistic Regression (val)", y_val_np, val_pred_lr)
joblib.dump(lr, os.path.join(OUT_DIR, "model_lr.joblib"))
print("Saved Logistic Regression ->", os.path.join(OUT_DIR, "model_lr.joblib"))

# Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_np, y_train_np)
val_pred_rf = rf.predict(X_val_np)
test_summary['rf_val'] = evaluate_print("Random Forest (val)", y_val_np, val_pred_rf)
joblib.dump(rf, os.path.join(OUT_DIR, "model_rf.joblib"))
print("Saved Random Forest ->", os.path.join(OUT_DIR, "model_rf.joblib"))

# ----------------------
# Train small Neural Network (Keras)
# ----------------------
print("\nTraining Neural Network (Keras)...")
input_dim = X_train_np.shape[1]
model = keras.Sequential(
    [
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks: early stopping to avoid overfitting (patience = how many epochs to wait)
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

history = model.fit(
    X_train_np,
    y_train_np,
    validation_data=(X_val_np, y_val_np),
    epochs=NN_EPOCHS,
    batch_size=NN_BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)

val_prob_nn = model.predict(X_val_np).ravel()
val_pred_nn = (val_prob_nn >= 0.5).astype(int)
test_summary['nn_val'] = evaluate_print("Neural Network (val)", y_val_np, val_pred_nn)

model_path = os.path.join(OUT_DIR, "model_nn.h5")
model.save(model_path)
print("Saved Neural Network ->", model_path)

# ----------------------
# Final evaluation on TEST set
# ----------------------
print("\nFinal evaluation on TEST set:")
results_test = {}
# LR test
lr_test_pred = lr.predict(X_test_np)
results_test['lr_test'] = evaluate_print("Logistic Regression (test)", y_test_np, lr_test_pred)

# RF test
rf_test_pred = rf.predict(X_test_np)
results_test['rf_test'] = evaluate_print("Random Forest (test)", y_test_np, rf_test_pred)

# NN test
nn_test_prob = model.predict(X_test_np).ravel()
nn_test_pred = (nn_test_prob >= 0.5).astype(int)
results_test['nn_test'] = evaluate_print("Neural Network (test)", y_test_np, nn_test_pred)

# Save test summary (dictionary of metrics)
import json
summary = {"val": test_summary, "test": results_test}
with open(os.path.join(OUT_DIR, "test_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

print("\nSaved test_summary.json to", os.path.join(OUT_DIR, "test_summary.json"))
print("\nAll models and artifacts saved to", OUT_DIR)
