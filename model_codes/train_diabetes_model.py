import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
diab_df = pd.read_csv("U:\\aiml\\DISEASEPREDEC\\Frontend\\data\\diabetes.csv")

Q1 = diab_df.quantile(0.25)
Q3 = diab_df.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = diab_df[~((diab_df < lower_bound) | (diab_df> upper_bound)).any(axis=1)]

X = df_cleaned .drop("Outcome", axis=1)
y = df_cleaned ["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rmodel = RandomForestClassifier(
    n_estimators=800,          # Increased from 200
    max_depth=8,               # Slightly shallower to reduce overfitting
    min_samples_split=6,       # More conservative splitting
    class_weight='balanced',   # Handles class imbalance
    random_state=42,
    max_features='sqrt'        # Better for high-dimensional data
)
rmodel.fit(X_train_res_scaled, y_train_res)
y_ran_pred = rmodel.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_ran_pred):.2f}")
# # Handle zero values (minimum required cleaning)
# zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# df[zero_features] = df[zero_features].replace(0, df.median())

# # Split data
# X = df.drop("Outcome", axis=1)
# y = df["Outcome"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Better Random Forest with optimized settings
# model = RandomForestClassifier(
#     n_estimators=300,          # Increased from 200
#     max_depth=5,               # Slightly shallower to reduce overfitting
#     min_samples_split=6,       # More conservative splitting
#     class_weight='balanced',   # Handles class imbalance
#     random_state=42,
#     max_features='sqrt'        # Better for high-dimensional data
# )
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# # # Save model
# # joblib.dump(model, "U:\\demodiab\\trained_models\\diabetes_trained.pkl")

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow import keras
# diab_df = pd.read_csv("U:\\demodiab\\datasets\\diabetes.csv")

# X = diab_df.drop("Outcome", axis=1)
# y = diab_df["Outcome"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)