# train_breast_cancer_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
file_path = r'U:\aiml\DISEASEPREDEC\Frontend\data\breast-cancer.csv'
# Load dataset
df = pd.read_csv(file_path)  # Replace if you have your own dataset

# Drop ID or irrelevant columns (modify as needed)
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

# Encode target label
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Features and labels
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds)}")


joblib.dump(model, "U:\\aiml\\DISEASEPREDEC\\Frontend\\models\\breast_cancer.sav")
print("Model saved to models/breast_cancer.sav")
