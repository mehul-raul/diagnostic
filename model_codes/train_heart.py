import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load data
heart_df = pd.read_csv("U:\\aiml\\DISEASEPREDEC\\Frontend\\data\\heart-disease.csv")
X = heart_df.drop("target", axis=1)
y = heart_df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Create pipeline that includes scaler + model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=6,
        class_weight='balanced',
        random_state=42,
        max_features='sqrt'
    ))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Accuracy test
y_pred = pipeline.predict(X_test)
print("Heart Accuracy:", accuracy_score(y_test, y_pred))

# ✅ Save the full pipeline (not just model)
joblib.dump(pipeline, "U:\\aiml\\DISEASEPREDEC\\Frontend\\models\\heart_disease_trained.sav")

# joblib.dump(model, "U:\\aiml\\DISEASEPREDEC\\Frontend\\models\\heart_trained.pkl")