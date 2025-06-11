import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("U:\\aiml\\DISEASEPREDEC\\Frontend\\data\\Covid.csv")

le = LabelEncoder()

# Encode all categorical columns (symptoms + target)
for col in df.columns:
    if df[col].dtype == 'object':  # Target "COVID-19" and symptom columns
        df[col] = le.fit_transform(df[col])


X = df.drop("COVID-19", axis=1)  # Symptoms
y = df["COVID-19"]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# print("actual:", y_test.values)
# print(f"Predictions: {y_pred}")
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

joblib.dump(model, "U:\\aiml\\DISEASEPREDEC\\Frontend\\models\\covid19_trained.pkl")