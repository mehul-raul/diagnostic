import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt
import joblib
# Load data
df = pd.read_csv("U:\\aiml\\DISEASEPREDEC\\Frontend\\data\\lung_cancer_survey.csv")


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['LUNG_CANCER']=encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER']=encoder.fit_transform(df['GENDER'])
df.head()
# Split features and target
X=df.drop(['LUNG_CANCER'],axis=1)
y=df['LUNG_CANCER']
print(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("actual:", y_test.values)
print(f"Predictions: {y_pred}")
print("Accuracy:", accuracy_score(y_test, y_pred))
sample = [[1, 66, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2]]
prediction = model.predict(sample)
print("Prediction for sample row:", prediction)
# joblib.dump(model, "U:\\aiml\\DISEASEPREDEC\\Frontend\\models\\lung_trained.pkl")