import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from text_preprocessing import TextPreprocessor


df = pd.read_csv("spam.csv")
X = df["Message"]
y = df["Category"]

# Preprocessing
preprocessor = TextPreprocessor()
X_vec = preprocessor.fit_transform(X)
preprocessor.save()

# Train-test хуваах
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Модел хадгалах
joblib.dump(model, "spam_model_RF.pkl")

y_pred = model.predict(X_test)
print("Нарийвчлал:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
