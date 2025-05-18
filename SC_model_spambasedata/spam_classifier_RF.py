
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_preprocess_data
import joblib

X_train, X_test, y_train, y_test = load_and_preprocess_data("spambase.data")

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Нарийвчлал: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'random_forest_model.pkl')