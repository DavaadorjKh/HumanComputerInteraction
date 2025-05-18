from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_preprocess_data
import joblib

# 1. Өгөгдөл унших ба боловсруулалт
X_train, X_test, y_train, y_test = load_and_preprocess_data("spambase.data")

# 2. Logistic Regression загвар үүсгэх
model = LogisticRegression(max_iter=1000)

# 3. Сургалт хийх
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Нарийвчлал: {accuracy:.2f}")

print(classification_report(y_test, y_pred))
joblib.dump(model, 'logistic_regression_model.pkl')
