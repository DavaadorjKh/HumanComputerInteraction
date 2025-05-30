import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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

# Train-test хуваалт
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model сургах
model = MultinomialNB()
model.fit(X_train, y_train)

# Модел хадгалах
joblib.dump(model, "spam_model_NB.pkl")

# Үнэлгээ
y_pred = model.predict(X_test)
print("Нарийвчлал:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
