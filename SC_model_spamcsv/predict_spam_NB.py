import joblib
from text_preprocessing import TextPreprocessor

# Ачаалж байна
model = joblib.load("spam_model_NB.pkl")
preprocessor = TextPreprocessor()
preprocessor.load()  # vectorizer-г ачаалж байна

# Хэрэглэгчээс текст
user_input = input("Шалгах текстээ оруулна уу: ")

# Векторчлох
vec = preprocessor.transform([user_input])

# Таамаглал
pred = model.predict(vec)[0]
print("⚠️ SPAM байна." if pred == "spam" else "✅ SPAM биш.")
