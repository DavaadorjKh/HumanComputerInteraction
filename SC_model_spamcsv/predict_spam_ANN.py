import joblib
from text_preprocessing import TextPreprocessor

model = joblib.load("spam_model_ANN.pkl")
preprocessor = TextPreprocessor()
preprocessor.load()  # vectorizer

# Хэрэглэгчээс текст
user_input = input("Шалгах текстээ оруулна уу: ")

# Векторчлох
vec = preprocessor.transform([user_input])

# Таамаглал
pred = model.predict(vec)[0]
print("SPAM байна." if pred == "spam" else "SPAM биш.")
