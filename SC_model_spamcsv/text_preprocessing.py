from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class TextPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def save(self, path="text_vectorizer.pkl"):
        joblib.dump(self.vectorizer, path)

    def load(self, path="text_vectorizer.pkl"):
        self.vectorizer = joblib.load(path)
