import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, header=None)

    # Онцлог шинжүүд болон label ялгах
    X = data.iloc[:, :-1]  # Сүүлийн баганаас бусад нь X
    y = data.iloc[:, -1]   # Сүүлийн багана нь y

    # Сургалт болон тестийн өгөгдөлд хуваах
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
