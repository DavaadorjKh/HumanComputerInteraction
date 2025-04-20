from data_preprocessing import load_and_preprocess_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

X_train, X_test, y_train, y_test = load_and_preprocess_data("spambase.data")

# Label-ийг one-hot болгох
encoder = LabelEncoder()
y_train_encoded = to_categorical(encoder.fit_transform(y_train))
y_test_encoded = to_categorical(encoder.transform(y_test))

# ANN model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

model.save("spam_ann_model.h5")

loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Нарийвчлал: {accuracy:.2f}")
