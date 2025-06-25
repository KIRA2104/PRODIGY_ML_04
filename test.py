import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


DATA_DIR = "/Users/gauravtalele/omkar/leapGestRecog"  

IMG_SIZE = 64
MODEL_PATH = "/Users/gauravtalele/omkar/gesture-part4/gesture_model.h5"
ENCODER_PATH = "/Users/gauravtalele/omkar/gesture-part4/label_encoder.pkl"

images = []
labels = []


for subject_folder in os.listdir(DATA_DIR):
    subject_path = os.path.join(DATA_DIR, subject_folder)
    if not os.path.isdir(subject_path):
        continue
    for gesture_folder in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue
        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(gesture_folder)


images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
labels = np.array(labels)


if os.path.exists(ENCODER_PATH):
    le = joblib.load(ENCODER_PATH)
    labels_encoded = le.transform(labels)
else:
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    joblib.dump(le, ENCODER_PATH)
    print("✅ LabelEncoder saved!")

labels_categorical = to_categorical(labels_encoded)


X_train, X_test, y_train, y_test = train_test_split(
    images, labels_categorical, test_size=0.2, random_state=42
)


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if os.path.exists(MODEL_PATH):
    print("📦 Loading saved model...")
    model = load_model(MODEL_PATH)
    history = None  
else:
    print("🧠 Training new model...")
    model = build_model()
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    model.save(MODEL_PATH)
    print("✅ Model saved to disk!")


loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")


if history:
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
