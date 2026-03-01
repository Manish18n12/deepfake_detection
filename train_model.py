import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# ===============================
# SETTINGS
# ===============================

data_dir = "dataset"   # <-- make sure this matches your folder name
img_size = 128
epochs = 20   # You can increase to 15 or 20 later

# ===============================
# LOAD DATA
# ===============================

categories = ["real", "fake"]

data = []
labels = []

for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (img_size, img_size))
            data.append(img_array)
            labels.append(label)
        except:
            pass

data = np.array(data) / 255.0
labels = np.array(labels)

print("Total images loaded:", len(data))

# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# ===============================
# TRANSFER LEARNING MODEL (ResNet50)
# ===============================

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(img_size, img_size, 3)
)

# Freeze base model layers
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# ===============================
# COMPILE MODEL
# ===============================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# TRAIN MODEL
# ===============================

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

# ===============================
# SAVE MODEL
# ===============================

model.save("deepfake_resnet_model.h5")

print("ResNet model training completed ✅")