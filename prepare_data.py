import os
import cv2
import numpy as np

data_dir = "dataset"   # make sure this matches your folder name
categories = ["real", "fake"]

img_size = 128
data = []
labels = []

for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (img_size, img_size))
            data.append(img_array)
            labels.append(label)
        except:
            pass

data = np.array(data) / 255.0
labels = np.array(labels)

print("Total images loaded:", len(data))
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)