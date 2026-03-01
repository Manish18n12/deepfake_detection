import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("deepfake_resnet_model.h5")

img_size = 128

# CHANGE IMAGE PATH HERE
image_path = "woman-face-avatar.jpg"

img = cv2.imread(image_path)
img = cv2.resize(img, (img_size, img_size))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("FAKE IMAGE ❌")
else:
    print("REAL IMAGE ✅")

print("Confidence:", prediction[0][0])