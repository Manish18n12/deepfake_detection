from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime

app = Flask(__name__)

model = load_model("deepfake_resnet_model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

img_size = 128


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)[0][0]

            if prediction > 0.5:
                result = f"FAKE IMAGE ❌ ({prediction:.2f})"
            else:
                result = f"REAL IMAGE ✅ ({prediction:.2f})"

            image_path = filepath

    return render_template("index.html",
                           result=result,
                           image_path=image_path)


@app.route("/gallery")
def gallery():
    images = os.listdir(app.config["UPLOAD_FOLDER"])
    images = images[::-1]  # latest first
    return render_template("gallery.html", images=images)


if __name__ == "__main__":
    app.run(debug=True)