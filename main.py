from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import requests

app = FastAPI()

# 🔗 Google Drive model link
MODEL_URL = "https://drive.google.com/uc?export=download&id=1IsSaOlu8JeXmczd2c8qXo2a1TU3kJHC3"
MODEL_PATH = "model.h5"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# ✅ FIX: compile=False lagaya
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

classes = ["Acne", "Eczema", "Psoriasis", "Normal"]

def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = preprocess_image(image)
    pred = model.predict(img)

    class_index = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return {
        "prediction": classes[class_index],
        "confidence": confidence
    }
