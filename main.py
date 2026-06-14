# backend/main.py
"""
FastAPI Backend for Skin Disease Detection System
Uses Xception model trained on HAM10000 dataset
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import io
import uvicorn
import os
import traceback

app = FastAPI(
    title="DermAI - Skin Disease Detection API",
    description="FastAPI backend for skin disease detection using Xception ML model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DISEASE_CLASSES = {
    0: {"code": "akiec", "name": "Actinic Keratoses & Intraepithelial Carcinoma", "severity": "high"},
    1: {"code": "bcc", "name": "Basal Cell Carcinoma", "severity": "high"},
    2: {"code": "bkl", "name": "Benign Keratosis-like Lesions", "severity": "low"},
    3: {"code": "df", "name": "Dermatofibroma", "severity": "low"},
    4: {"code": "mel", "name": "Melanoma", "severity": "critical"},
    5: {"code": "nv", "name": "Melanocytic Nevi", "severity": "low"},
    6: {"code": "vasc", "name": "Vascular Lesions", "severity": "medium"},
}

CONSULTATION_REQUIRED = {"critical", "high"}

MODEL_PATH = os.getenv("MODEL_PATH", "models/HAM10000_Xception.keras")
model = None
model_error = None


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=1)

    return loss


@app.on_event("startup")
async def load_ml_model():
    global model, model_error

    try:
        print(f"🔍 Checking model path: {MODEL_PATH}")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📁 Models folder exists: {os.path.exists('models')}")
        print(f"📄 Model file exists: {os.path.exists(MODEL_PATH)}")

        if os.path.exists(MODEL_PATH):
            print(f"📦 Model file size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB")

        model = load_model(
            MODEL_PATH,
            custom_objects={
                "focal_loss": focal_loss(),
                "loss": focal_loss(),
            },
            compile=False,
            safe_mode=False,
        )

        model_error = None
        print(f"✅ Model loaded successfully from {MODEL_PATH}")

    except Exception as e:
        model = None
        model_error = str(e)
        print(f"❌ Failed to load model: {e}")
        print("===== FULL MODEL LOAD ERROR =====")
        traceback.print_exc()
        print("=================================")


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    return img_array


def get_confidence_status(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.50:
        return "medium"
    return "low"


@app.get("/")
async def root():
    return {
        "message": "DermAI API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_error": model_error,
        "version": "1.0.0",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"ML model not loaded. Error: {model_error}",
        )

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()

    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size must be under 5MB")

    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        img_array = preprocess_image(img)
        predictions = model.predict(img_array, verbose=0)

        probabilities = predictions[0].tolist()
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        disease_info = DISEASE_CLASSES[predicted_idx]

        all_probs = {
            DISEASE_CLASSES[i]["code"]: float(probabilities[i])
            for i in range(len(DISEASE_CLASSES))
        }

        confidence_status = get_confidence_status(confidence)

        return JSONResponse(
            content={
                "disease_class": disease_info["code"],
                "disease_name": disease_info["name"],
                "confidence": round(confidence, 4),
                "confidence_status": confidence_status,
                "requires_consultation": disease_info["severity"] in CONSULTATION_REQUIRED
                or confidence_status == "low",
                "severity": disease_info["severity"],
                "all_probabilities": all_probs,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/recommendations/{disease_code}")
async def get_recommendations(disease_code: str):
    recommendations = {
        "mel": [
            {
                "title": "Seek Immediate Dermatologist Consultation",
                "description": "Melanoma is a serious form of skin cancer. Immediate medical evaluation is essential.",
                "precautions": "Avoid sun exposure on affected area. Do not attempt self-treatment.",
            },
            {
                "title": "Sun Protection",
                "description": "Use broad-spectrum SPF 50+ sunscreen and protective clothing daily.",
                "precautions": "Apply sunscreen 30 minutes before sun exposure and reapply every 2 hours.",
            },
        ],
        "bcc": [
            {
                "title": "Dermatologist Evaluation Required",
                "description": "Basal cell carcinoma is treatable when caught early. Schedule an appointment promptly.",
                "precautions": "Do not ignore or attempt to treat it yourself.",
            },
            {
                "title": "Protect from UV Radiation",
                "description": "Limit sun exposure, especially between 10am and 4pm.",
                "precautions": "Wear wide-brimmed hats, UV-protective clothing, and sunglasses.",
            },
        ],
        "akiec": [
            {
                "title": "Medical Consultation Recommended",
                "description": "Actinic keratosis can progress to skin cancer. Early treatment is advisable.",
                "precautions": "Avoid further sun damage to the affected area.",
            },
            {
                "title": "Moisturize and Protect",
                "description": "Keep the skin moisturized and protected from harsh elements.",
                "precautions": "Use gentle, fragrance-free moisturizers and high SPF sunscreen.",
            },
        ],
        "nv": [
            {
                "title": "Regular Monitoring",
                "description": "Melanocytic nevi are usually benign but should be monitored for changes.",
                "precautions": "Monitor for changes in size, color, shape, or bleeding using the ABCDE rule.",
            },
            {
                "title": "Annual Skin Check",
                "description": "Schedule annual dermatology visits for professional skin screenings.",
                "precautions": "Photograph the mole periodically to track any changes.",
            },
        ],
        "bkl": [
            {
                "title": "Routine Monitoring",
                "description": "Benign keratosis is non-cancerous but should be monitored.",
                "precautions": "Watch for rapid growth, bleeding, or significant changes in appearance.",
            },
        ],
        "df": [
            {
                "title": "Generally Harmless",
                "description": "Dermatofibromas are benign growths and usually require no treatment.",
                "precautions": "Consult a dermatologist if the growth becomes painful, itchy, or grows rapidly.",
            },
        ],
        "vasc": [
            {
                "title": "Monitor for Changes",
                "description": "Vascular lesions are often benign but should be evaluated.",
                "precautions": "Consult a specialist if lesions bleed spontaneously or grow rapidly.",
            },
        ],
    }

    return JSONResponse(
        content=recommendations.get(
            disease_code,
            [
                {
                    "title": "Consult a Dermatologist",
                    "description": "Please consult a qualified dermatologist for proper evaluation.",
                    "precautions": "This AI result is not a medical diagnosis.",
                }
            ],
        )
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )