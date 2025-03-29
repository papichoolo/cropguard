from typing import Tuple
import tensorflow as tf
import numpy as np
import json
import logging
import os
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import requests
from pydantic import BaseModel  
from fastapi import Request
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import io
import dotenv
from fastapi import FastAPI, Response
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.units import inch

from fastapi import FastAPI, File, HTTPException, UploadFile


# Limit TensorFlow memory growth - add this BEFORE loading your model
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    # CPU-only usage pattern
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

dotenv.load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Load the trained model and class labels
tf_loaded = False
model = None
CLASS_LABELS = None
analysis = {}

# TensorFlow GPU memory optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic request models
class CropAnalysisRequest(BaseModel):
    class_name: str
    confidence: float
    image_size: list[int]

class StateInput(BaseModel):
    state: str

# Load TensorFlow model dynamically
@app.get("/load-tf/")
async def load_tensorflow():
    global tf_loaded, model, CLASS_LABELS
    if not tf_loaded:
        try:
            model_url = "https://raw.githubusercontent.com/papichoolo/cropguard/master/multiclass_inceptionv3.keras"
            model_path = "multiclass_inceptionv3.keras"
            response = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(response.content)
            model = load_model(model_path)
            with open("class_labels.json", "r") as f:
                CLASS_LABELS = json.load(f)
            tf_loaded = True
            return {"status": "TensorFlow loaded successfully", "model_loaded": True}
        except Exception as e:
            return {"status": "Failed to load TensorFlow", "error": str(e)}
    return {"status": "TensorFlow already loaded", "model_loaded": True}

# Health check endpoint
@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

# Image processing function
def read_file_as_image(data) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(io.BytesIO(data)).convert('RGB')
    img_resized = img.resize((299, 299), resample=Image.BICUBIC)
    return np.array(img_resized), img_resized.size

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    try:
        contents = await file.read()
        image_data, image_size = read_file_as_image(contents)
        img_batch = np.expand_dims(image_data, 0)
        predictions = model.predict(img_batch, verbose=0)
        predicted_class = CLASS_LABELS[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return {"class_name": predicted_class, "confidence": confidence, "image_size": image_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        # Log the full error with traceback
        import traceback
        logging.error(f"Prediction error: {str(e)}")
        logging.error(traceback.format_exc())
        
        # Return appropriate error
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Reset file pointer in case we need to reuse it
        await file.seek(0)  # Raise an HTTPException with the error message
    


@app.post("/analyze/")
async def analyze(request: CropAnalysisRequest):
    global analysis
    try:
        # Construct the prompt using structured input
        prompt = prompt = (
            "You are an agricultural expert providing guidance to farmers. Given the following crop disease detection report, "
            "provide a JSON-formatted response with information about the disease, its common causes, recommended remedies (including both chemical and organic/traditional methods), "
            "and preventative measures. The remedies should prioritize readily available and affordable options suitable for small-scale farming operations. "
            "Consider factors such as monsoon seasons, local climate, and typical farming practices in the region when suggesting remedies. "
            "Structure your response according to the schema below. The language should be accessible and easy to understand for farmers. "
            "Consider including easily available alternatives if one specific medicine is unavailable.\n\n"
            "{\n"
            '  "disease": "Disease Name",\n'
            '  "confidence": "Confidence Score (percentage)",\n'
            '  "causes": [\n'
            '    {\n'
            '      "cause": "Specific cause of the disease",\n'
            '      "description": "Brief explanation of the cause in the context of the local farming environment"\n'
            "    }\n"
            "  ],\n"
            '  "remedies": {\n'
            '    "chemical": [\n'
            '      {\n'
            '        "product": "Recommended chemical product (with active ingredient)",\n'
            '        "dosage": "Recommended dosage and application instructions",\n'
            '        "alternatives": ["Alternative chemical product (with active ingredient)", "Another alternative"]\n'
            "      }\n"
            "    ],\n"
            '    "organic": [\n'
            '      {\n'
            '        "method": "Organic/Traditional Remedy",\n'
            '        "description": "Detailed explanation of how to prepare and apply the remedy, emphasizing readily available ingredients.",\n'
            '        "alternatives": ["Alternative organic/traditional remedy", "Another alternative"]\n'
            "      }\n"
            "    ]\n"
            "  },\n"
            '  "prevention": [\n'
            '    "Preventative Measure 1 (explained in a practical way)",\n'
            '    "Preventative Measure 2 (explained in a practical way)",\n'
            '    "Preventative Measure 3 (explained in a practical way)"\n'
            "  ],\n"
            '  "notes": "Important Considerations for farmers regarding weather conditions, harvesting, or other important aspects."\n'
            "}\n\n"
            "Respond in JSON format ONLY. Do not include any introductory or concluding remarks. Your response must validate against the provided schema.\n\n"
            "Here is a sample crop disease detection report:\n\n"
            "{\n"
            f'  "class": "{request.class_name}",\n'
            f'  "confidence": {request.confidence},\n'
            f'  "image_size": {request.image_size}\n'
            "}\n"
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        
        structured_response = response.text.strip("```json\n").strip("\n```")
        analysis = json.loads(structured_response)
        print(analysis)
        return analysis
        

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return {"error": "Internal Server Error"}
    

@app.get("/generate-report/")
async def generate_report():
    global analysis

    if not analysis:
        return {"error": "No analysis data available. Please generate an analysis first."}

    # Create a BytesIO buffer to store PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    content = []

    # Title
    content.append(Paragraph("<b>Crop Disease Analysis Report</b>", styles["Title"]))

    # Formatting function for text with wrapping
    def add_wrapped_text(label, text):
        if isinstance(text, list):  # Handling lists
            content.append(Paragraph(f"<b>{label}:</b>", styles["Normal"]))
            for item in text:
                if isinstance(item, dict):  # Handling nested dictionaries
                    for sub_key, sub_value in item.items():
                        content.append(Paragraph(f"- <b>{sub_key}:</b> {sub_value}", styles["Normal"]))
                else:
                    content.append(Paragraph(f"- {item}", styles["Normal"]))
        elif isinstance(text, dict):  # Handling remedy dictionaries
            content.append(Paragraph(f"<b>{label}:</b>", styles["Normal"]))
            for sub_key, sub_value in text.items():
                content.append(Paragraph(f"<b>{sub_key.capitalize()}:</b>", styles["Normal"]))
                for item in sub_value:
                    for sub_sub_key, sub_sub_value in item.items():
                        content.append(Paragraph(f"- <b>{sub_sub_key}:</b> {sub_sub_value}", styles["Normal"]))
        else:
            content.append(Paragraph(f"<b>{label}:</b> {text}", styles["Normal"]))

        content.append(Paragraph("<br/>", styles["Normal"]))  # Adds spacing

    # Add all analysis fields to the report
    for key, value in analysis.items():
        add_wrapped_text(key.capitalize(), value)

    # Build the PDF
    doc.build(content)

    # Move buffer pointer to the beginning
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=analysis_report.pdf"})



@app.post("/analyze-crop-soil/")
async def analyze_crop_soil(data: StateInput):
    global analysis
    try:
        prompt = f"""
        You are an expert in agricultural sciences. Based on the Indian state: {data.state}, and from the given disease name {analysis['disease']}, analyze the following:
        1. The general soil health and fertility in the region.
        2. The main crops that are currently cultivated in this region.
        3. The potential risks or deficiencies in the soil that might affect crop health.
        4. Additional crops that can be successfully grown in this region based on soil conditions, climate, and farming practices.
        5. Practical recommendations for improving soil fertility and maximizing crop yield.

        Respond in the following structured JSON format:

        {{
          "state": "{data.state}",
          "current_crop":"crop that was taken for analysis",
          "soil_health": "General health status of soil (e.g., fertile, acidic, saline, etc.)",
          "current_crops": ["List of major crops grown"],
          "soil_deficiencies": ["List of key deficiencies or risks, ESPECIALLY those related to the disease detected"],
          "suggested_crops": ["List of additional crops that can be grown"],
          "improvement_tips": [
            "Tip 1 for improving soil health",
            "Tip 2 for increasing crop yield",
            "Tip 3 for sustainable farming"
          ]
        }}

        Respond in JSON format ONLY, without any extra text.
        """

        # Generate response from Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        # Convert response to JSON format
        structured_response = response.text.strip("```json\n").strip("\n```")
        state_info = json.loads(structured_response)

        return state_info

    except Exception as e:
        return {"error": str(e)}