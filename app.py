from typing import Tuple
import tensorflow as tf
import numpy as np
import json
import logging
import os
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

dotenv.load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_KEY")

analysis={}

# Load the trained model
model = load_model("multiclass_inceptionv3.keras")
client = genai.Client(api_key=GEMINI_API_KEY)

class CropAnalysisRequest(BaseModel):
    class_name: str  # Renamed 'class' to 'class_name' since 'class' is a reserved keyword
    confidence: float
    image_size: list[int]

class StateInput(BaseModel):
    state: str

# Load class labels
with open("class_labels.json", "r") as f:
    CLASS_LABELS = json.load(f)

# Define image preprocessing function
def read_file_as_image(data) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(io.BytesIO(data)).convert('RGB')  # Open the image and convert it to RGB color space
    img_resized = img.resize((299, 299), resample=Image.BICUBIC)  # Resize the image to 224 x 224
    image = np.array(img_resized)  # Convert the image to a numpy array
    return image, img_resized.size  # Return the image and its size

# Initialize FastAPI
app = FastAPI()

@app.post("/predict/")  # A decorator to create a route for the predict endpoint
async def predict(file: UploadFile = File(...)):  # The function that will be executed when the endpoint is called
    try:  # A try block to handle any errors that may occur
        image_data, image_size = read_file_as_image(await file.read())  # Read the image file
        img_batch = np.expand_dims(image_data, 0)  # Add an extra dimension to the image
        
        predictions = model.predict(img_batch)  # Make a prediction
        predicted_class = CLASS_LABELS[np.argmax(predictions[0])]  # Get the predicted class
        confidence = float(np.max(predictions[0]))  # Get the confidence of the prediction
 
        return {  # Return the prediction
            'class_name': predicted_class,   
            'confidence': confidence,
            'image_size': image_size  # Optionally return the original image size
        }
    except Exception as e:  # If an error occurs
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))  # Raise an HTTPException with the error message
    


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