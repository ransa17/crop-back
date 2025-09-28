import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import requests
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting API server setup...")

apiKey = "AIzaSyCjFAHyFzSZdqSew1IA4UHspe18hRgMg1M"

def get_recommendation_from_gemini(prompt: str):
    """Generates a human-readable recommendation from Gemini API."""
    logging.info("Generating AI expert summary using Gemini API...")
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
    }
    retries = 3
    for i in range(retries):
        try:
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No recommendation available.')
            return text
        except requests.exceptions.RequestException as e:
            logging.error(f"API call failed: {e}")
            if i < retries - 1:
                wait_time = 2 ** i
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return "Failed to get recommendation from AI after multiple retries."
        except Exception as e:
            logging.error(f"Failed to parse AI response: {e}")
            return "AI response was unreadable."

def get_weekly_weather_data(district: str):
    """Simulates fetching average weekly weather data for a given district."""
    logging.info(f"Fetching average weekly weather data for {district}...")
    weather_data = {
        'Cuttack': {'rainfall_mm': 25, 'avg_temp_c': 20, 'humidity_pct': 70},
        'Balasore': {'rainfall_mm': 5, 'avg_temp_c': 30, 'humidity_pct': 85},
        'Sambalpur': {'rainfall_mm': 50, 'avg_temp_c': 25, 'humidity_pct': 90},
        'Ganjam': {'rainfall_mm': 10, 'avg_temp_c': 35, 'humidity_pct': 60},
        # Add all districts as needed
    }
    return weather_data.get(district, {'rainfall_mm': 0, 'avg_temp_c': 0, 'humidity_pct': 0})

try:
    logging.info("Loading pre-trained models and preprocessor...")
    rf_model_yield = joblib.load('rf_model_yield.pkl')
    rf_model_irrigation = joblib.load('rf_model_irrigation.pkl')
    rf_model_fertilizer = joblib.load('rf_model_fertilizer.pkl')
    rf_model_pest = joblib.load('rf_model_pest.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    logging.info("Models and preprocessor loaded successfully.")
except FileNotFoundError:
    logging.error("Model files not found. Please run 'python yield_predictor.py' first.")
    raise HTTPException(status_code=500, detail="Model files not found. Please train the model first.")

app = FastAPI(title="AI Farming Platform API")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    crop_type: str
    district: str
    crop_growth_stage: str
    soil_ph: float
    soil_nitrogen_mg_kg: float
    soil_moisture_pct: float
    soil_organic_matter_pct: float = 0.0
@app.post("/predict")
async def predict_yield(request: PredictionRequest):
    logging.info(f"Received prediction request for {request.crop_type} in {request.district}.")
    try:
        avg_weekly_weather = get_weekly_weather_data(request.district)
        input_data = pd.DataFrame([{
            'crop_type': request.crop_type,
            'district': request.district,
            'crop_growth_stage': request.crop_growth_stage,
            'rainfall_mm': avg_weekly_weather['rainfall_mm'],
            'avg_temp_c': avg_weekly_weather['avg_temp_c'],
            'humidity_pct': avg_weekly_weather['humidity_pct'],
            'soil_ph': request.soil_ph,
            'soil_nitrogen_mg_kg': request.soil_nitrogen_mg_kg,
            'soil_moisture_pct': request.soil_moisture_pct,
            'soil_organic_matter_pct': request.soil_organic_matter_pct
        }])
        processed_data = preprocessor.transform(input_data)
        predicted_yield = rf_model_yield.predict(processed_data)[0]
        irrigation_rec_model = rf_model_irrigation.predict(processed_data)[0]
        fertilizer_rec_model = rf_model_fertilizer.predict(processed_data)[0]
        pest_control_rec_model = rf_model_pest.predict(processed_data)[0]

        # Prepare summaries for prompt
        weather_summary = f"rainfall: {avg_weekly_weather['rainfall_mm']}mm, temp: {avg_weekly_weather['avg_temp_c']}°C, humidity: {avg_weekly_weather['humidity_pct']}%"
        soil_summary = f"pH: {request.soil_ph}, Nitrogen: {request.soil_nitrogen_mg_kg}mg/kg, Moisture: {request.soil_moisture_pct}%, Organic Matter: {request.soil_organic_matter_pct}%"

        # Build prompts for Gemini, including model prediction
        irrigation_prompt = (
            f"As an agricultural expert, provide a concise, single-paragraph irrigation recommendation for a {request.crop_type} crop, "
            f"not more than 60 words, at the {request.crop_growth_stage} stage with the following conditions: {weather_summary} and {soil_summary}. "
            f"The predicted yield is {predicted_yield:.2f} kg/hectare. "
            f"The model suggests '{irrigation_rec_model}' irrigation. "
            f"Please combine the model suggestion with your own advice for best results."
        )
        fertilizer_prompt = (
            f"As an agricultural expert, provide a concise, single-paragraph fertilization recommendation for a {request.crop_type} crop, "
            f"not more than 60 words, at the {request.crop_growth_stage} stage with the following conditions: {weather_summary} and {soil_summary}. "
            f"The predicted yield is {predicted_yield:.2f} kg/hectare. "
            f"The model suggests '{fertilizer_rec_model}' fertilizer. "
            f"Please combine the model suggestion with your own advice for best results."
        )
        pest_control_prompt = (
            f"As an agricultural expert, provide a concise, single-paragraph pest control recommendation for a {request.crop_type} crop, "
            f"not more than 60 words, at the {request.crop_growth_stage} stage with the following conditions: {weather_summary} and {soil_summary}. "
            f"The predicted yield is {predicted_yield:.2f} kg/hectare. "
            f"The model suggests '{pest_control_rec_model}' pest control. "
            f"Please combine the model suggestion with your own advice for best results."
        )
        summary_prompt = (
            f"As an agricultural expert, provide a brief, overall summary for a small-scale farmer to improve their {request.crop_type} crop, "
            f"not more than 100 words. Include key insights from the following conditions: {weather_summary}, {soil_summary}, and a predicted yield of {predicted_yield:.2f} kg/hectare. "
            f"Model suggestions: irrigation='{irrigation_rec_model}', fertilizer='{fertilizer_rec_model}', pest control='{pest_control_rec_model}'. "
            f"Combine these with your own advice."
        )

        # Get AI recommendations
        irrigation_rec_ai = get_recommendation_from_gemini(irrigation_prompt)
        fertilizer_rec_ai = get_recommendation_from_gemini(fertilizer_prompt)
        pest_control_rec_ai = get_recommendation_from_gemini(pest_control_prompt)
        summary_rec_ai = get_recommendation_from_gemini(summary_prompt)

        return {
            "predicted_yield": float(predicted_yield),
            "irrigation_rec_model": str(irrigation_rec_model),
            "irrigation_rec_ai": irrigation_rec_ai,
            "fertilizer_rec_model": str(fertilizer_rec_model),
            "fertilizer_rec_ai": fertilizer_rec_ai,
            "pest_control_rec_model": str(pest_control_rec_model),
            "pest_control_rec_ai": pest_control_rec_ai,
            "summary_rec_ai": summary_rec_ai
        }
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
