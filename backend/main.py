from fastapi import FastAPI
import __main__
import joblib
import pandas as pd
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from utils import add_engagement

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Backward compatibility for older pickles saved from train.py when it ran as
# a script and serialized the transformer as __main__.add_engagement.
setattr(__main__, "add_engagement", add_engagement)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "https://student-performance-liard.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reg_model = joblib.load(MODEL_DIR / "score_model.pkl")
clf_model = joblib.load(MODEL_DIR / "grade_model.pkl")


class PredictionInput(BaseModel):
    weekly_self_study_hours: float = Field(..., ge=0, le=40)
    attendance_percentage: float = Field(..., ge=0, le=100)
    class_participation: float = Field(..., ge=0, le=10)


def summarize_band(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Strong"
    if score >= 55:
        return "Developing"
    if score >= 40:
        return "At Risk"
    return "Critical Support"


def estimate_risk(score: float, engagement: float) -> str:
    if score >= 80 and engagement >= 70:
        return "Low"
    if score >= 60 and engagement >= 50:
        return "Moderate"
    return "High"


def build_strengths(data: PredictionInput) -> list[str]:
    strengths = []

    if data.attendance_percentage >= 90:
        strengths.append("Attendance is excellent and supports stable performance.")
    if data.weekly_self_study_hours >= 15:
        strengths.append("Self-study effort is strong for sustained score growth.")
    if data.class_participation >= 7:
        strengths.append("Class participation suggests active learning habits.")

    if not strengths:
        strengths.append("You already have a baseline to build from.")

    return strengths


def build_recommendations(data: PredictionInput, score: float) -> list[str]:
    tips = []

    if data.attendance_percentage < 85:
        tips.append("Raise attendance consistency to reduce score volatility.")
    if data.weekly_self_study_hours < 12:
        tips.append("Add 3 to 5 more self-study hours each week for stronger outcomes.")
    if data.class_participation < 6:
        tips.append("Increase class participation through questions, discussions, or recaps.")
    if score < 70:
        tips.append("Focus on weekly revision blocks and short concept reviews after each class.")

    if not tips:
        tips.append("Maintain the current routine and target mock assessments for further gains.")

    return tips


def build_focus_area(data: PredictionInput) -> str:
    values = {
        "self-study": data.weekly_self_study_hours / 40,
        "attendance": data.attendance_percentage / 100,
        "participation": data.class_participation / 10,
    }
    lowest_area = min(values, key=values.get)
    return lowest_area.replace("-", " ").title()

@app.get("/")
def home():
    return {"message": "Student Performance API running"}


@app.get("/presets")
def presets():
    return {
        "profiles": [
            {
                "name": "High Achiever",
                "weekly_self_study_hours": 18,
                "attendance_percentage": 94,
                "class_participation": 8,
            },
            {
                "name": "Balanced Learner",
                "weekly_self_study_hours": 12,
                "attendance_percentage": 86,
                "class_participation": 6,
            },
            {
                "name": "Needs Support",
                "weekly_self_study_hours": 6,
                "attendance_percentage": 72,
                "class_participation": 4,
            },
        ]
    }

@app.post("/predict")
def predict(data: PredictionInput):
    df = pd.DataFrame([data.model_dump()])
    df = add_engagement(df)

    score = round(float(reg_model.predict(df)[0]), 1)
    grade = clf_model.predict(df)[0]
    engagement = round(float(df["engagement"].iloc[0]), 1)
    consistency_index = round(float(df["consistency_index"].iloc[0]), 1)
    support_need_index = round(float(df["support_need_index"].iloc[0]), 1)

    return {
        "predicted_score": score,
        "predicted_grade": grade,
        "performance_band": summarize_band(score),
        "risk_level": estimate_risk(score, engagement),
        "focus_area": build_focus_area(data),
        "insights": {
            "engagement_score": engagement,
            "consistency_index": consistency_index,
            "support_need_index": support_need_index,
        },
        "strengths": build_strengths(data),
        "recommendations": build_recommendations(data, score),
    }
