import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List

# Resolve absolute paths so it works both locally and on Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'Mental_Health_Model.pkl'))

top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']

app = FastAPI(title="Mental Health Signal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ─────────────────────────────────────────────────────────
class StudentData(BaseModel):
    age                     : int   = Field(..., ge=10, le=100)
    gender                  : Literal['Male', 'Female']
    country                 : str
    academic_level          : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform      : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat',
                                      'Twitter', 'YouTube', 'TikTok', 'LINE',
                                      'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat']
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Medium', 'Low', 'Very High', 'High']


class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


class AnalyzeResponse(BaseModel):
    score          : float
    category       : str
    risk_factors   : List[str]
    positives      : List[str]
    recommendations: List[dict]   # [{icon, title, detail}]


# ── Helper: build personalized recommendations ───────────────────────────────
def build_recommendations(data: StudentData, score: float) -> AnalyzeResponse:
    risk_factors : List[str] = []
    positives    : List[str] = []
    tips         : List[dict] = []

    stress_map = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
    stress_idx = stress_map[data.stress_level]

    # ── Category ────────────────────────────────────────────────────────────
    if score < 4.0:
        category = "strained"
    elif score < 7.0:
        category = "balanced"
    else:
        category = "strong"

    # ── Risk factors & positives ─────────────────────────────────────────────
    # Sleep
    if data.sleep_hours_per_night < 6.0:
        risk_factors.append(f"Low sleep ({data.sleep_hours_per_night:.1f} hrs/night)")
        tips.append({
            "icon": "🌙",
            "title": "Prioritize sleep",
            "detail": (
                f"You're sleeping only {data.sleep_hours_per_night:.1f} hrs. "
                "Aim for 7–9 hrs — even one extra hour can measurably reduce stress hormones. "
                "Try setting a consistent bedtime 30 min earlier each week."
            ),
        })
    elif data.sleep_hours_per_night > 9.5:
        risk_factors.append(f"Oversleeping ({data.sleep_hours_per_night:.1f} hrs/night) — can signal low energy")
        tips.append({
            "icon": "☀️",
            "title": "Regulate your sleep schedule",
            "detail": (
                "Sleeping more than 9–10 hrs regularly can actually increase fatigue. "
                "Try to wake at a consistent time and get morning sunlight to reset your body clock."
            ),
        })
    else:
        positives.append(f"Good sleep hygiene ({data.sleep_hours_per_night:.1f} hrs/night) ✓")

    # Physical activity
    if data.physical_activity_hours < 0.5:
        risk_factors.append("Very low physical activity")
        tips.append({
            "icon": "🏃",
            "title": "Start moving — even 20 min counts",
            "detail": (
                "No physical activity is one of the strongest predictors of low mood. "
                "Start with a 20-minute walk daily. You don't need a gym — bodyweight exercises "
                "at home have the same mental health benefit."
            ),
        })
    elif data.physical_activity_hours < 1.0:
        tips.append({
            "icon": "🚶",
            "title": "Increase activity gradually",
            "detail": (
                f"You get {data.physical_activity_hours:.1f} hr of activity/day — good start! "
                "Aim for at least 30 min of moderate exercise daily. "
                "Try adding a short lunchtime walk or an evening stretch routine."
            ),
        })
    else:
        positives.append(f"Active lifestyle ({data.physical_activity_hours:.1f} hrs/day of activity) ✓")

    # Screen time
    if data.avg_daily_usage_hours > 6.0:
        risk_factors.append(f"Excessive screen time ({data.avg_daily_usage_hours:.1f} hrs/day)")
        tips.append({
            "icon": "📵",
            "title": "Set screen-time boundaries",
            "detail": (
                f"At {data.avg_daily_usage_hours:.1f} hrs/day of social media, you're well above the "
                "2–3 hr threshold linked to mental wellness. Try app timers, grayscale mode, "
                "or a hard stop 1 hour before bed."
            ),
        })
    elif data.avg_daily_usage_hours > 3.0:
        tips.append({
            "icon": "📱",
            "title": "Trim screen time slightly",
            "detail": (
                f"You use social media {data.avg_daily_usage_hours:.1f} hrs/day. "
                "Studies suggest keeping it under 3 hrs/day improves mood. "
                "Try batching social media to 2–3 scheduled sessions instead of constant checking."
            ),
        })
    else:
        positives.append(f"Healthy screen time ({data.avg_daily_usage_hours:.1f} hrs/day) ✓")

    # Daily unlocks
    if data.daily_unlocks > 100:
        risk_factors.append(f"High phone-checking frequency ({data.daily_unlocks} unlocks/day)")
        tips.append({
            "icon": "🔔",
            "title": "Cut notification interruptions",
            "detail": (
                f"Unlocking your phone {data.daily_unlocks}× a day fragments your focus and elevates anxiety. "
                "Turn off non-essential notifications, use Do Not Disturb during study, "
                "and try leaving your phone in another room for 2-hour blocks."
            ),
        })
    elif data.daily_unlocks > 60:
        tips.append({
            "icon": "⏱️",
            "title": "Be intentional with your phone",
            "detail": (
                f"At {data.daily_unlocks} unlocks/day, habitual checking is creeping in. "
                "Try keeping your phone face-down and only checking at defined intervals."
            ),
        })

    # Stress level
    if stress_idx >= 2:
        risk_factors.append(f"Elevated stress level ({data.stress_level})")
        tips.append({
            "icon": "🧘",
            "title": "Build a stress-reduction routine",
            "detail": (
                f"You rated stress as '{data.stress_level}'. "
                "Practices like 5-min deep breathing, journaling, or progressive muscle relaxation "
                "can lower cortisol within days. Consider the '4-7-8' breathing technique before bed."
            ),
        })
    else:
        positives.append(f"Manageable stress level ({data.stress_level}) ✓")

    # Platform-specific tip
    passive_platforms = {"Instagram", "TikTok", "Snapchat", "Facebook"}
    if data.most_used_platform in passive_platforms:
        tips.append({
            "icon": "🔄",
            "title": f"Rethink your {data.most_used_platform} use",
            "detail": (
                f"{data.most_used_platform} is a comparison-heavy platform linked to lower "
                "self-esteem in multiple studies. Try unfollowing accounts that trigger negative "
                "feelings, or replace 30 min of scrolling with a hobby or podcast."
            ),
        })

    # Purpose-specific tip
    if data.purpose_of_use == "Entertainment":
        tips.append({
            "icon": "📚",
            "title": "Mix entertainment with meaningful content",
            "detail": (
                "Using social media primarily for entertainment is correlated with lower wellbeing. "
                "Try mixing in educational content, skill-building videos, or connecting with interest communities."
            ),
        })
    elif data.purpose_of_use == "Education":
        positives.append("Using social media for education — positive use pattern ✓")

    # Study hours balance
    if data.study_hours > 10.0:
        risk_factors.append(f"Overworking — studying {data.study_hours:.1f} hrs/day")
        tips.append({
            "icon": "⚖️",
            "title": "Balance study with recovery",
            "detail": (
                f"Studying {data.study_hours:.1f} hrs/day without adequate breaks leads to burnout. "
                "Use the Pomodoro technique (50 min study, 10 min break), "
                "and protect at least 2 hrs of pure leisure/recovery time each day."
            ),
        })
    elif data.study_hours >= 3.0:
        positives.append(f"Balanced study load ({data.study_hours:.1f} hrs/day) ✓")

    # Strong score positive reinforcement
    if score >= 7.0:
        tips.append({
            "icon": "🌟",
            "title": "You're doing well — keep it up",
            "detail": (
                "Your habits are in a strong place. To stay there: maintain your sleep schedule "
                "through busy periods, keep physical activity consistent, and check in on your "
                "stress before it builds. Small maintenance is much easier than recovery."
            ),
        })

    # Always have at least 3 recommendations
    if len(tips) < 3:
        tips.append({
            "icon": "🤝",
            "title": "Social connection matters",
            "detail": (
                "Strong social bonds are one of the best buffers against poor mental health. "
                "Try to spend quality offline time with friends or family at least 2–3× per week."
            ),
        })
        tips.append({
            "icon": "🥗",
            "title": "Nutrition supports your mood",
            "detail": (
                "Diet is often overlooked in mental wellness. Reducing processed foods and sugar, "
                "and increasing omega-3s (fish, flax), magnesium (leafy greens), and B vitamins "
                "can noticeably improve mood and focus within weeks."
            ),
        })

    return AnalyzeResponse(
        score           = round(score, 2),
        category        = category,
        risk_factors    = risk_factors[:4],       # cap to top 4
        positives       = positives[:4],
        recommendations = tips[:6],               # cap to 6 tips
    )


# ── Build input DataFrame ────────────────────────────────────────────────────
def build_input(data: StudentData) -> pd.DataFrame:
    country_group = data.country if data.country in top_countries else "Other"
    return pd.DataFrame([{
        "Age"                     : data.age,
        "Gender"                  : data.gender,
        "Country"                 : data.country,
        "Academic_Level"          : data.academic_level,
        "Most_Used_Platform"      : data.most_used_platform,
        "Purpose_Of_Use"          : data.purpose_of_use,
        "Avg_Daily_Usage_Hours"   : data.avg_daily_usage_hours,
        "Daily_Unlocks"           : data.daily_unlocks,
        "Study_Hours"             : data.study_hours,
        "Physical_Activity_Hours" : data.physical_activity_hours,
        "Sleep_Hours_Per_Night"   : data.sleep_hours_per_night,
        "Stress_Level"            : data.stress_level,
        "Grouped_country"         : country_group,
    }])


# ── Static file routes ───────────────────────────────────────────────────────
@app.get('/', include_in_schema=False)
def serve_index():
    return FileResponse(os.path.join(BASE_DIR, 'index.html'))

@app.get('/style.css', include_in_schema=False)
def serve_css():
    return FileResponse(os.path.join(BASE_DIR, 'style.css'), media_type='text/css')

@app.get('/script.js', include_in_schema=False)
def serve_js():
    return FileResponse(os.path.join(BASE_DIR, 'script.js'), media_type='application/javascript')


# ── API routes ───────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    prediction = model.predict(build_input(data))[0]
    return PredictionResponse(predicted_mental_health_score=round(float(prediction), 2))

@app.post('/analyze', response_model=AnalyzeResponse)
def analyze(data: StudentData):
    prediction = float(model.predict(build_input(data))[0])
    prediction = max(0.0, min(10.0, prediction))
    return build_recommendations(data, prediction)