"""
train_model.py
==============
Generates a synthetic dataset grounded in mental-health research
correlations and trains a GradientBoosting pipeline.

Run with:
    python train_model.py
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

# ────────────────────────────────────────────────────────────────────────────
# 1.  Synthetic dataset generation
# ────────────────────────────────────────────────────────────────────────────
N = 4000

PLATFORMS = [
    "Facebook", "LinkedIn", "Instagram", "Snapchat",
    "Twitter", "YouTube", "TikTok", "LINE",
    "KakaoTalk", "VKontakte", "WhatsApp", "WeChat",
]
# Passive/comparison platforms tend to harm mental health more
PLATFORM_WEIGHT = {
    "Instagram": -0.4, "TikTok": -0.35, "Snapchat": -0.3,
    "Facebook": -0.25, "Twitter": -0.2,
    "YouTube": -0.1, "WhatsApp": -0.05,
    "LinkedIn": 0.1, "WeChat": 0.0,
    "LINE": 0.0, "KakaoTalk": 0.0, "VKontakte": -0.1,
}

PURPOSES = ["Networking", "Education", "Entertainment", "News"]
PURPOSE_WEIGHT = {
    "Education": 0.35, "Networking": 0.1,
    "News": -0.15, "Entertainment": -0.25,
}

COUNTRIES = [
    "India", "USA", "Canada", "Australia", "UK",
    "Germany", "Mexico", "Turkey", "France", "Other",
]

ACADEMIC_LEVELS = ["High School", "Undergraduate", "Graduate"]
STRESS_LEVELS   = ["Low", "Medium", "High", "Very High"]
STRESS_MAP      = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}

rows = []
for _ in range(N):
    age             = int(np.clip(np.random.normal(21, 4), 10, 40))
    gender          = random.choice(["Male", "Female"])
    country         = random.choice(COUNTRIES)
    academic_level  = random.choice(ACADEMIC_LEVELS)
    platform        = random.choice(PLATFORMS)
    purpose         = random.choice(PURPOSES)

    # Correlated features: stressed students sleep less, study more/less
    stress_idx      = np.random.choice([0, 1, 2, 3], p=[0.20, 0.35, 0.30, 0.15])
    stress_level    = STRESS_LEVELS[stress_idx]

    # Sleep: high-stress → less sleep
    sleep_base      = 8.0 - stress_idx * 0.7
    sleep_hours     = float(np.clip(np.random.normal(sleep_base, 0.8), 3.0, 10.0))

    # Physical activity: stressed people exercise less
    activity_base   = 1.5 - stress_idx * 0.2
    physical_activity = float(np.clip(np.random.normal(activity_base, 0.5), 0.0, 5.0))

    # Study hours
    study_hours     = float(np.clip(np.random.normal(4.5, 1.5), 0.0, 12.0))

    # Screen time: correlated with lower mental health
    screen_time     = float(np.clip(np.random.normal(4.0 + stress_idx * 0.5, 1.5), 0.5, 14.0))

    # Daily unlocks: correlated with screen time
    daily_unlocks   = int(np.clip(np.random.normal(screen_time * 12, 10), 5, 200))

    # ── Score calculation (domain-grounded formula) ──────────────────────────
    score = 5.5  # baseline

    # Sleep: optimal is 7-9 hrs
    sleep_delta = sleep_hours - 7.5
    score += np.tanh(sleep_delta * 0.5) * 1.5

    # Physical activity: each hour up to 2 hrs helps
    score += min(physical_activity, 2.0) * 0.6

    # Stress
    score -= stress_idx * 0.9

    # Screen time: >3 hrs starts hurting
    excess_screen = max(0, screen_time - 3.0)
    score -= excess_screen * 0.18

    # Daily unlocks: compulsive checking hurts
    score -= max(0, (daily_unlocks - 60)) * 0.008

    # Platform & purpose adjustments
    score += PLATFORM_WEIGHT.get(platform, 0)
    score += PURPOSE_WEIGHT.get(purpose, 0)

    # Study hours: moderate study is positive, extreme overwork is negative
    score += (4.0 - abs(study_hours - 4.0)) * 0.08

    # Gender small difference (research-based average tendency)
    if gender == "Female":
        score -= 0.1

    # Add realistic noise
    score += np.random.normal(0, 0.55)
    score = float(np.clip(score, 0.5, 10.0))

    rows.append({
        "Age"                     : age,
        "Gender"                  : gender,
        "Country"                 : country,
        "Academic_Level"          : academic_level,
        "Most_Used_Platform"      : platform,
        "Purpose_Of_Use"          : purpose,
        "Avg_Daily_Usage_Hours"   : round(screen_time, 2),
        "Daily_Unlocks"           : daily_unlocks,
        "Study_Hours"             : round(study_hours, 2),
        "Physical_Activity_Hours" : round(physical_activity, 2),
        "Sleep_Hours_Per_Night"   : round(sleep_hours, 2),
        "Stress_Level"            : stress_level,
        "Grouped_country"         : country,  # already from the list
        "Mental_Health_Score"     : round(score, 4),
    })

df = pd.DataFrame(rows)
print(f"Dataset shape: {df.shape}")
print(f"Score stats:\n{df['Mental_Health_Score'].describe().round(3)}\n")

# ────────────────────────────────────────────────────────────────────────────
# 2.  Feature / target split
# ────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "Age", "Gender", "Country", "Academic_Level",
    "Most_Used_Platform", "Purpose_Of_Use",
    "Avg_Daily_Usage_Hours", "Daily_Unlocks",
    "Study_Hours", "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night", "Stress_Level",
    "Grouped_country",
]
TARGET = "Mental_Health_Score"

X = df[FEATURES]
y = df[TARGET]

# ────────────────────────────────────────────────────────────────────────────
# 3.  Preprocessing
# ────────────────────────────────────────────────────────────────────────────
numeric_features = [
    "Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks",
    "Study_Hours", "Physical_Activity_Hours", "Sleep_Hours_Per_Night",
]
ordinal_features = ["Stress_Level"]
nominal_features = [
    "Gender", "Country", "Academic_Level",
    "Most_Used_Platform", "Purpose_Of_Use", "Grouped_country",
]

stress_order = [["Low", "Medium", "High", "Very High"]]

preprocessor = ColumnTransformer(transformers=[
    ("num",     StandardScaler(),                                           numeric_features),
    ("ordinal", OrdinalEncoder(categories=stress_order,
                               handle_unknown="use_encoded_value",
                               unknown_value=-1),                           ordinal_features),
    ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_features),
], remainder="drop")

# ────────────────────────────────────────────────────────────────────────────
# 4.  Model pipeline
# ────────────────────────────────────────────────────────────────────────────
gbr = GradientBoostingRegressor(
    n_estimators   = 500,
    learning_rate  = 0.05,
    max_depth      = 4,
    min_samples_leaf = 10,
    subsample      = 0.8,
    random_state   = 42,
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model",        gbr),
])

# ────────────────────────────────────────────────────────────────────────────
# 5.  Cross-validation
# ────────────────────────────────────────────────────────────────────────────
print("Running 5-fold cross-validation …")
kf  = KFold(n_splits=5, shuffle=True, random_state=42)
mae = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
r2  = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
print(f"  CV MAE : {(-mae).mean():.4f}  ± {mae.std():.4f}")
print(f"  CV R²  : {r2.mean():.4f}  ± {r2.std():.4f}\n")

# ────────────────────────────────────────────────────────────────────────────
# 6.  Fit on full dataset & save
# ────────────────────────────────────────────────────────────────────────────
print("Fitting on full dataset …")
pipeline.fit(X, y)

y_pred   = pipeline.predict(X)
train_mae = mean_absolute_error(y, y_pred)
train_r2  = r2_score(y, y_pred)
print(f"  Train MAE : {train_mae:.4f}")
print(f"  Train R²  : {train_r2:.4f}\n")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Mental_Health_Model.pkl")
joblib.dump(pipeline, out_path)
print(f"✅  Model saved → {out_path}")
