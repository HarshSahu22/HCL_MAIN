import os
import joblib
import pandas as pd
from typing import Dict, Any, Tuple

# Resolve absolute paths for models
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "resume_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.pkl")

# Load trained artifacts (supports single bundled best_model.pkl or separate files)
model, scaler, metadata = None, None, None
try:
    if os.path.exists(BEST_MODEL_PATH):
        bundle = joblib.load(BEST_MODEL_PATH)
        if isinstance(bundle, dict) and "model" in bundle:
            model = bundle["model"]
            scaler = bundle["scaler"]
            metadata = bundle["metadata"]
    if model is None:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        metadata = joblib.load(METADATA_PATH)
except Exception as e:
    print(f"Warning: Model loading failed: {e}")

def predict_candidate_shortlist(features_dict: Dict[str, Any], threshold: float = 0.50) -> Dict[str, Any]:
    """
    Predict whether a candidate is shortlisted based on extracted features using the trained ML model.
    """
    if model is None or scaler is None or metadata is None:
        raise RuntimeError("ML Model or Scaler not loaded properly.")
        
    edu_map = metadata.get("edu_mapping", {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3})
    edu_val = edu_map.get(features_dict.get("education_level", "Bachelors"), 1)
    
    # Construct feature dataframe matching exact training order
    candidate_df = pd.DataFrame([{
        "years_experience": float(features_dict.get("years_experience", 0)),
        "skills_match_score": float(features_dict.get("skills_match_score", 0.0)),
        "education_level": edu_val,
        "project_count": float(features_dict.get("project_count", 0)),
        "resume_length": float(features_dict.get("resume_length", 300)),
        "github_activity": float(features_dict.get("github_activity", 200))
    }])[metadata["feature_order"]]
    
    # Scale features and predict probability
    scaled_features = scaler.transform(candidate_df)
    prob_shortlisted = float(model.predict_proba(scaled_features)[0, 1])
    is_shortlisted = prob_shortlisted >= threshold
    
    # Generate explainable reasons & suggestions
    reasons = []
    strengths = []
    improvements = []
    
    skills_score = float(features_dict.get("skills_match_score", 0.0))
    exp = float(features_dict.get("years_experience", 0))
    proj = float(features_dict.get("project_count", 0))
    edu = features_dict.get("education_level", "Bachelors")
    gh = float(features_dict.get("github_activity", 200))
    
    # Strengths
    if skills_score >= 70:
        strengths.append(f"Strong skill alignment with Job Description ({skills_score:.1f}% match).")
    if exp >= 5:
        strengths.append(f"Solid professional experience ({exp:.0f} years).")
    if proj >= 6:
        strengths.append(f"High volume of practical project work ({proj:.0f} projects detected).")
    if edu in ["Masters", "PhD"]:
        strengths.append(f"Advanced degree qualification ({edu}).")
    if gh >= 400:
        strengths.append("High GitHub/open-source engineering activity.")
        
    # Gaps / Improvement points
    if skills_score < 60:
        improvements.append("Skill match with Job Description is below target (acquire or highlight missing key technologies).")
    if exp < 2:
        improvements.append("Early career / low recorded years of experience.")
    if proj < 4:
        improvements.append("Resume lists fewer than 4 distinct portfolio projects or technical achievements.")
    if gh < 250:
        improvements.append("GitHub link missing or low commit/repository activity detected.")
        
    if is_shortlisted:
        decision_title = "Candidate Shortlisted"
        summary = f"The candidate meets the role requirements with a high shortlist probability of {prob_shortlisted*100:.1f}%."
    else:
        decision_title = "Candidate Not Shortlisted"
        summary = f"The candidate does not currently meet the shortlisting threshold ({prob_shortlisted*100:.1f}% vs {threshold*100:.0f}% required)."
        
    return {
        "is_shortlisted": is_shortlisted,
        "probability": round(prob_shortlisted * 100, 1),
        "threshold": round(threshold * 100, 1),
        "decision_title": decision_title,
        "summary": summary,
        "model_name": metadata.get("model_name", "Gradient Boosting Classifier"),
        "strengths": strengths if strengths else ["Basic profile qualifications met."],
        "improvements": improvements if improvements else ["Profile is well-rounded for this role."],
        "features_used": candidate_df.to_dict(orient="records")[0]
    }
