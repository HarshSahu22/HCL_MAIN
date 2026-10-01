import os
import sys
import re
from typing import Dict, List, Any, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import extract_skills_from_text, extract_education_level

def parse_job_description(jd_text: str) -> Dict[str, Any]:
    """Extract required skills, experience, and education from a Job Description."""
    skills = extract_skills_from_text(jd_text)
    education = extract_education_level(jd_text)
    
    # Extract minimum years of experience required
    exp_matches = re.findall(r"(\d{1,2})\+?\s*(?:-\s*\d{1,2})?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)", jd_text, re.IGNORECASE)
    min_exp = 0
    if exp_matches:
        min_exp = min([int(m) for m in exp_matches if int(m) <= 25])
        
    return {
        "jd_text": jd_text,
        "required_skills": skills,
        "required_education": education,
        "min_experience": min_exp
    }

def compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """Compute semantic text similarity between resume and JD using TF-IDF and cosine similarity."""
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(min(max(sim * 100, 0.0), 100.0))
    except Exception:
        return 50.0

def evaluate_match(resume_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compare candidate resume against Job Description and return match metrics and skill gap breakdown."""
    resume_skills = set(resume_data.get("skills", []))
    jd_skills = set(jd_data.get("required_skills", []))
    
    if jd_skills:
        matched_skills = sorted(list(resume_skills.intersection(jd_skills)))
        missing_skills = sorted(list(jd_skills - resume_skills))
        extra_skills = sorted(list(resume_skills - jd_skills))
        skill_score = (len(matched_skills) / len(jd_skills)) * 100.0
    else:
        matched_skills = sorted(list(resume_skills))
        missing_skills = []
        extra_skills = []
        skill_score = min(len(resume_skills) * 10.0, 100.0)
        
    tfidf_score = compute_tfidf_similarity(resume_data.get("raw_text", ""), jd_data.get("jd_text", ""))
    
    # Combined skill match score (70% skill coverage + 30% semantic text alignment)
    final_skills_match_score = (skill_score * 0.70) + (tfidf_score * 0.30)
    final_skills_match_score = min(max(final_skills_match_score, 0.0), 100.0)
    
    # Experience match check
    candidate_exp = resume_data.get("years_experience", 0)
    required_exp = jd_data.get("min_experience", 0)
    exp_status = "Meets or exceeds" if candidate_exp >= required_exp else f"Below requirement ({candidate_exp} yrs vs {required_exp} yrs required)"
    
    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills,
        "skill_score_pct": round(skill_score, 1),
        "tfidf_similarity_pct": round(tfidf_score, 1),
        "composite_skills_match_score": round(final_skills_match_score, 1),
        "candidate_experience": candidate_exp,
        "required_experience": required_exp,
        "experience_status": exp_status
    }
