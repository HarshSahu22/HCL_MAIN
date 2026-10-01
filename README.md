# 🎯 AI-Based Resume Screening and Candidate Shortlisting System

An intelligent, end-to-end Machine Learning and Natural Language Processing (NLP) web platform that automatically parses resumes (PDF / Word DOCX / Text), evaluates candidates against Job Descriptions (JDs), performs skill-gap analysis, and accurately predicts candidate shortlisting decisions.

---

## 🌟 Key Features

1. **Multi-Format Resume Parser (`resume_parser.py`)**:
   - Supports **PDF** (via `pypdf`/`pdfplumber`), **Word (`.docx`)** (via `python-docx`), and plain **`.txt`** files.
   - Extracts candidate name, email, phone number, LinkedIn/GitHub links, education level (High School, Bachelors, Masters, PhD), years of work experience, portfolio project count, and GitHub activity score.
   - Built-in technical skills taxonomy spanning 100+ languages, frameworks, AI/ML tools, cloud platforms, and databases.

2. **Job Description (JD) Semantic Matcher (`matcher.py`)**:
   - Extracts mandatory technical skills, required education, and minimum experience thresholds from any pasted Job Description.
   - Computes **Direct Skill Overlap & Gap Analysis** (Matched Skills vs Missing Skills vs Bonus Skills).
   - Computes **TF-IDF Vectorization & Cosine Similarity** to measure contextual text relevance.

3. **Machine Learning Shortlisting Engine (`screening_engine.py`)**:
   - Uses a trained **Gradient Boosting Classifier** model trained on 30,000+ candidate screening records (`models/resume_model.pkl`).
   - Produces probability confidence scores (`0% - 100%`) and decisive verdict: `🎉 SHORTLISTED` vs `⚠️ NOT SHORTLISTED`.
   - Generates **Explainable AI Insights**: lists key strengths, red flags, and personalized resume improvement recommendations.

4. **Interactive Streamlit Web Application (`app.py`)**:
   - **Single Candidate Screening**: Deep-dive analysis with donut charts, skill pill tags, score breakdown progress meters, and raw text preview.
   - **Batch Candidate Leaderboard**: Upload dozens of resumes at once against a single Job Description, auto-rank candidates on a dynamic leaderboard, and export results to CSV.
   - **Manual Feature Simulator**: Interactive sliders to simulate any hypothetical candidate profile.
   - **System Info & Dataset Visualizer**: Training dataset histograms, box plots, and model architecture documentation.

---

## 🚀 How to Run the Application

### 1. Activate the Virtual Environment
```bash
# In Command Prompt / PowerShell:
.\venv\Scripts\activate
```

### 2. Launch the Streamlit App
```bash
cd ai_resume_analyser
streamlit run app.py
```

Or from the root directory:
```bash
streamlit run ai_resume_analyser/app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
HCL_MAIN/
├── ai_resume_analyser/
│   ├── app.py                      # Main Streamlit web application & UI
│   ├── resume_parser.py            # PDF/DOCX parser & entity extractor
│   ├── matcher.py                  # JD parser & TF-IDF semantic matcher
│   ├── screening_engine.py         # ML prediction pipeline & explainability
│   ├── sample_data.py              # Sample JDs and preset resumes
│   ├── generate_sample_resumes.py  # Helper to generate test PDF/DOCX files
│   ├── ai_resume_screening.csv     # 30,000+ training records dataset
│   ├── ai_resume.ipynb             # Model training notebook
│   ├── sample_resumes/             # Ready-to-use test resumes (.pdf & .docx)
│   └── models/
│       ├── resume_model.pkl        # Trained Gradient Boosting model
│       ├── scaler.pkl              # Feature standard scaler
│       └── metadata.pkl            # Model metadata & feature order
├── requirements.txt                # Project dependencies
└── README.md                       # Documentation
```
