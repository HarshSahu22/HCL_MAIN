import os
import io
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from resume_parser import parse_resume, extract_text_from_file
from matcher import parse_job_description, evaluate_match
from screening_engine import predict_candidate_shortlist, metadata

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Resume Screening & Shortlisting System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1e38 0%, #0f172a 50%, #132238 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    
    .main-header h1 {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    .card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
    }
    
    .verdict-shortlisted {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .verdict-rejected {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05));
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .badge-match {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    
    .badge-missing {
        display: inline-block;
        background: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    
    .badge-extra {
        display: inline-block;
        background: rgba(148, 163, 184, 0.15);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 3px;
    }
    
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="main-header">
    <h1>🎯 AI Resume Screening & Shortlisting System</h1>
    <p>Automated PDF/DOCX Resume Parsing, Job Description Semantic Matching & ML Candidate Decision Engine</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1586281380349-632531db7ed4?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.title("Navigation & Config")
    app_mode = st.radio(
        "Select Workflow:",
        ["📄 Single Resume Screening", "📊 Batch Resume Screening (Leaderboard)"],
        index=0
    )
    
    st.divider()
    st.subheader("⚙️ Decision Settings")
    shortlist_threshold = st.slider(
        "Shortlisting Probability Cutoff:",
        min_value=0.30,
        max_value=0.85,
        value=0.50,
        step=0.05,
        help="Candidates with model prediction probability above this threshold will be shortlisted."
    )
    
    st.markdown("---")
    model_name = metadata.get("model_name", "Gradient Boosting") if metadata else "Gradient Boosting"
    st.info(f"🤖 **Active ML Engine:**\n`{model_name}`\nTrained on 30,000+ candidate profiles.")

# ==========================================
# WORKFLOW 1: SINGLE RESUME SCREENING
# ==========================================
if app_mode == "📄 Single Resume Screening":
    st.subheader("1. Job Description & Candidate Resume Input")
    
    col_jd, col_resume = st.columns([1, 1], gap="medium")
    
    with col_jd:
        st.markdown("### 📋 Job Description (JD)")
        jd_input_text = st.text_area(
            "Paste Job Description:",
            value="",
            height=260,
            placeholder="Paste Job Description requirements, required skills, degree, and years of experience here..."
        )
        
    with col_resume:
        st.markdown("### 📄 Candidate Resume")
        uploaded_file = st.file_uploader(
            "Upload Candidate Resume (PDF / DOCX / TXT):",
            type=["pdf", "docx", "txt"],
            help="Supports PDF, Word (.docx), and plain text (.txt) documents"
        )
        
        parsed_resume_data = None
        if uploaded_file is not None:
            with st.spinner("Extracting candidate information and parsing skills..."):
                parsed_resume_data = parse_resume(uploaded_file)
            st.success(f"✅ Loaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        else:
            st.info("👆 Please upload a candidate resume file (PDF, DOCX, or TXT) to begin.")
            
    st.markdown("---")
    
    # Process and Screen Button
    if st.button("🚀 Analyze & Screen Candidate", type="primary", use_container_width=True):
        if not jd_input_text.strip():
            st.warning("⚠️ Please provide a Job Description to evaluate the candidate against.")
        elif not parsed_resume_data:
            st.warning("⚠️ Please upload or select a Candidate Resume.")
        else:
            with st.spinner("Analyzing Job Description, matching skills, and computing ML shortlisting probability..."):
                time.sleep(0.4) # smooth transition
                jd_data = parse_job_description(jd_input_text)
                match_results = evaluate_match(parsed_resume_data, jd_data)
                
                # Combine into candidate feature dict for ML model
                features = {
                    "years_experience": parsed_resume_data["years_experience"],
                    "skills_match_score": match_results["composite_skills_match_score"],
                    "education_level": parsed_resume_data["education"],
                    "project_count": parsed_resume_data["project_count"],
                    "resume_length": parsed_resume_data["resume_length"],
                    "github_activity": parsed_resume_data["github_activity"]
                }
                
                decision = predict_candidate_shortlist(features, threshold=shortlist_threshold)
                
                # Render Verdict Banner
                prob = decision["probability"]
                is_short = decision["is_shortlisted"]
                
                st.subheader("2. AI Screening Decision & Results")
                
                if is_short:
                    st.markdown(f"""
                    <div class="verdict-shortlisted">
                        <h2 style="color: #10b981; margin: 0 0 0.5rem 0;">🎉 CANDIDATE SHORTLISTED</h2>
                        <h3 style="color: #34d399; margin: 0; font-size: 1.6rem;">{prob:.1f}% Shortlisting Probability (Cutoff: {shortlist_threshold*100:.0f}%)</h3>
                        <p style="color: #94a3b8; margin: 0.6rem 0 0 0; font-size: 1.05rem;">{decision['summary']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-rejected">
                        <h2 style="color: #ef4444; margin: 0 0 0.5rem 0;">⚠️ CANDIDATE NOT SHORTLISTED</h2>
                        <h3 style="color: #f87171; margin: 0; font-size: 1.6rem;">{prob:.1f}% Shortlisting Probability (Cutoff: {shortlist_threshold*100:.0f}%)</h3>
                        <p style="color: #94a3b8; margin: 0.6rem 0 0 0; font-size: 1.05rem;">{decision['summary']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # High Level Metrics Row
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value" style="color: #38bdf8;">{parsed_resume_data['name']}</div>
                        <div class="metric-label">Candidate</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value" style="color: #818cf8;">{match_results['composite_skills_match_score']}%</div>
                        <div class="metric-label">Skills Match</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{parsed_resume_data['years_experience']} yrs</div>
                        <div class="metric-label">Experience</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{parsed_resume_data['education']}</div>
                        <div class="metric-label">Degree</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m5:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{parsed_resume_data['project_count']}</div>
                        <div class="metric-label">Projects</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m6:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{parsed_resume_data['github_activity']}</div>
                        <div class="metric-label">GitHub Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Detailed Analysis Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["🔍 Skill Gap & Overlap", "📊 Semantic & Experience Breakdown", "💡 AI Insights & Feedback", "📄 Extracted Resume Text"])
                
                with tab1:
                    col_s1, col_s2 = st.columns([1, 1], gap="medium")
                    
                    with col_s1:
                        st.markdown(f"#### ✅ Matched Required Skills ({len(match_results['matched_skills'])})")
                        if match_results["matched_skills"]:
                            pills = "".join([f'<span class="badge-match">{s}</span>' for s in match_results["matched_skills"]])
                            st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)
                        else:
                            st.info("No direct skill matches found between resume and job description.")
                            
                        st.markdown(f"<br>#### 🌟 Additional Candidate Skills ({len(match_results['extra_skills'])})", unsafe_allow_html=True)
                        if match_results["extra_skills"]:
                            pills_extra = "".join([f'<span class="badge-extra">{s}</span>' for s in match_results["extra_skills"][:15]])
                            st.markdown(f"<div>{pills_extra}</div>", unsafe_allow_html=True)
                        else:
                            st.caption("None detected")
                            
                    with col_s2:
                        st.markdown(f"#### ❌ Missing JD Skills / Skill Gaps ({len(match_results['missing_skills'])})")
                        if match_results["missing_skills"]:
                            pills_miss = "".join([f'<span class="badge-missing">{s}</span>' for s in match_results["missing_skills"]])
                            st.markdown(f"<div>{pills_miss}</div>", unsafe_allow_html=True)
                        else:
                            st.success("🎯 Perfect Match! No required skills are missing.")
                            
                        # Skill Match Progress Chart
                        total_req = len(jd_data["required_skills"])
                        matched_cnt = len(match_results["matched_skills"])
                        miss_cnt = len(match_results["missing_skills"])
                        
                        if total_req > 0:
                            fig_donut = go.Figure(data=[go.Pie(
                                labels=['Matched Skills', 'Missing Skills'],
                                values=[matched_cnt, miss_cnt],
                                hole=.6,
                                marker_colors=['#10b981', '#ef4444'],
                                textinfo='label+percent'
                            )])
                            fig_donut.update_layout(
                                showlegend=False,
                                height=220,
                                margin=dict(l=10, r=10, t=20, b=10),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color="#cbd5e1")
                            )
                            st.plotly_chart(fig_donut, use_container_width=True)
                            
                with tab2:
                    col_b1, col_b2 = st.columns([1, 1], gap="medium")
                    with col_b1:
                        st.markdown("#### 📈 Compatibility Breakdown")
                        st.write(f"**Direct Skill Overlap:** `{match_results['skill_score_pct']}%`")
                        st.progress(float(match_results['skill_score_pct']) / 100.0)
                        
                        st.write(f"**TF-IDF Semantic Text Similarity:** `{match_results['tfidf_similarity_pct']}%`")
                        st.progress(float(match_results['tfidf_similarity_pct']) / 100.0)
                        
                        st.write(f"**Composite Skills Match Score:** `{match_results['composite_skills_match_score']}%`")
                        st.progress(float(match_results['composite_skills_match_score']) / 100.0)
                        
                    with col_b2:
                        st.markdown("#### ⏳ Experience & Degree Verification")
                        st.write(f"**Candidate Experience:** `{parsed_resume_data['years_experience']} years`")
                        st.write(f"**JD Experience Requirement:** `{jd_data['min_experience']} years`")
                        st.write(f"**Experience Check:** `{match_results['experience_status']}`")
                        
                        st.markdown("---")
                        st.write(f"**Candidate Education:** `{parsed_resume_data['education']}`")
                        st.write(f"**Detected Links:**")
                        for k, v in parsed_resume_data['links'].items():
                            if v:
                                st.markdown(f"- **{k.capitalize()}:** [{v}]({v if v.startswith('http') else 'https://' + v})")

                with tab3:
                    st.markdown("#### 💡 Candidate Insights & Hiring Feedback")
                    col_fb1, col_fb2 = st.columns(2)
                    with col_fb1:
                        st.markdown("##### 🟢 Key Strengths")
                        for st_item in decision["strengths"]:
                            st.markdown(f"- {st_item}")
                    with col_fb2:
                        st.markdown("##### 🟡 Improvement Areas / Gaps")
                        for imp_item in decision["improvements"]:
                            st.markdown(f"- {imp_item}")
                            
                with tab4:
                    st.markdown("#### 📝 Raw Extracted Resume Text")
                    st.text_area("Extracted Content:", value=parsed_resume_data["raw_text"], height=300)

# ==========================================
# WORKFLOW 2: BATCH SCREENING (LEADERBOARD)
# ==========================================
elif app_mode == "📊 Batch Resume Screening (Leaderboard)":
    st.subheader("1. Job Description & Multiple Resumes Upload")
    
    jd_batch_input = st.text_area(
        "Paste Job Description:",
        value="",
        height=180,
        placeholder="Paste Job Description requirements, required skills, degree, and years of experience here..."
    )
    
    batch_files = st.file_uploader(
        "Upload Multiple Resumes (PDF / DOCX / TXT):",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload multiple candidate resumes at once to rank them automatically."
    )
    
    if st.button("🚀 Screen & Rank All Candidates", type="primary", use_container_width=True):
        if not jd_batch_input.strip():
            st.warning("⚠️ Please provide a Job Description.")
        elif not batch_files:
            st.warning("⚠️ Please upload at least one resume file (PDF, DOCX, or TXT).")
        else:
            candidates_to_process = []
            
            for file_obj in batch_files:
                parsed = parse_resume(file_obj)
                candidates_to_process.append(parsed)
                    
            jd_batch_data = parse_job_description(jd_batch_input)
            
            # Progress Bar & Processing
            progress_bar = st.progress(0)
            results_list = []
            
            for idx, cand in enumerate(candidates_to_process):
                match_res = evaluate_match(cand, jd_batch_data)
                features = {
                    "years_experience": cand["years_experience"],
                    "skills_match_score": match_res["composite_skills_match_score"],
                    "education_level": cand["education"],
                    "project_count": cand["project_count"],
                    "resume_length": cand["resume_length"],
                    "github_activity": cand["github_activity"]
                }
                decision = predict_candidate_shortlist(features, threshold=shortlist_threshold)
                
                results_list.append({
                    "Candidate Name": cand["name"],
                    "File": cand["filename"],
                    "Status": "✅ Shortlisted" if decision["is_shortlisted"] else "❌ Not Shortlisted",
                    "Probability (%)": decision["probability"],
                    "Skills Match (%)": match_res["composite_skills_match_score"],
                    "Matched Skills": ", ".join(match_res["matched_skills"]),
                    "Missing Skills": ", ".join(match_res["missing_skills"]),
                    "Experience (Yrs)": cand["years_experience"],
                    "Education": cand["education"],
                    "Projects": cand["project_count"],
                    "GitHub Score": cand["github_activity"],
                    "Email": cand["email"]
                })
                progress_bar.progress((idx + 1) / len(candidates_to_process))
                
            df_results = pd.DataFrame(results_list).sort_values(by=["Probability (%)", "Skills Match (%)"], ascending=False).reset_index(drop=True)
            df_results.index += 1 # 1-based ranking
            
            st.success(f"🎉 Successfully screened {len(df_results)} candidate profiles!")
            
            # Summary Metrics
            shortlisted_cnt = (df_results["Status"] == "✅ Shortlisted").sum()
            rejected_cnt = len(df_results) - shortlisted_cnt
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Resumes Screened", len(df_results))
            c2.metric("Shortlisted Candidates", shortlisted_cnt, delta=f"{(shortlisted_cnt/len(df_results))*100:.0f}% Pass Rate")
            c3.metric("Not Shortlisted", rejected_cnt)
            c4.metric("Average Skill Match", f"{df_results['Skills Match (%)'].mean():.1f}%")
            
            st.markdown("### 🏆 Candidate Ranking Leaderboard")
            
            # Filter options
            filter_mode = st.radio("Filter Leaderboard:", ["All Candidates", "Shortlisted Only", "Not Shortlisted Only"], horizontal=True)
            
            display_df = df_results.copy()
            if filter_mode == "Shortlisted Only":
                display_df = display_df[display_df["Status"] == "✅ Shortlisted"]
            elif filter_mode == "Not Shortlisted Only":
                display_df = display_df[display_df["Status"] == "❌ Not Shortlisted"]
                
            st.dataframe(
                display_df[[
                    "Candidate Name", "Status", "Probability (%)", "Skills Match (%)",
                    "Experience (Yrs)", "Education", "Projects", "GitHub Score", "Email"
                ]],
                use_container_width=True
            )
            
            # Visual Analytics
            st.markdown("### 📊 Comparative Candidate Analytics")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_bar = px.bar(
                    df_results,
                    x="Candidate Name",
                    y="Probability (%)",
                    color="Status",
                    color_discrete_map={"✅ Shortlisted": "#10b981", "❌ Not Shortlisted": "#ef4444"},
                    title="Shortlisting Probability by Candidate",
                    text="Probability (%)"
                )
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#cbd5e1"))
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with col_chart2:
                fig_scatter = px.scatter(
                    df_results,
                    x="Skills Match (%)",
                    y="Experience (Yrs)",
                    size="Probability (%)",
                    color="Status",
                    hover_name="Candidate Name",
                    color_discrete_map={"✅ Shortlisted": "#10b981", "❌ Not Shortlisted": "#ef4444"},
                    title="Skills Match vs. Experience"
                )
                fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#cbd5e1"))
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            # Download CSV
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Shortlist Report (CSV)",
                data=csv_data,
                file_name="candidate_shortlist_report.csv",
                mime="text/csv",
                use_container_width=True
            )
