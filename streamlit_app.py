"""
AI Career Recommendation System - Streamlit App
Streamlit Cloud Deployment Version
"""

import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import os
from datetime import datetime
from typing import List, Dict, Any
import json

# Page configuration
st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .career-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .skill-badge {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .missing-skill {
        background-color: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'sbert_model' not in st.session_state:
    st.session_state.sbert_model = None
if 'career_df' not in st.session_state:
    st.session_state.career_df = None
if 'career_embeddings' not in st.session_state:
    st.session_state.career_embeddings = None

@st.cache_resource
def load_model():
    """Load the SBERT model"""
    try:
        model_path = './model/sbert_fine_tuned_model'
        if os.path.exists(model_path):
            model = SentenceTransformer(model_path)
            st.success("✓ Local fine-tuned SBERT model loaded")
        else:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            st.info("✓ Pre-trained SBERT model loaded (all-MiniLM-L6-v2)")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_career_dataset():
    """Load the career dataset"""
    try:
        df = pd.read_csv('career_dataset_linkedin.csv')
        st.success(f"✓ Career dataset loaded ({len(df)} careers)")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def compute_career_embeddings(model, df):
    """Compute embeddings for all careers"""
    try:
        with st.spinner("Computing career embeddings..."):
            career_texts = []
            for _, row in df.iterrows():
                text = f"{row.get('job_title', '')} {row.get('job_description', '')} {row.get('skills', '')}"
                career_texts.append(text)
            
            embeddings = model.encode(career_texts, convert_to_tensor=True, show_progress_bar=False)
            st.success("✓ Career embeddings computed")
            return embeddings
    except Exception as e:
        st.error(f"Error computing embeddings: {e}")
        return None

def get_recommendations(user_skills: List[str], education_level: str, experience_years: int, model, df, embeddings, top_k: int = 5):
    """Get career recommendations based on user input"""
    try:
        # Build user context
        user_skills_text = " ".join(user_skills).lower()
        
        # Detect skill category and add relevant context
        context = ""
        if any(word in user_skills_text for word in ["science", "research", "biology", "chemistry", "physics", "laboratory"]):
            context = " research scientific analysis experimentation"
        elif any(word in user_skills_text for word in ["history", "political", "policy", "government", "governance", "social"]):
            context = " analysis policy research governance writing"
        elif any(word in user_skills_text for word in ["art", "design", "creative", "graphic", "visual"]):
            context = " creative design visual arts"
        elif any(word in user_skills_text for word in ["programming", "coding", "java", "python", "software", "developer", "engineering"]):
            context = " software development programming technology"
        elif any(word in user_skills_text for word in ["business", "management", "marketing", "sales"]):
            context = " business management strategy"
        elif any(word in user_skills_text for word in ["teaching", "education", "training"]):
            context = " education teaching learning"
        
        # Build user query
        user_text = " ".join(user_skills) + f" {education_level}" + context
        
        # Encode user input
        user_embedding = model.encode(user_text, convert_to_tensor=True)
        
        # Compute similarities
        similarities = util.cos_sim(user_embedding, embeddings)[0]
        
        # Get top results
        top_results = torch.topk(similarities, k=min(top_k, len(df)))
        
        recommendations = []
        for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices), start=1):
            career = df.iloc[idx.item()]
            raw_score = float(score)
            
            # Normalize score (0.15-0.75 -> 0-100)
            normalized_score = max(0, min(100, (raw_score - 0.15) / (0.75 - 0.15) * 100))
            
            # Determine confidence level
            if normalized_score >= 70:
                confidence = "High"
            elif normalized_score >= 50:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Extract career skills
            career_skills_str = str(career.get('skills', ''))
            career_skills = [s.strip().lower() for s in career_skills_str.split(',') if s.strip()]
            
            # Find missing skills
            user_skills_lower = [s.lower() for s in user_skills]
            missing_skills = [s for s in career_skills if s not in user_skills_lower][:5]
            matched_skills = [s for s in user_skills_lower if s in career_skills]
            
            recommendations.append({
                'rank': rank,
                'career_name': career.get('job_title', 'Unknown'),
                'company': career.get('company_name', 'N/A'),
                'location': career.get('location', 'N/A'),
                'score': normalized_score,
                'raw_score': raw_score,
                'confidence': confidence,
                'description': career.get('job_description', 'No description available')[:300] + "...",
                'skills': career.get('skills', 'N/A'),
                'missing_skills': missing_skills[:5],
                'matched_skills': matched_skills,
                'match_percentage': len(matched_skills) / max(len(user_skills_lower), 1) * 100
            })
        
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">🎯 AI-Powered Career Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discover your perfect career path with advanced AI technology</div>', unsafe_allow_html=True)
    
    # Initialize models and data
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI models and career data..."):
            st.session_state.sbert_model = load_model()
            st.session_state.career_df = load_career_dataset()
            
            if st.session_state.sbert_model and st.session_state.career_df is not None:
                st.session_state.career_embeddings = compute_career_embeddings(
                    st.session_state.sbert_model,
                    st.session_state.career_df
                )
                if st.session_state.career_embeddings is not None:
                    st.session_state.model_loaded = True
    
    if not st.session_state.model_loaded:
        st.error("⚠️ Failed to load required resources. Please check the model and dataset files.")
        return
    
    # Sidebar - User Input
    st.sidebar.header("📝 Your Profile")
    
    # Skills input
    st.sidebar.subheader("Skills")
    skills_input = st.sidebar.text_area(
        "Enter your skills (one per line):",
        placeholder="python\ndata analysis\ncommunication\nproject management",
        height=150
    )
    
    # Parse skills
    user_skills = [skill.strip() for skill in skills_input.split('\n') if skill.strip()]
    
    # Education level
    education_level = st.sidebar.selectbox(
        "Education Level:",
        ["High School", "Associate", "Bachelor's", "Master's", "PhD", "Other"]
    )
    
    # Experience years
    experience_years = st.sidebar.slider(
        "Years of Experience:",
        min_value=0,
        max_value=30,
        value=2,
        step=1
    )
    
    # Number of recommendations
    top_k = st.sidebar.slider(
        "Number of Recommendations:",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )
    
    # Get recommendations button
    if st.sidebar.button("🔍 Get Recommendations", type="primary", use_container_width=True):
        if not user_skills:
            st.sidebar.error("⚠️ Please enter at least one skill!")
        else:
            with st.spinner("Analyzing your profile and finding best matches..."):
                recommendations = get_recommendations(
                    user_skills=user_skills,
                    education_level=education_level,
                    experience_years=experience_years,
                    model=st.session_state.sbert_model,
                    df=st.session_state.career_df,
                    embeddings=st.session_state.career_embeddings,
                    top_k=top_k
                )
                
                st.session_state.recommendations = recommendations
                st.session_state.user_skills = user_skills
    
    # Display recommendations
    if 'recommendations' in st.session_state and st.session_state.recommendations:
        st.markdown("---")
        st.header("🎯 Your Personalized Career Recommendations")
        
        # Display user profile summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Skills Provided", len(st.session_state.user_skills))
        with col2:
            st.metric("Education Level", education_level)
        with col3:
            st.metric("Experience", f"{experience_years} years")
        
        st.markdown("---")
        
        # Display each recommendation
        for rec in st.session_state.recommendations:
            with st.expander(
                f"#{rec['rank']} - {rec['career_name']} | Match: {rec['score']:.1f}% | Confidence: {rec['confidence']}",
                expanded=(rec['rank'] <= 3)
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Company:** {rec['company']}")
                    st.markdown(f"**Location:** {rec['location']}")
                    st.markdown(f"**Description:**")
                    st.write(rec['description'])
                
                with col2:
                    st.markdown(f"**Match Score:** {rec['score']:.1f}%")
                    st.markdown(f"**Confidence:** {rec['confidence']}")
                    st.markdown(f"**Skill Match:** {rec['match_percentage']:.0f}%")
                
                # Matched skills
                if rec['matched_skills']:
                    st.markdown("**✅ Your Matching Skills:**")
                    skills_html = "".join([f'<span class="skill-badge">{skill}</span>' for skill in rec['matched_skills'][:10]])
                    st.markdown(skills_html, unsafe_allow_html=True)
                
                # Missing skills
                if rec['missing_skills']:
                    st.markdown("**📚 Skills to Learn:**")
                    missing_html = "".join([f'<span class="skill-badge missing-skill">{skill}</span>' for skill in rec['missing_skills']])
                    st.markdown(missing_html, unsafe_allow_html=True)
                
                st.markdown("---")
    
    else:
        # Welcome message when no recommendations yet
        st.info("👈 Enter your skills and profile information in the sidebar to get personalized career recommendations!")
        
        # Display sample careers
        st.markdown("### 💼 Sample Careers in Our Database")
        if st.session_state.career_df is not None:
            sample_careers = st.session_state.career_df.sample(min(5, len(st.session_state.career_df)))
            for _, career in sample_careers.iterrows():
                st.markdown(f"- **{career.get('job_title', 'Unknown')}** at {career.get('company_name', 'N/A')}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🚀 AI-Powered Career Recommendation System | Powered by SBERT & Deep Learning"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
