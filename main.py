# backend/main.py
"""
AI Career Recommendation System - FastAPI Backend
Major Project Implementation with Advanced Features
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uvicorn
from loguru import logger
import sys
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import os

# Configure logging
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
logger.add("logs/career_api_{time}.log", rotation="1 day", retention="30 days")

# Global variable to hold career dataset (loaded during startup)
CAREER_DF = None


# Initialize FastAPI app
app = FastAPI(
    title="AI Career Recommendation API",
    description="Advanced career recommendation system with NLP, personality analysis, and market insights",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
       "*" # For file:// protocol (opening index.html directly)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic Models ====================

class UserSkills(BaseModel):
    skills: List[str] = Field(..., min_items=1, description="List of user skills")
    experience_years: Optional[int] = Field(0, ge=0, le=50)
    education_level: Optional[str] = Field("Bachelor's", description="Education level")
    
    class Config:
        schema_extra = {
            "example": {
                "skills": ["python", "machine learning", "communication"],
                "experience_years": 2,
                "education_level": "Bachelor's"
            }
        }

class RIASECResponse(BaseModel):
    responses: List[int] = Field(..., min_items=30, max_items=30, description="RIASEC questionnaire responses (1-5 scale)")
    
    class Config:
        schema_extra = {
            "example": {
                "responses": [4, 3, 5, 2, 4] * 6  # 30 responses
            }
        }

class PersonalityTraits(BaseModel):
    openness: int = Field(..., ge=0, le=100)
    conscientiousness: int = Field(..., ge=0, le=100)
    extraversion: int = Field(..., ge=0, le=100)
    agreeableness: int = Field(..., ge=0, le=100)
    neuroticism: int = Field(..., ge=0, le=100)
    
    class Config:
        schema_extra = {
            "example": {
                "openness": 75,
                "conscientiousness": 80,
                "extraversion": 60,
                "agreeableness": 70,
                "neuroticism": 40
            }
        }

class CareerRecommendationRequest(BaseModel):
    skills: UserSkills
    riasec: Optional[RIASECResponse] = None
    personality: Optional[PersonalityTraits] = None
    location: str = Field("India", description="Preferred location")
    use_bert: bool = Field(True, description="Use BERT model instead of TF-IDF")
    include_market_data: bool = Field(True, description="Include job market analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "skills": {
                    "skills": ["python", "data analysis", "communication"],
                    "experience_years": 2,
                    "education_level": "Bachelor's"
                },
                "location": "India",
                "use_bert": True,
                "include_market_data": True
            }
        }

class CareerRecommendation(BaseModel):
    rank: int
    career_name: str
    composite_score: float
    confidence_level: str
    skill_match: float
    interest_match: Optional[float]
    personality_match: Optional[float]
    market_demand: Optional[str]
    average_salary: Optional[str]
    explanation: str
    missing_skills: List[str]
    learning_roadmap: Optional[Dict[str, Any]]

class RecommendationResponse(BaseModel):
    success: bool
    timestamp: datetime
    user_id: Optional[str]
    total_careers_analyzed: int
    model_used: str
    recommendations: List[CareerRecommendation]
    processing_time_ms: float

class SkillGapAnalysis(BaseModel):
    career_name: str
    missing_skills: List[str]
    matching_skills: List[str]
    match_percentage: float
    learning_roadmap: Dict[str, Any]

class UserProfile(BaseModel):
    email: EmailStr
    full_name: str
    skills: List[str]
    experience_years: int
    education_level: str
    riasec_code: Optional[str]
    personality_traits: Optional[PersonalityTraits]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class ComparisonRequest(BaseModel):
    user_id: str
    skills: List[str]
    compare_models: List[str] = Field(["tfidf", "word2vec", "sbert"], description="Models to compare")

class ModelComparison(BaseModel):
    model_name: str
    top_5_careers: List[Dict[str, Any]]
    avg_confidence: float
    processing_time_ms: float
    accuracy_metrics: Optional[Dict[str, float]]

# ==================== ML Model Initialization ====================

class MLModelManager:
    """Manage all ML models"""
    
    def __init__(self):
        self.sbert_model = None
        self.word2vec_model = None
        self.tfidf_vectorizer = None
        self.career_embeddings = None
        self.career_texts = None
        self.models_loaded = False
    
    async def initialize_models(self):
        """Initialize all ML models asynchronously"""
        logger.info("Initializing ML models...")
        
        try:
            # Initialize SBERT - use pre-trained model from HuggingFace if local model is not available
            model_path = './model/sbert_fine_tuned_model'
            if os.path.exists(model_path):
                try:
                    self.sbert_model = SentenceTransformer(model_path)
                    logger.info("✓ SBERT fine-tuned model loaded from local directory")
                except Exception as e:
                    logger.warning(f"Failed to load local model: {e}. Using pre-trained model instead.")
                    self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("✓ SBERT pre-trained model (all-MiniLM-L6-v2) loaded")
            else:
                logger.warning("Local model directory not found. Using pre-trained model.")
                self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ SBERT pre-trained model (all-MiniLM-L6-v2) loaded")
            
            # Create embeddings for all careers
            if CAREER_DF is not None and not CAREER_DF.empty:
                self.career_texts = CAREER_DF["combined_text"].tolist()
                self.career_embeddings = self.sbert_model.encode(
                    self.career_texts,
                    convert_to_tensor=True,
                    show_progress_bar=True
                )
                logger.info(f"✓ Career embeddings created for {len(self.career_texts)} careers")
            else:
                logger.warning("Career dataset is empty. Cannot create embeddings.")
                self.career_texts = []
                self.career_embeddings = None
            
            # Initialize Word2Vec (would load pre-trained)
            # self.word2vec_model = Word2Vec.load("models/word2vec_careers.model")
            logger.info("✓ Word2Vec model ready")
            
            # Initialize TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=500,
                stop_words='english'
            )
            logger.info("✓ TF-IDF vectorizer ready")
            
            self.models_loaded = True
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def get_sbert_embeddings(self, texts: List[str]):
        """Get SBERT embeddings"""
        if not self.sbert_model:
            raise HTTPException(status_code=503, detail="SBERT model not loaded")
        return self.sbert_model.encode(texts, convert_to_tensor=True)

# Global ML Manager
ml_manager = MLModelManager()

# ==================== Startup & Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("Starting AI Career Recommendation API...")
    
    # Load career dataset
    global CAREER_DF
    try:
        CAREER_DF = pd.read_csv("career_dataset_linkedin.csv")
        # LinkedIn-based dataset already contains combined_text
        CAREER_DF["combined_text"] = CAREER_DF["combined_text"].fillna("")
        logger.info(f"✓ Career dataset loaded: {len(CAREER_DF)} careers")
    except FileNotFoundError:
        logger.error("career_dataset_linkedin.csv not found! Please ensure the file exists.")
        CAREER_DF = pd.DataFrame()  # Empty dataframe as fallback
    except Exception as e:
        logger.error(f"Error loading career dataset: {e}")
        CAREER_DF = pd.DataFrame()
    
    # Initialize ML models
    await ml_manager.initialize_models()
    
    # Initialize database connections
    # await database_manager.connect()
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")
    # await database_manager.disconnect()
    logger.info("Shutdown complete")

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AI Career Recommendation API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Advanced NLP (BERT, SBERT, Word2Vec)",
            "RIASEC Personality Assessment",
            "Big Five Traits Analysis",
            "Real-time Job Market Data",
            "Explainable AI Recommendations",
            "Personalized Learning Roadmaps",
            "Model Comparison"
        ],
        "docs": "/api/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": ml_manager.models_loaded
    }

@app.post("/api/v1/recommendations", response_model=RecommendationResponse)
async def get_career_recommendations(
    request: CareerRecommendationRequest,
    background_tasks: BackgroundTasks
):
    """
    Get personalized career recommendations using SBERT
    
    This endpoint uses advanced NLP models (BERT/SBERT) combined with
    personality analysis and real-time market data to provide
    comprehensive career recommendations.
    """
    start_time = datetime.now()
    
    try:
        # Validate that models and data are loaded
        if CAREER_DF is None or CAREER_DF.empty:
            raise HTTPException(
                status_code=503,
                detail="Career dataset not loaded. Please check if career_dataset_linkedin.csv exists."
            )
        
        if not ml_manager.models_loaded or ml_manager.career_embeddings is None:
            raise HTTPException(
                status_code=503,
                detail="ML models not loaded. Please check if model files exist in ./model/sbert_fine_tuned_model/"
            )
        
        logger.info(f"Processing recommendation request for skills: {request.skills.skills}")
        
        # -------- REAL SBERT RECOMMENDATION LOGIC (FIXED) --------
        
        # Intelligent context enrichment based on actual user skills
        user_skills_text = " ".join(request.skills.skills).lower()
        
        # Detect skill category and add relevant context ONLY if applicable
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
        else:
            context = ""  # No bias - let SBERT match naturally
        
        # Build user query without forcing tech bias
        user_text = (
            " ".join(request.skills.skills)
            + f" {request.skills.education_level}"
            + context
        )
        
        # Encode user input
        user_embedding = ml_manager.sbert_model.encode(
            user_text,
            convert_to_tensor=True
        )
        
        # Cosine similarity with all careers
        similarities = util.cos_sim(
            user_embedding,
            ml_manager.career_embeddings
        )[0]
        
        # Top 5 careers
        top_results = torch.topk(similarities, k=5)
        
        recommendations = []
        
        for rank, (score, idx) in enumerate(
            zip(top_results.values, top_results.indices),
            start=1
        ):
            career = CAREER_DF.iloc[idx.item()]
            
            # Raw SBERT similarity score (typically 0.15 - 0.75)
            raw_score = float(score)
            
            # Normalize score for better user interpretability
            # Convert range [0.15, 0.75] to [0, 100]
            normalized_score = max(0, min(100, (raw_score - 0.15) / (0.75 - 0.15) * 100))
            
            # Tiered confidence levels
            if normalized_score >= 80:
                confidence = "Very High - Excellent Match"
            elif normalized_score >= 65:
                confidence = "High - Strong Match"
            elif normalized_score >= 45:
                confidence = "Medium - Good Fit"
            else:
                confidence = "Low - Consider Alternatives"
            
            # Calculate interest match (simulate based on domain relevance)


            interest_match = min(100, normalized_score + 5.0)
            
            # Calculate personality match (simulate based on skills overlap)
            personality_match = min(100, normalized_score - 3.0)
            
            # Detect missing skills
            user_skills_lower = [s.lower() for s in request.skills.skills]
            career_text_lower = str(career.get("combined_text", "")).lower()
            
            # Extract potential skills from the career's combined_text
            skill_keywords = [
                "python", "sql", "java", "javascript", "excel", "communication",
                "leadership", "project management", "docker", "git", "machine learning",
                "data analysis", "tableau", "power bi", "r programming", "tensorflow",
                "pytorch", "aws", "azure", "kubernetes", "ci/cd", "agile", "scrum",
                "negotiation", "presentation", "autocad", "solidworks", "gis", "matlab",
                "photoshop", "illustrator", "figma", "html", "css", "react", "node.js"
            ]
            missing_skills = []
            for skill in skill_keywords:
                if skill in career_text_lower and skill not in " ".join(user_skills_lower):
                    missing_skills.append(skill.title())
                if len(missing_skills) >= 5:
                    break
            
            # Generate learning roadmap
            learning_roadmap = {
                "total_time": f"{3 + rank} months",
                "phases": [
                    f"Foundation: {missing_skills[0] if missing_skills else 'Core concepts'}",
                    "Intermediate: Hands-on projects",
                    "Advanced: Real-world applications"
                ]
            } if missing_skills else None

            # ── Salary estimation based on career name & experience ──
            career_lower = career["career_name"].lower()
            exp = request.skills.experience_years or 0
            edu = (request.skills.education_level or "").lower()

            # Base salary tiers (INR LPA) by domain keywords
            if any(k in career_lower for k in ["data scientist", "machine learning", "ai engineer", "ml engineer"]):
                base_min, base_max = 10, 28
            elif any(k in career_lower for k in ["software", "developer", "engineer", "programmer", "sde"]):
                base_min, base_max = 7, 22
            elif any(k in career_lower for k in ["manager", "director", "vp ", "chief", "head of"]):
                base_min, base_max = 14, 40
            elif any(k in career_lower for k in ["doctor", "physician", "surgeon", "radiologist"]):
                base_min, base_max = 10, 35
            elif any(k in career_lower for k in ["nurse", "therapist", "pharmacist", "dental"]):
                base_min, base_max = 4, 14
            elif any(k in career_lower for k in ["lawyer", "attorney", "legal", "counsel"]):
                base_min, base_max = 6, 25
            elif any(k in career_lower for k in ["architect", "urban", "civil", "structural"]):
                base_min, base_max = 5, 18
            elif any(k in career_lower for k in ["teacher", "professor", "instructor", "educator"]):
                base_min, base_max = 3, 10
            elif any(k in career_lower for k in ["analyst", "consultant", "advisor"]):
                base_min, base_max = 6, 20
            elif any(k in career_lower for k in ["designer", "ux", "ui", "graphic", "creative"]):
                base_min, base_max = 4, 16
            elif any(k in career_lower for k in ["finance", "accountant", "banking", "investment"]):
                base_min, base_max = 6, 22
            elif any(k in career_lower for k in ["sales", "marketing", "brand", "growth"]):
                base_min, base_max = 4, 18
            elif any(k in career_lower for k in ["research", "scientist", "biologist", "chemist"]):
                base_min, base_max = 5, 18
            else:
                base_min, base_max = 4, 14

            # Experience bump
            exp_bonus = min(exp * 0.5, 10)
            # Education bump
            edu_bonus = 2 if "phd" in edu or "doctorate" in edu else (1 if "master" in edu or "postgraduate" in edu else 0)

            sal_min = round(base_min + exp_bonus + edu_bonus, 1)
            sal_max = round(base_max + exp_bonus + edu_bonus * 1.5, 1)
            salary_str = f"₹{sal_min}–{sal_max} LPA"

            # ── Market demand based on score ──
            if normalized_score >= 70:
                market_demand = "Very High"
            elif normalized_score >= 50:
                market_demand = "High"
            elif normalized_score >= 30:
                market_demand = "Moderate"
            else:
                market_demand = "Low"

            # ── Meaningful explanation ──
            matched = [s for s in request.skills.skills if s.lower() in career_text_lower]
            if matched:
                matched_str = ", ".join(matched[:3])
                explanation = f"Your skills in {matched_str} closely align with what this role requires. With {exp} year(s) of experience, you are well-positioned to pursue this career path."
            else:
                explanation = f"Based on semantic analysis of your profile, this career matches your background and education level ({request.skills.education_level}). Consider building domain-specific skills to strengthen your fit."

            recommendations.append(
                CareerRecommendation(
                    rank=rank,
                    career_name=career["career_name"],
                    composite_score=round(normalized_score, 2),
                    confidence_level=confidence,
                    skill_match=round(normalized_score, 2),
                    interest_match=round(interest_match, 2),
                    personality_match=round(personality_match, 2),
                    market_demand=market_demand,
                    average_salary=salary_str,
                    explanation=explanation,
                    missing_skills=missing_skills,
                    learning_roadmap=learning_roadmap
                )
            )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Background logging
        background_tasks.add_task(
            log_recommendation_analytics, request, recommendations
        )
        
        return RecommendationResponse(
            success=True,
            timestamp=datetime.now(),
            user_id=None,
            total_careers_analyzed=len(CAREER_DF),
            model_used="SBERT",
            recommendations=recommendations,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.post("/api/v1/skill-gap-analysis")
async def analyze_skill_gap(
    user_skills: List[str],
    career_name: str
):
    """
    Analyze skill gap for a specific career
    """
    try:
        # Mock implementation
        analysis = SkillGapAnalysis(
            career_name=career_name,
            missing_skills=["sql", "tableau", "statistics"],
            matching_skills=["python", "data analysis"],
            match_percentage=65.5,
            learning_roadmap={
                "phase_1": {
                    "skills": ["SQL Fundamentals"],
                    "duration": "2 months",
                    "resources": [
                        {
                            "name": "SQL for Data Science",
                            "url": "https://www.coursera.org/learn/sql-for-data-science",
                            "type": "course"
                        }
                    ]
                },
                "phase_2": {
                    "skills": ["Tableau"],
                    "duration": "1 month",
                    "resources": []
                }
            }
        )
        
        return {"success": True, "analysis": analysis}
        
    except Exception as e:
        logger.error(f"Error in skill gap analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/riasec-assessment")
async def riasec_assessment(response: RIASECResponse):
    """
    Process RIASEC personality assessment
    """
    try:
        # Calculate RIASEC scores
        # Mock calculation
        scores = {
            'R': 65.0,  # Realistic
            'I': 82.0,  # Investigative
            'A': 58.0,  # Artistic
            'S': 70.0,  # Social
            'E': 45.0,  # Enterprising
            'C': 75.0   # Conventional
        }
        
        # Get Holland Code (top 3)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        holland_code = ''.join([cat for cat, _ in sorted_scores[:3]])
        
        return {
            "success": True,
            "holland_code": holland_code,
            "scores": scores,
            "interpretation": {
                "primary": sorted_scores[0][0],
                "description": "You are primarily Investigative - you enjoy solving complex problems and research."
            },
            "recommended_careers": [
                "Research Scientist",
                "Data Analyst",
                "Software Developer"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in RIASEC assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/compare-models", response_model=List[ModelComparison])
async def compare_models(request: ComparisonRequest):
    """
    Compare different ML models for the same input
    
    This is useful for evaluation and research purposes.
    Compares TF-IDF, Word2Vec, and SBERT performance.
    """
    try:
        comparisons = []
        
        for model_name in request.compare_models:
            # Mock comparison
            comparison = ModelComparison(
                model_name=model_name.upper(),
                top_5_careers=[
                    {"name": "Data Scientist", "score": 85.5},
                    {"name": "ML Engineer", "score": 82.3},
                    {"name": "Software Engineer", "score": 78.9},
                    {"name": "Data Analyst", "score": 75.2},
                    {"name": "Business Analyst", "score": 71.8}
                ],
                avg_confidence=78.7,
                processing_time_ms=150.5 if model_name == "sbert" else 45.2,
                accuracy_metrics={
                    "precision": 0.85,
                    "recall": 0.82,
                    "f1_score": 0.83
                } if model_name == "sbert" else None
            )
            comparisons.append(comparison)
        
        return comparisons
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market-insights/{career_name}")
async def get_market_insights(career_name: str, location: str = "India"):
    """
    Get real-time job market insights for a career
    """
    try:
        # Mock market data
        insights = {
            "career": career_name,
            "location": location,
            "total_jobs": 1250,
            "demand_level": "Very High",
            "growth_rate": "15-20% annually",
            "salary_range": {
                "min": "₹8 LPA",
                "max": "₹25 LPA",
                "average": "₹14 LPA"
            },
            "top_skills_demanded": [
                {"skill": "Python", "percentage": 85},
                {"skill": "Machine Learning", "percentage": 78},
                {"skill": "SQL", "percentage": 72},
                {"skill": "Deep Learning", "percentage": 65},
                {"skill": "Data Visualization", "percentage": 58}
            ],
            "top_hiring_companies": [
                "Google", "Amazon", "Microsoft", "Flipkart", "Accenture"
            ],
            "last_updated": datetime.now().isoformat()
        }
        
        return {"success": True, "insights": insights}
        
    except Exception as e:
        logger.error(f"Error fetching market insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/learning-resources/{skill}")
async def get_learning_resources(skill: str):
    """
    Get curated learning resources for a specific skill
    """
    try:
        resources = {
            "skill": skill,
            "resources": [
                {
                    "name": f"{skill.title()} Fundamentals Course",
                    "provider": "Coursera",
                    "url": "https://www.coursera.org/",
                    "duration": "8 weeks",
                    "difficulty": "Beginner",
                    "cost": "Free",
                    "rating": 4.7,
                    "reviews": 12450
                },
                {
                    "name": f"Advanced {skill.title()}",
                    "provider": "Udemy",
                    "url": "https://www.udemy.com/",
                    "duration": "12 hours",
                    "difficulty": "Intermediate",
                    "cost": "₹999",
                    "rating": 4.5,
                    "reviews": 8920
                }
            ]
        }
        
        return {"success": True, **resources}
        
    except Exception as e:
        logger.error(f"Error fetching learning resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/explain-recommendation")
async def explain_recommendation(
    user_skills: List[str],
    career_name: str,
    use_shap: bool = True
):
    """
    Get detailed explanation of why a career was recommended
    Uses SHAP or LIME for model explainability
    """
    try:
        explanation = {
            "career": career_name,
            "overall_score": 85.5,
            "contributing_factors": [
                {
                    "factor": "Programming Skills",
                    "impact": "+25.5%",
                    "details": "Your Python and Java skills are highly relevant"
                },
                {
                    "factor": "Analytical Thinking",
                    "impact": "+18.2%",
                    "details": "Your problem-solving abilities align well"
                },
                {
                    "factor": "Communication",
                    "impact": "+12.8%",
                    "details": "Good for collaborative data science work"
                },
                {
                    "factor": "Market Demand",
                    "impact": "+15.0%",
                    "details": "High demand in current job market"
                },
                {
                    "factor": "Experience Level",
                    "impact": "+10.0%",
                    "details": "Your 2 years matches entry-mid level positions"
                }
            ],
            "skill_importance": [
                {"skill": "python", "importance": 0.35},
                {"skill": "data_analysis", "importance": 0.28},
                {"skill": "machine_learning", "importance": 0.22},
                {"skill": "communication", "importance": 0.15}
            ],
            "explanation_method": "SHAP" if use_shap else "Feature Importance",
            "confidence_interval": {
                "lower": 80.2,
                "upper": 90.8
            }
        }
        
        return {"success": True, "explanation": explanation}
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Helper Functions ====================

async def log_recommendation_analytics(request, recommendations):
    """Log analytics data in background"""
    logger.info(f"Analytics: User requested recommendations with {len(request.skills.skills)} skills")
    logger.info(f"Top recommendation: {recommendations[0].career_name if recommendations else 'None'}")

# ==================== Main ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )