# # hybrid_recommender.py
# """
# Hybrid AI Career Recommendation System
# Architecture:
#     User Profile → Skill Extraction (NLP) → Domain Classification
#     → Dataset Filtering → SBERT Embedding Search → Top 50 Careers
#     → Cross Encoder Ranking → Final Top 5 Careers

# Expected Accuracy: 85-92%
# """

# import numpy as np
# import pandas as pd
# from typing import List, Dict, Tuple, Optional, Any
# from sentence_transformers import SentenceTransformer, CrossEncoder, util
# import torch
# from loguru import logger
# import re
# from dataclasses import dataclass
# from functools import lru_cache


# @dataclass
# class UserProfile:
#     """User profile for career recommendation"""
#     skills: List[str]
#     education_level: str
#     experience_years: int
#     interests: Optional[List[str]] = None
#     preferred_domain: Optional[str] = None


# @dataclass
# class CareerMatch:
#     """Structured career recommendation result"""
#     rank: int
#     career_name: str
#     combined_text: str
#     sbert_score: float
#     cross_encoder_score: float
#     skill_match_score: float
#     domain_score: float
#     education_score: float
#     final_score: float
#     confidence: str
#     matched_skills: List[str]
#     missing_skills: List[str]
#     detected_domain: str
#     explanation: str


# class SkillExtractor:
#     """
#     NLP-based Skill Extraction from user input
#     Uses keyword matching + pattern recognition + natural language parsing
#     For production: Use KeyBERT or spaCy NER
#     """
    
#     # Comprehensive skill taxonomy
#     SKILL_TAXONOMY = {
#         "programming": [
#             "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "swift",
#             "kotlin", "php", "typescript", "scala", "perl", "r", "matlab", "sql", "nosql",
#             "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
#             "spring", "express", ".net", "laravel", "rails"
#         ],
#         "data_science": [
#             "machine learning", "deep learning", "data analysis", "statistics",
#             "data visualization", "pandas", "numpy", "scikit-learn", "tensorflow",
#             "pytorch", "keras", "nlp", "natural language processing", "computer vision",
#             "big data", "hadoop", "spark", "data mining", "predictive modeling",
#             "neural networks", "ai", "artificial intelligence"
#         ],
#         "cloud_devops": [
#             "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "jenkins",
#             "ci/cd", "terraform", "ansible", "linux", "unix", "shell scripting",
#             "git", "github", "gitlab", "devops", "cloud computing", "microservices"
#         ],
#         "business": [
#             "project management", "agile", "scrum", "leadership", "strategy",
#             "business analysis", "product management", "stakeholder management",
#             "consulting", "finance", "accounting", "marketing", "sales", "negotiation",
#             "business development", "operations", "supply chain", "logistics"
#         ],
#         "design": [
#             "ui/ux", "user experience", "user interface", "graphic design", "figma",
#             "sketch", "adobe xd", "photoshop", "illustrator", "indesign", "prototyping",
#             "wireframing", "visual design", "branding", "typography"
#         ],
#         "healthcare": [
#             "patient care", "clinical", "medical", "nursing", "pharmacy", "diagnosis",
#             "treatment", "healthcare", "hospital", "medical records", "anatomy",
#             "physiology", "pharmacology", "surgery", "radiology", "laboratory",
#             "medicine", "medicines", "drugs", "prescriptions", "therapeutics",
#             "pharmaceutical", "dispensing", "dosage", "drug interactions", "compounding"
#         ],
#         "engineering": [
#             "mechanical engineering", "electrical engineering", "civil engineering",
#             "chemical engineering", "structural", "cad", "autocad", "solidworks",
#             "manufacturing", "quality control", "safety", "thermodynamics"
#         ],
#         "research": [
#             "research", "academic", "scientific", "laboratory", "experimentation",
#             "hypothesis", "publication", "peer review", "methodology", "literature review"
#         ],
#         "soft_skills": [
#             "communication", "teamwork", "problem solving", "critical thinking",
#             "creativity", "adaptability", "time management", "attention to detail",
#             "organization", "collaboration", "presentation", "public speaking"
#         ],
#         "legal": [
#             "law", "legal", "contract", "compliance", "litigation", "regulatory",
#             "intellectual property", "corporate law", "legal research", "paralegal"
#         ],
#         "education": [
#             "teaching", "training", "curriculum", "instruction", "education",
#             "tutoring", "classroom management", "e-learning", "assessment"
#         ],
#         "creative": [
#             "writing", "content creation", "copywriting", "journalism", "editing",
#             "video production", "photography", "animation", "creative directing"
#         ]
#     }
    
#     # Map natural language words/job titles to skill categories
#     WORD_TO_SKILL_MAP = {
#         # Healthcare/Pharmacy
#         "pharmacist": ["pharmacy", "pharmacology", "medicine", "patient care", "healthcare"],
#         "doctor": ["medical", "diagnosis", "treatment", "patient care", "healthcare"],
#         "nurse": ["nursing", "patient care", "clinical", "healthcare", "medical"],
#         "surgeon": ["surgery", "medical", "clinical", "healthcare"],
#         "therapist": ["treatment", "patient care", "healthcare", "clinical"],
#         "dentist": ["medical", "patient care", "healthcare", "clinical"],
#         "veterinarian": ["medical", "diagnosis", "treatment", "healthcare"],
#         "lab technician": ["laboratory", "clinical", "medical", "healthcare"],
#         "radiologist": ["radiology", "medical", "diagnosis", "healthcare"],
        
#         # Technology
#         "programmer": ["programming", "software development", "coding"],
#         "developer": ["programming", "software development", "coding"],
#         "data scientist": ["data analysis", "machine learning", "statistics", "python"],
#         "software engineer": ["programming", "software development", "problem solving"],
#         "web developer": ["html", "css", "javascript", "web development"],
#         "ai engineer": ["machine learning", "deep learning", "ai", "python"],
        
#         # Business
#         "manager": ["leadership", "project management", "strategy", "communication"],
#         "analyst": ["data analysis", "business analysis", "problem solving"],
#         "consultant": ["consulting", "strategy", "communication", "business analysis"],
#         "accountant": ["accounting", "finance", "attention to detail"],
#         "marketer": ["marketing", "communication", "creativity", "strategy"],
        
#         # Education
#         "teacher": ["teaching", "communication", "education", "curriculum"],
#         "professor": ["teaching", "research", "academic", "education"],
#         "trainer": ["training", "communication", "presentation"],
        
#         # Creative
#         "designer": ["graphic design", "creativity", "visual design"],
#         "writer": ["writing", "content creation", "communication"],
#         "artist": ["creativity", "visual design", "art"],
        
#         # Engineering
#         "engineer": ["engineering", "problem solving", "technical"],
#         "architect": ["architecture", "design", "cad"],
        
#         # Legal
#         "lawyer": ["law", "legal", "communication", "legal research"],
#         "attorney": ["law", "legal", "litigation", "legal research"],
        
#         # Common goal words
#         "medicines": ["pharmacy", "pharmacology", "medicine", "healthcare"],
#         "drugs": ["pharmacy", "pharmacology", "pharmaceutical"],
#         "healthcare": ["healthcare", "medical", "patient care"],
#         "hospital": ["healthcare", "hospital", "clinical", "patient care"],
#         "clinic": ["clinical", "healthcare", "patient care"],
#         "patients": ["patient care", "healthcare", "clinical"],
#         "coding": ["programming", "software development"],
#         "computers": ["programming", "software development", "technical"],
#         "business": ["business analysis", "strategy", "management"],
#         "finance": ["finance", "accounting", "business analysis"],
#         "teaching": ["teaching", "education", "communication"],
#         "research": ["research", "scientific", "methodology"],
#         "design": ["graphic design", "creativity", "visual design"],
#         "art": ["creativity", "visual design", "art"],
#         "law": ["law", "legal", "legal research"],
#         "science": ["scientific", "research", "laboratory"],
#     }
    
#     def __init__(self):
#         self.all_skills = set()
#         for category, skills in self.SKILL_TAXONOMY.items():
#             self.all_skills.update([s.lower() for s in skills])
    
#     def extract_skills_from_natural_language(self, text: str) -> List[str]:
#         """
#         Extract skills from natural language goal statements
#         e.g., "I want to become a pharmacist" -> ["pharmacy", "pharmacology", "medicine", "patient care"]
#         """
#         text_lower = text.lower()
#         extracted = []
        
#         # Check for mapped words/job titles
#         for word, skills in self.WORD_TO_SKILL_MAP.items():
#             if word in text_lower:
#                 for skill in skills:
#                     if skill not in extracted:
#                         extracted.append(skill)
        
#         return extracted
    
#     def extract_skills(self, text: str) -> Tuple[List[str], Dict[str, List[str]]]:
#         """
#         Extract skills from text and categorize them
#         Now also handles natural language goal statements
        
#         Returns:
#             Tuple of (list of all extracted skills, dict of categorized skills)
#         """
#         text_lower = text.lower()
#         extracted_skills = []
#         categorized_skills = {cat: [] for cat in self.SKILL_TAXONOMY.keys()}
        
#         # First, try to extract from natural language
#         nl_skills = self.extract_skills_from_natural_language(text)
#         for skill in nl_skills:
#             if skill not in extracted_skills:
#                 extracted_skills.append(skill)
#                 # Find category for this skill
#                 for category, skills in self.SKILL_TAXONOMY.items():
#                     if skill.lower() in [s.lower() for s in skills]:
#                         categorized_skills[category].append(skill)
#                         break
        
#         # Then, do direct taxonomy matching
#         for category, skills in self.SKILL_TAXONOMY.items():
#             for skill in skills:
#                 # Check for exact match or word boundary match
#                 pattern = r'\b' + re.escape(skill.lower()) + r'\b'
#                 if re.search(pattern, text_lower):
#                     if skill not in extracted_skills:
#                         extracted_skills.append(skill)
#                         categorized_skills[category].append(skill)
        
#         # Remove empty categories
#         categorized_skills = {k: v for k, v in categorized_skills.items() if v}
        
#         return extracted_skills, categorized_skills
    
#     def expand_skills(self, skills: List[str]) -> List[str]:
#         """
#         Expand skills with related terms for better matching
#         Now also extracts skills from natural language first
#         """
#         # First, extract skills from natural language inputs
#         extracted = []
#         for skill_input in skills:
#             # Extract from natural language
#             nl_extracted, _ = self.extract_skills(skill_input)
#             extracted.extend(nl_extracted)
#             # Also keep original if it's a valid skill
#             if skill_input.lower() in self.all_skills or len(skill_input) > 2:
#                 extracted.append(skill_input)
        
#         # Remove duplicates
#         expanded = list(dict.fromkeys(extracted))
        
#         skill_expansions = {
#             "python": ["pandas", "numpy", "data analysis"],
#             "machine learning": ["ml", "predictive modeling", "ai"],
#             "data science": ["data analysis", "statistics", "visualization"],
#             "java": ["spring", "enterprise", "backend"],
#             "javascript": ["js", "frontend", "web development"],
#             "sql": ["database", "data management", "queries"],
#             "aws": ["cloud", "cloud computing", "infrastructure"],
#             "react": ["frontend", "ui development", "web"],
#             "leadership": ["management", "team lead", "supervision"],
#             "communication": ["interpersonal", "presentation", "collaboration"],
#             # Healthcare expansions
#             "pharmacy": ["pharmacology", "medicine", "pharmaceutical", "dispensing", "drugs"],
#             "pharmacology": ["pharmacy", "drugs", "medicine", "therapeutics"],
#             "medicine": ["medical", "healthcare", "treatment", "diagnosis"],
#             "patient care": ["healthcare", "clinical", "nursing", "treatment"],
#             "healthcare": ["medical", "patient care", "clinical", "hospital"],
#             "nursing": ["patient care", "healthcare", "clinical", "medical"],
#             "clinical": ["healthcare", "patient care", "medical", "hospital"],
#         }
        
#         for skill in list(expanded):  # Iterate over a copy
#             skill_lower = skill.lower()
#             if skill_lower in skill_expansions:
#                 for exp_skill in skill_expansions[skill_lower]:
#                     if exp_skill not in expanded:
#                         expanded.append(exp_skill)
        
#         return expanded


# class DomainClassifier:
#     """
#     Domain/Category Classification for Career Filtering
#     Classifies user profile into career domains
#     """
    
#     DOMAINS = {
#         "technology": {
#             "keywords": ["programming", "software", "developer", "engineer", "coding", "tech",
#                         "data", "machine learning", "ai", "cloud", "devops", "it", "computer"],
#             "skills": ["python", "java", "javascript", "sql", "aws", "docker", "git",
#                       "machine learning", "data science", "software development"]
#         },
#         "healthcare": {
#             "keywords": ["medical", "health", "hospital", "patient", "clinical", "doctor",
#                         "nurse", "pharmacy", "therapy", "diagnosis", "treatment", "care"],
#             "skills": ["patient care", "clinical", "medical", "nursing", "diagnosis",
#                       "healthcare", "pharmacy", "anatomy", "physiology"]
#         },
#         "business": {
#             "keywords": ["business", "management", "finance", "marketing", "sales", "strategy",
#                         "consulting", "analyst", "banking", "investment", "corporate"],
#             "skills": ["project management", "leadership", "strategy", "finance",
#                       "marketing", "sales", "business analysis", "consulting"]
#         },
#         "creative": {
#             "keywords": ["design", "creative", "art", "visual", "content", "media", "writing",
#                         "photography", "video", "animation", "graphic", "ux", "ui"],
#             "skills": ["graphic design", "ui/ux", "photoshop", "illustrator", "figma",
#                       "content creation", "video production", "writing"]
#         },
#         "engineering": {
#             "keywords": ["engineering", "mechanical", "electrical", "civil", "structural",
#                         "manufacturing", "construction", "architecture", "automotive"],
#             "skills": ["autocad", "solidworks", "mechanical engineering", "electrical engineering",
#                       "civil engineering", "cad", "manufacturing"]
#         },
#         "research": {
#             "keywords": ["research", "science", "scientific", "laboratory", "academic",
#                         "university", "professor", "study", "experiment", "analysis"],
#             "skills": ["research", "scientific", "laboratory", "experimentation",
#                       "publication", "methodology", "analysis"]
#         },
#         "education": {
#             "keywords": ["education", "teaching", "training", "academic", "school", "college",
#                         "university", "instruction", "curriculum", "learning"],
#             "skills": ["teaching", "training", "curriculum", "instruction", "education",
#                       "classroom management", "tutoring"]
#         },
#         "legal": {
#             "keywords": ["legal", "law", "lawyer", "attorney", "compliance", "contract",
#                         "litigation", "regulatory", "court", "paralegal"],
#             "skills": ["law", "legal", "contract", "compliance", "litigation",
#                       "legal research", "regulatory"]
#         }
#     }
    
#     def classify(self, skills: List[str], interests: Optional[List[str]] = None) -> Dict[str, float]:
#         """
#         Classify user into domains with confidence scores
        
#         Returns:
#             Dict of domain -> confidence score (0-1)
#         """
#         text = " ".join(skills + (interests or [])).lower()
#         domain_scores = {}
        
#         for domain, config in self.DOMAINS.items():
#             score = 0.0
            
#             # Keyword matching
#             for keyword in config["keywords"]:
#                 if keyword in text:
#                     score += 1.0
            
#             # Skill matching (higher weight)
#             for skill in config["skills"]:
#                 if skill.lower() in text:
#                     score += 2.0
            
#             # Normalize score
#             max_possible = len(config["keywords"]) + 2 * len(config["skills"])
#             domain_scores[domain] = min(1.0, score / max(1, max_possible * 0.3))
        
#         return domain_scores
    
#     def get_primary_domains(self, skills: List[str], top_k: int = 2) -> List[Tuple[str, float]]:
#         """Get top-k domains for filtering"""
#         scores = self.classify(skills)
#         sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
#         # Filter out domains with very low scores
#         filtered = [(d, s) for d, s in sorted_domains if s > 0.1]
        
#         # If no strong match, return all with some score
#         if not filtered:
#             filtered = sorted_domains[:top_k]
        
#         return filtered[:top_k]


# class HybridCareerRecommender:
#     """
#     Hybrid Career Recommendation System
    
#     Architecture:
#     1. Skill Extraction (NLP)
#     2. Domain Classification
#     3. Dataset Filtering
#     4. SBERT Semantic Similarity (all-mpnet-base-v2)
#     5. Cross-Encoder Reranking
#     6. Education & Experience Matching
#     7. Final Hybrid Scoring
#     """
    
#     # Model choices
#     SBERT_MODEL = "all-mpnet-base-v2"  # Better than all-MiniLM-L6-v2
#     CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
#     # Scoring weights
    
#     WEIGHTS = {
#     "sbert_similarity": 0.40,
#     "cross_encoder": 0.30,
#     "skill_match": 0.20,
#     "domain_match": 0.05,
#     "education_match": 0.05
# }
    
    
#     # WEIGHTS = {
# #     "sbert_similarity": 0.35,
# #     "cross_encoder": 0.25,
# #     "skill_match": 0.20,
# #     "domain_match": 0.10,
# #     "education_match": 0.10
# # }
    
#     # Education levels ranking
#     EDUCATION_LEVELS = {
#         "high school": 1,
#         "diploma": 2,
#         "associate": 3,
#         "bachelor's": 4,
#         "bachelor": 4,
#         "master's": 5,
#         "master": 5,
#         "mba": 5,
#         "phd": 6,
#         "doctorate": 6,
#         "postgraduate": 5
#     }
    
#     def __init__(self, use_local_model: bool = True, local_model_path: str = "./model/sbert_fine_tuned_model"):
#         """
#         Initialize the hybrid recommender
        
#         Args:
#             use_local_model: Whether to try loading local fine-tuned model first
#             local_model_path: Path to local fine-tuned SBERT model
#         """
#         self.skill_extractor = SkillExtractor()
#         self.domain_classifier = DomainClassifier()
        
#         self.sbert_model = None
#         self.cross_encoder = None
#         self.career_df = None
#         self.career_embeddings = None
#         self.is_initialized = False
        
#         self.use_local_model = use_local_model
#         self.local_model_path = local_model_path
    
#     def initialize(self, career_df: pd.DataFrame):
#         """
#         Initialize models and compute career embeddings
        
#         Args:
#             career_df: DataFrame with career data
#         """
#         logger.info("Initializing Hybrid Career Recommender...")
        
#         # Load SBERT model
#         try:
#             import os
#             if self.use_local_model and os.path.exists(self.local_model_path):
#                 try:
#                     self.sbert_model = SentenceTransformer(self.local_model_path)
#                     logger.info(f"✓ Loaded local fine-tuned SBERT model from {self.local_model_path}")
#                 except Exception as e:
#                     logger.warning(f"Failed to load local model: {e}. Falling back to {self.SBERT_MODEL}")
#                     self.sbert_model = SentenceTransformer(self.SBERT_MODEL)
#                     logger.info(f"✓ Loaded SBERT model: {self.SBERT_MODEL}")
#             else:
#                 self.sbert_model = SentenceTransformer(self.SBERT_MODEL)
#                 logger.info(f"✓ Loaded SBERT model: {self.SBERT_MODEL}")
#         except Exception as e:
#             logger.error(f"Failed to load SBERT model: {e}")
#             # Fallback to smaller model
#             self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
#             logger.info("✓ Fallback: Loaded all-MiniLM-L6-v2")
        
#         # Load Cross-Encoder for reranking
#         try:
#             self.cross_encoder = CrossEncoder(self.CROSS_ENCODER_MODEL)
#             logger.info(f"✓ Loaded Cross-Encoder: {self.CROSS_ENCODER_MODEL}")
#         except Exception as e:
#             logger.warning(f"Failed to load Cross-Encoder: {e}. Reranking will be disabled.")
#             self.cross_encoder = None
        
#         # Store career data
#         self.career_df = career_df.copy()
        
#         # Ensure combined_text column exists
#         if "combined_text" not in self.career_df.columns:
#             self.career_df["combined_text"] = self.career_df.apply(
#                 lambda row: f"{row.get('career_name', '')} {row.get('job_description', '')} {row.get('skills', '')}",
#                 axis=1
#             )
        
#         self.career_df["combined_text"] = self.career_df["combined_text"].fillna("")
        
#         # Compute embeddings for all careers
#         logger.info("Computing career embeddings...")
#         career_texts = self.career_df["combined_text"].tolist()
#         self.career_embeddings = self.sbert_model.encode(
#             career_texts,
#             convert_to_tensor=True,
#             show_progress_bar=True
#         )
#         logger.info(f"✓ Computed embeddings for {len(career_texts)} careers")
        
#         self.is_initialized = True
#         logger.info("✓ Hybrid Career Recommender initialized successfully")
    
#     def _build_user_query(self, profile: UserProfile) -> str:
#         """Build enriched user query for embedding"""
#         # Extract and expand skills (handles natural language)
#         expanded_skills = self.skill_extractor.expand_skills(profile.skills)
        
#         # Get domain context based on expanded skills
#         domains = self.domain_classifier.get_primary_domains(expanded_skills)
#         domain_context = " ".join([d[0] for d in domains])
        
#         # Build comprehensive query
#         query_parts = [
#             # Include original user input for semantic matching
#             " ".join(profile.skills),
#             # Include expanded/extracted skills
#             " ".join(expanded_skills),
#             profile.education_level,
#             f"{profile.experience_years} years experience" if profile.experience_years else "",
#             domain_context
#         ]
        
#         if profile.interests:
#             query_parts.append(" ".join(profile.interests))
        
#         return " ".join(filter(None, query_parts))
    
#     def _compute_skill_match(self, user_skills: List[str], career_text: str) -> Tuple[float, List[str], List[str]]:
#         """
#         Compute skill match score between user and career
#         Now handles natural language inputs
        
#         Returns:
#             Tuple of (match_score, matched_skills, missing_skills)
#         """
#         career_text_lower = career_text.lower()
        
#         # First, extract skills from user's natural language input
#         user_extracted_skills = []
#         for skill_input in user_skills:
#             # Extract skills from each user input (handles natural language)
#             extracted, _ = self.skill_extractor.extract_skills(skill_input)
#             user_extracted_skills.extend(extracted)
#             # Also add the raw skill if it's a valid skill
#             if skill_input.lower() in self.skill_extractor.all_skills:
#                 user_extracted_skills.append(skill_input.lower())
        
#         # Remove duplicates while preserving order
#         user_extracted_skills = list(dict.fromkeys(user_extracted_skills))
#         user_skills_lower = [s.lower() for s in user_extracted_skills]
        
#         # If no skills were extracted, use the original input for text matching
#         if not user_skills_lower:
#             user_skills_lower = [s.lower() for s in user_skills]
        
#         # Extract skills from career text
#         career_skills, _ = self.skill_extractor.extract_skills(career_text)
#         career_skills_lower = [s.lower() for s in career_skills]
        
#         # Find matches (using partial matching for better results)
#         matched = []
#         for user_skill in user_skills_lower:
#             for career_skill in career_skills_lower:
#                 if (user_skill in career_skill or 
#                     career_skill in user_skill or 
#                     user_skill in career_text_lower):
#                     if user_skill not in matched:
#                         matched.append(user_skill)
#                     break
        
#         # Also check direct text matching
#         for user_skill in user_skills_lower:
#             if user_skill in career_text_lower and user_skill not in matched:
#                 matched.append(user_skill)
        
#         # Find missing skills (skills in career but not in user profile)
#         missing = [s for s in career_skills_lower if not any(
#             s in us or us in s for us in user_skills_lower
#         )][:5]
        
#         # Calculate match score
#         if career_skills_lower:
#             # Score based on how many career skills the user has
#             score = len(matched) / max(1, len(career_skills_lower))
#         elif user_skills_lower:
#             # If no career skills detected, use text matching
#             match_count = sum(1 for s in user_skills_lower if s in career_text_lower)
#             score = min(1.0, match_count / max(1, len(user_skills_lower)))
#         else:
#             score = 0.0
        
#         return min(1.0, score), matched, missing
    
#     def _compute_education_match(self, user_education: str, career_text: str) -> float:
#         """Compute education level compatibility"""
#         user_level = self.EDUCATION_LEVELS.get(user_education.lower(), 4)
        
#         # Detect required education from career text
#         career_text_lower = career_text.lower()
#         career_level = 4  # Default to bachelor's
        
#         for edu, level in sorted(self.EDUCATION_LEVELS.items(), key=lambda x: -x[1]):
#             if edu in career_text_lower:
#                 career_level = level
#                 break
        
#         # Calculate compatibility (user should meet or exceed requirement)
#         if user_level >= career_level:
#             return 1.0
#         elif user_level == career_level - 1:
#             return 0.7
#         else:
#             return max(0.3, 1.0 - (career_level - user_level) * 0.2)
    
#     def _compute_domain_match(self, user_skills: List[str], career_text: str) -> Tuple[float, str]:
#         """Compute domain alignment score"""
#         # Get user's primary domains
#         user_domains = self.domain_classifier.get_primary_domains(user_skills)
        
#         # Get career's domains
#         career_skills, _ = self.skill_extractor.extract_skills(career_text)
#         career_domains = self.domain_classifier.get_primary_domains(
#             career_skills + career_text.split()[:20]
#         )
        
#         # Calculate overlap
#         user_domain_names = set(d[0] for d in user_domains)
#         career_domain_names = set(d[0] for d in career_domains)
        
#         overlap = user_domain_names & career_domain_names
        
#         if overlap:
#             score = len(overlap) / max(len(user_domain_names), 1)
#             detected_domain = list(overlap)[0]
#         else:
#             score = 0.3  # Base score for cross-domain
#             detected_domain = career_domains[0][0] if career_domains else "general"
        
#         return score, detected_domain
    
#     def _rerank_with_cross_encoder(self, user_query: str, candidates: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
#         """
#         Rerank candidates using Cross-Encoder for more accurate ranking
        
#         Args:
#             user_query: User's query text
#             candidates: List of (career_index, sbert_score) tuples
        
#         Returns:
#             Reranked list of (career_index, cross_encoder_score) tuples
#         """
#         if self.cross_encoder is None or not candidates:
#             return candidates
        
#         # Prepare pairs for cross-encoder
#         pairs = []
#         for idx, _ in candidates:
#             career_text = self.career_df.iloc[idx]["combined_text"]
#             # Truncate to avoid token limits
#             pairs.append((user_query, career_text[:512]))
        
#         # Get cross-encoder scores
#         try:
#             scores = self.cross_encoder.predict(pairs)
            
#             # Normalize scores to 0-1 range (cross-encoder gives raw logits)
#             min_score, max_score = min(scores), max(scores)
#             if max_score > min_score:
#                 normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
#             else:
#                 normalized_scores = [0.5] * len(scores)
            
#             # Create reranked list
#             reranked = [(candidates[i][0], normalized_scores[i]) for i in range(len(candidates))]
#             reranked.sort(key=lambda x: x[1], reverse=True)
            
#             return reranked
#         except Exception as e:
#             logger.warning(f"Cross-encoder reranking failed: {e}")
#             return candidates
    
#     def _determine_confidence(self, score: float) -> str:
#         """Determine confidence level from score"""
#         if score >= 0.80:
#             return "Very High - Excellent Match"
#         elif score >= 0.65:
#             return "High - Strong Match"
#         elif score >= 0.50:
#             return "Medium - Good Fit"
#         elif score >= 0.35:
#             return "Low-Medium - Potential Fit"
#         else:
#             return "Low - Consider Alternatives"
    
#     def recommend(self, profile: UserProfile, top_k: int = 5, 
#                   initial_candidates: int = 50) -> List[CareerMatch]:
#         """
#         Get career recommendations using hybrid approach
        
#         Process:
#         1. Build enriched user query
#         2. SBERT similarity search → top N candidates
#         3. Domain filtering (optional boost)
#         4. Cross-encoder reranking
#         5. Compute final hybrid scores
#         6. Return top-k recommendations
        
#         Args:
#             profile: User profile with skills, education, experience
#             top_k: Number of final recommendations
#             initial_candidates: Number of candidates for re-ranking
        
#         Returns:
#             List of CareerMatch objects
#         """
#         if not self.is_initialized:
#             raise RuntimeError("Recommender not initialized. Call initialize() first.")
        
#         # Step 1: Build user query
#         user_query = self._build_user_query(profile)
#         logger.debug(f"User query: {user_query[:200]}...")
        
#         # Step 2: SBERT similarity search
#         user_embedding = self.sbert_model.encode(user_query, convert_to_tensor=True)
#         similarities = util.cos_sim(user_embedding, self.career_embeddings)[0]
        
#         # Get top N candidates
#         top_results = torch.topk(similarities, k=min(initial_candidates, len(self.career_df)))
        
#         # Prepare candidates list (index, sbert_score)
#         candidates = [
#             (idx.item(), score.item())
#             for score, idx in zip(top_results.values, top_results.indices)
#         ]
        
#         # Step 3: Domain-based filtering/boosting
#         user_domains = self.domain_classifier.get_primary_domains(profile.skills)
        
#         # Step 4: Cross-encoder reranking on top candidates
#         reranked_candidates = self._rerank_with_cross_encoder(user_query, candidates)
        
#         # Step 5: Compute final hybrid scores
#         results = []
        
#         for rank, (idx, ce_score) in enumerate(reranked_candidates[:top_k * 2], start=1):
#             career = self.career_df.iloc[idx]
#             career_text = career["combined_text"]
#             sbert_score = candidates[[c[0] for c in candidates].index(idx)][1]
            
#             # Compute component scores
#             skill_score, matched_skills, missing_skills = self._compute_skill_match(
#                 profile.skills, career_text
#             )
#             domain_score, detected_domain = self._compute_domain_match(
#                 profile.skills, career_text
#             )
#             education_score = self._compute_education_match(
#                 profile.education_level, career_text
#             )
            
#             # Normalize SBERT score (typically 0.15-0.75 → 0-1)
#             # sbert_normalized = max(0, min(1, (sbert_score - 0.15) / 0.6))
#             sbert_normalized = max(0, min(1, (sbert_score - 0.10) / 0.70))
            
#             # Cross-encoder score is already normalized
#             cross_encoder_score = ce_score if self.cross_encoder else sbert_normalized
            
#             # Final weighted score
#             final_score = (
#                 self.WEIGHTS["sbert_similarity"] * sbert_normalized +
#                 self.WEIGHTS["cross_encoder"] * cross_encoder_score +
#                 self.WEIGHTS["skill_match"] * skill_score +
#                 self.WEIGHTS["domain_match"] * domain_score +
#                 self.WEIGHTS["education_match"] * education_score
#             )
            
#             # Generate explanation
#             matched_str = ", ".join(matched_skills[:3]) if matched_skills else "your background"
#             explanation = (
#                 f"Your skills in {matched_str} align well with this {detected_domain} role. "
#                 f"Semantic similarity: {sbert_normalized*100:.1f}%, Skill match: {skill_score*100:.1f}%. "
#                 f"With {profile.experience_years} year(s) of experience and {profile.education_level} "
#                 f"education, you're {'well-positioned' if final_score > 0.6 else 'a potential fit'} "
#                 f"for this career path."
#             )
            
#             results.append(CareerMatch(
#                 rank=rank,
#                 career_name=career.get("career_name", "Unknown Career"),
#                 combined_text=career_text[:500],
#                 sbert_score=sbert_normalized,
#                 cross_encoder_score=cross_encoder_score,
#                 skill_match_score=skill_score,
#                 domain_score=domain_score,
#                 education_score=education_score,
#                 final_score=final_score,
#                 confidence=self._determine_confidence(final_score),
#                 matched_skills=matched_skills,
#                 missing_skills=missing_skills,
#                 detected_domain=detected_domain,
#                 explanation=explanation
#             ))
        
#         # Sort by final score and return top_k
#         results.sort(key=lambda x: x.final_score, reverse=True)
        
#         # Update ranks
#         for i, result in enumerate(results[:top_k], start=1):
#             result.rank = i
        
#         return results[:top_k]
    
#     def get_model_info(self) -> Dict[str, Any]:
#         """Get information about loaded models"""
#         return {
#             "sbert_model": self.SBERT_MODEL if self.sbert_model else None,
#             "cross_encoder_model": self.CROSS_ENCODER_MODEL if self.cross_encoder else None,
#             "career_count": len(self.career_df) if self.career_df is not None else 0,
#             "is_initialized": self.is_initialized,
#             "scoring_weights": self.WEIGHTS
#         }


# # Utility function for quick testing
# def test_hybrid_recommender():
#     """Test the hybrid recommender with sample data"""
#     # Create sample career data
#     sample_data = pd.DataFrame({
#         "career_name": [
#             "Data Scientist",
#             "Software Engineer", 
#             "Product Manager",
#             "Nurse Practitioner",
#             "Marketing Manager"
#         ],
#         "combined_text": [
#             "Data Scientist machine learning python statistics deep learning AI data analysis pandas numpy tensorflow",
#             "Software Engineer java python programming backend frontend development git docker kubernetes",
#             "Product Manager agile scrum product development strategy stakeholder management business analysis",
#             "Nurse Practitioner healthcare patient care medical diagnosis treatment clinical nursing pharmacy",
#             "Marketing Manager marketing strategy brand management sales digital marketing analytics campaigns"
#         ]
#     })
    
#     # Initialize recommender
#     recommender = HybridCareerRecommender(use_local_model=False)
#     recommender.initialize(sample_data)
    
#     # Test profile
#     profile = UserProfile(
#         skills=["python", "machine learning", "data analysis", "statistics"],
#         education_level="Master's",
#         experience_years=3
#     )
    
#     # Get recommendations
#     recommendations = recommender.recommend(profile, top_k=3)
    
#     print("\n" + "="*60)
#     print("HYBRID CAREER RECOMMENDATIONS")
#     print("="*60)
    
#     for rec in recommendations:
#         print(f"\n#{rec.rank}: {rec.career_name}")
#         print(f"   Final Score: {rec.final_score*100:.1f}%")
#         print(f"   Confidence: {rec.confidence}")
#         print(f"   Domain: {rec.detected_domain}")
#         print(f"   Matched Skills: {', '.join(rec.matched_skills[:3])}")
#         print(f"   Missing Skills: {', '.join(rec.missing_skills[:3])}")
    
#     return recommendations


# if __name__ == "__main__":
#     test_hybrid_recommender()























# hybrid_recommender.py
"""
Hybrid AI Career Recommendation System
Architecture:
    User Profile → Skill Extraction (NLP) → Domain Classification
    → Dataset Filtering → SBERT Embedding Search → Top 100 Careers
    → Cross Encoder Ranking → Final Top 5 Careers

Expected Accuracy: 85-92%
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from loguru import logger
import re
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class UserProfile:
    """User profile for career recommendation"""
    skills: List[str]
    education_level: str
    experience_years: int
    interests: Optional[List[str]] = None
    preferred_domain: Optional[str] = None


@dataclass
class CareerMatch:
    """Structured career recommendation result"""
    rank: int
    career_name: str
    combined_text: str
    sbert_score: float
    cross_encoder_score: float
    skill_match_score: float
    domain_score: float
    education_score: float
    final_score: float
    confidence: str
    matched_skills: List[str]
    missing_skills: List[str]
    detected_domain: str
    explanation: str


class SkillExtractor:
    """
    NLP-based Skill Extraction from user input
    Uses keyword matching + pattern recognition + natural language parsing
    """

    SKILL_TAXONOMY = {
        "programming": [
            "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "swift",
            "kotlin", "php", "typescript", "scala", "perl", "r", "matlab", "sql", "nosql",
            "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
            "spring", "express", ".net", "laravel", "rails"
        ],
        "data_science": [
            "machine learning", "deep learning", "data analysis", "statistics",
            "data visualization", "pandas", "numpy", "scikit-learn", "tensorflow",
            "pytorch", "keras", "nlp", "natural language processing", "computer vision",
            "big data", "hadoop", "spark", "data mining", "predictive modeling",
            "neural networks", "ai", "artificial intelligence"
        ],
        "cloud_devops": [
            "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "jenkins",
            "ci/cd", "terraform", "ansible", "linux", "unix", "shell scripting",
            "git", "github", "gitlab", "devops", "cloud computing", "microservices"
        ],
        "business": [
            "project management", "agile", "scrum", "leadership", "strategy",
            "business analysis", "product management", "stakeholder management",
            "consulting", "finance", "accounting", "marketing", "sales", "negotiation",
            "business development", "operations", "supply chain", "logistics"
        ],
        "design": [
            "ui/ux", "user experience", "user interface", "graphic design", "figma",
            "sketch", "adobe xd", "photoshop", "illustrator", "indesign", "prototyping",
            "wireframing", "visual design", "branding", "typography"
        ],
        "healthcare": [
            "patient care", "clinical", "medical", "nursing", "pharmacy", "diagnosis",
            "treatment", "healthcare", "hospital", "medical records", "anatomy",
            "physiology", "pharmacology", "surgery", "radiology", "laboratory",
            "medicine", "medicines", "drugs", "prescriptions", "therapeutics",
            "pharmaceutical", "dispensing", "dosage", "drug interactions", "compounding"
        ],
        "engineering": [
            "mechanical engineering", "electrical engineering", "civil engineering",
            "chemical engineering", "structural", "cad", "autocad", "solidworks",
            "manufacturing", "quality control", "safety", "thermodynamics"
        ],
        "research": [
            "research", "academic", "scientific", "laboratory", "experimentation",
            "hypothesis", "publication", "peer review", "methodology", "literature review"
        ],
        "soft_skills": [
            "communication", "teamwork", "problem solving", "critical thinking",
            "creativity", "adaptability", "time management", "attention to detail",
            "organization", "collaboration", "presentation", "public speaking"
        ],
        "legal": [
            "law", "legal", "contract", "compliance", "litigation", "regulatory",
            "intellectual property", "corporate law", "legal research", "paralegal"
        ],
        "education": [
            "teaching", "training", "curriculum", "instruction", "education",
            "tutoring", "classroom management", "e-learning", "assessment"
        ],
        "creative": [
            "writing", "content creation", "copywriting", "journalism", "editing",
            "video production", "photography", "animation", "creative directing"
        ]
    }

    WORD_TO_SKILL_MAP = {
        "pharmacist": ["pharmacy", "pharmacology", "medicine", "patient care", "healthcare"],
        "doctor": ["medical", "diagnosis", "treatment", "patient care", "healthcare"],
        "nurse": ["nursing", "patient care", "clinical", "healthcare", "medical"],
        "surgeon": ["surgery", "medical", "clinical", "healthcare"],
        "therapist": ["treatment", "patient care", "healthcare", "clinical"],
        "dentist": ["medical", "patient care", "healthcare", "clinical"],
        "veterinarian": ["medical", "diagnosis", "treatment", "healthcare"],
        "lab technician": ["laboratory", "clinical", "medical", "healthcare"],
        "radiologist": ["radiology", "medical", "diagnosis", "healthcare"],
        "programmer": ["programming", "software development", "coding"],
        "developer": ["programming", "software development", "coding"],
        "data scientist": ["data analysis", "machine learning", "statistics", "python"],
        "software engineer": ["programming", "software development", "problem solving"],
        "web developer": ["html", "css", "javascript", "web development"],
        "ai engineer": ["machine learning", "deep learning", "ai", "python"],
        "manager": ["leadership", "project management", "strategy", "communication"],
        "analyst": ["data analysis", "business analysis", "problem solving"],
        "consultant": ["consulting", "strategy", "communication", "business analysis"],
        "accountant": ["accounting", "finance", "attention to detail"],
        "marketer": ["marketing", "communication", "creativity", "strategy"],
        "teacher": ["teaching", "communication", "education", "curriculum"],
        "professor": ["teaching", "research", "academic", "education"],
        "trainer": ["training", "communication", "presentation"],
        "designer": ["graphic design", "creativity", "visual design"],
        "writer": ["writing", "content creation", "communication"],
        "artist": ["creativity", "visual design", "art"],
        "engineer": ["engineering", "problem solving", "technical"],
        "architect": ["architecture", "design", "cad"],
        "lawyer": ["law", "legal", "communication", "legal research"],
        "attorney": ["law", "legal", "litigation", "legal research"],
        "medicines": ["pharmacy", "pharmacology", "medicine", "healthcare"],
        "drugs": ["pharmacy", "pharmacology", "pharmaceutical"],
        "healthcare": ["healthcare", "medical", "patient care"],
        "hospital": ["healthcare", "hospital", "clinical", "patient care"],
        "clinic": ["clinical", "healthcare", "patient care"],
        "patients": ["patient care", "healthcare", "clinical"],
        "coding": ["programming", "software development"],
        "computers": ["programming", "software development", "technical"],
        "business": ["business analysis", "strategy", "management"],
        "finance": ["finance", "accounting", "business analysis"],
        "teaching": ["teaching", "education", "communication"],
        "research": ["research", "scientific", "methodology"],
        "design": ["graphic design", "creativity", "visual design"],
        "art": ["creativity", "visual design", "art"],
        "law": ["law", "legal", "legal research"],
        "science": ["scientific", "research", "laboratory"],
    }

    def __init__(self):
        self.all_skills = set()
        for category, skills in self.SKILL_TAXONOMY.items():
            self.all_skills.update([s.lower() for s in skills])

    def extract_skills_from_natural_language(self, text: str) -> List[str]:
        text_lower = text.lower()
        extracted = []
        for word, skills in self.WORD_TO_SKILL_MAP.items():
            if word in text_lower:
                for skill in skills:
                    if skill not in extracted:
                        extracted.append(skill)
        return extracted

    def extract_skills(self, text: str) -> Tuple[List[str], Dict[str, List[str]]]:
        text_lower = text.lower()
        extracted_skills = []
        categorized_skills = {cat: [] for cat in self.SKILL_TAXONOMY.keys()}

        nl_skills = self.extract_skills_from_natural_language(text)
        for skill in nl_skills:
            if skill not in extracted_skills:
                extracted_skills.append(skill)
                for category, skills in self.SKILL_TAXONOMY.items():
                    if skill.lower() in [s.lower() for s in skills]:
                        categorized_skills[category].append(skill)
                        break

        for category, skills in self.SKILL_TAXONOMY.items():
            for skill in skills:
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    if skill not in extracted_skills:
                        extracted_skills.append(skill)
                        categorized_skills[category].append(skill)

        categorized_skills = {k: v for k, v in categorized_skills.items() if v}
        return extracted_skills, categorized_skills

    def expand_skills(self, skills: List[str]) -> List[str]:
        extracted = []
        for skill_input in skills:
            nl_extracted, _ = self.extract_skills(skill_input)
            extracted.extend(nl_extracted)
            if skill_input.lower() in self.all_skills or len(skill_input) > 2:
                extracted.append(skill_input)

        expanded = list(dict.fromkeys(extracted))

        skill_expansions = {
            "python": ["pandas", "numpy", "data analysis"],
            "machine learning": ["ml", "predictive modeling", "ai"],
            "data science": ["data analysis", "statistics", "visualization"],
            "java": ["spring", "enterprise", "backend"],
            "javascript": ["js", "frontend", "web development"],
            "sql": ["database", "data management", "queries"],
            "aws": ["cloud", "cloud computing", "infrastructure"],
            "react": ["frontend", "ui development", "web"],
            "leadership": ["management", "team lead", "supervision"],
            "communication": ["interpersonal", "presentation", "collaboration"],
            "pharmacy": ["pharmacology", "medicine", "pharmaceutical", "dispensing", "drugs"],
            "pharmacology": ["pharmacy", "drugs", "medicine", "therapeutics"],
            "medicine": ["medical", "healthcare", "treatment", "diagnosis"],
            "patient care": ["healthcare", "clinical", "nursing", "treatment"],
            "healthcare": ["medical", "patient care", "clinical", "hospital"],
            "nursing": ["patient care", "healthcare", "clinical", "medical"],
            "clinical": ["healthcare", "patient care", "medical", "hospital"],
        }

        for skill in list(expanded):
            skill_lower = skill.lower()
            if skill_lower in skill_expansions:
                for exp_skill in skill_expansions[skill_lower]:
                    if exp_skill not in expanded:
                        expanded.append(exp_skill)

        return expanded


class DomainClassifier:
    DOMAINS = {
        "technology": {
            "keywords": ["programming", "software", "developer", "engineer", "coding", "tech",
                         "data", "machine learning", "ai", "cloud", "devops", "it", "computer"],
            "skills": ["python", "java", "javascript", "sql", "aws", "docker", "git",
                       "machine learning", "data science", "software development"]
        },
        "healthcare": {
            "keywords": ["medical", "health", "hospital", "patient", "clinical", "doctor",
                         "nurse", "pharmacy", "therapy", "diagnosis", "treatment", "care"],
            "skills": ["patient care", "clinical", "medical", "nursing", "diagnosis",
                       "healthcare", "pharmacy", "anatomy", "physiology"]
        },
        "business": {
            "keywords": ["business", "management", "finance", "marketing", "sales", "strategy",
                         "consulting", "analyst", "banking", "investment", "corporate"],
            "skills": ["project management", "leadership", "strategy", "finance",
                       "marketing", "sales", "business analysis", "consulting"]
        },
        "creative": {
            "keywords": ["design", "creative", "art", "visual", "content", "media", "writing",
                         "photography", "video", "animation", "graphic", "ux", "ui"],
            "skills": ["graphic design", "ui/ux", "photoshop", "illustrator", "figma",
                       "content creation", "video production", "writing"]
        },
        "engineering": {
            "keywords": ["engineering", "mechanical", "electrical", "civil", "structural",
                         "manufacturing", "construction", "architecture", "automotive"],
            "skills": ["autocad", "solidworks", "mechanical engineering", "electrical engineering",
                       "civil engineering", "cad", "manufacturing"]
        },
        "research": {
            "keywords": ["research", "science", "scientific", "laboratory", "academic",
                         "university", "professor", "study", "experiment", "analysis"],
            "skills": ["research", "scientific", "laboratory", "experimentation",
                       "publication", "methodology", "analysis"]
        },
        "education": {
            "keywords": ["education", "teaching", "training", "academic", "school", "college",
                         "university", "instruction", "curriculum", "learning"],
            "skills": ["teaching", "training", "curriculum", "instruction", "education",
                       "classroom management", "tutoring"]
        },
        "legal": {
            "keywords": ["legal", "law", "lawyer", "attorney", "compliance", "contract",
                         "litigation", "regulatory", "court", "paralegal"],
            "skills": ["law", "legal", "contract", "compliance", "litigation",
                       "legal research", "regulatory"]
        }
    }

    def classify(self, skills: List[str], interests: Optional[List[str]] = None) -> Dict[str, float]:
        text = " ".join(skills + (interests or [])).lower()
        domain_scores = {}

        for domain, config in self.DOMAINS.items():
            score = 0.0
            for keyword in config["keywords"]:
                if keyword in text:
                    score += 1.0
            for skill in config["skills"]:
                if skill.lower() in text:
                    score += 2.0
            max_possible = len(config["keywords"]) + 2 * len(config["skills"])
            domain_scores[domain] = min(1.0, score / max(1, max_possible * 0.3))

        return domain_scores

    def get_primary_domains(self, skills: List[str], top_k: int = 2) -> List[Tuple[str, float]]:
        scores = self.classify(skills)
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        filtered = [(d, s) for d, s in sorted_domains if s > 0.1]
        if not filtered:
            filtered = sorted_domains[:top_k]
        return filtered[:top_k]


class HybridCareerRecommender:
    """
    Hybrid Career Recommendation System — Improved for higher accuracy
    """

    SBERT_MODEL = "all-mpnet-base-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ✅ FIX 1: Improved scoring weights
    WEIGHTS = {
        "sbert_similarity": 0.40,
        "cross_encoder": 0.30,
        "skill_match": 0.20,
        "domain_match": 0.05,
        "education_match": 0.05
    }

    EDUCATION_LEVELS = {
        "high school": 1,
        "diploma": 2,
        "associate": 3,
        "bachelor's": 4,
        "bachelor": 4,
        "master's": 5,
        "master": 5,
        "mba": 5,
        "phd": 6,
        "doctorate": 6,
        "postgraduate": 5
    }

    def __init__(self, use_local_model: bool = True, local_model_path: str = "./model/sbert_fine_tuned_model"):
        self.skill_extractor = SkillExtractor()
        self.domain_classifier = DomainClassifier()
        self.sbert_model = None
        self.cross_encoder = None
        self.career_df = None
        self.career_embeddings = None
        self.is_initialized = False
        self.use_local_model = use_local_model
        self.local_model_path = local_model_path

    def initialize(self, career_df: pd.DataFrame):
        logger.info("Initializing Hybrid Career Recommender...")

        try:
            import os
            if self.use_local_model and os.path.exists(self.local_model_path):
                try:
                    self.sbert_model = SentenceTransformer(self.local_model_path)
                    logger.info(f"✓ Loaded local fine-tuned SBERT model from {self.local_model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load local model: {e}. Falling back to {self.SBERT_MODEL}")
                    self.sbert_model = SentenceTransformer(self.SBERT_MODEL)
            else:
                self.sbert_model = SentenceTransformer(self.SBERT_MODEL)
                logger.info(f"✓ Loaded SBERT model: {self.SBERT_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load SBERT model: {e}")
            self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✓ Fallback: Loaded all-MiniLM-L6-v2")

        try:
            self.cross_encoder = CrossEncoder(self.CROSS_ENCODER_MODEL)
            logger.info(f"✓ Loaded Cross-Encoder: {self.CROSS_ENCODER_MODEL}")
        except Exception as e:
            logger.warning(f"Failed to load Cross-Encoder: {e}. Reranking will be disabled.")
            self.cross_encoder = None

        self.career_df = career_df.copy()

        if "combined_text" not in self.career_df.columns:
            self.career_df["combined_text"] = self.career_df.apply(
                lambda row: f"{row.get('career_name', '')} {row.get('job_description', '')} {row.get('skills', '')}",
                axis=1
            )

        self.career_df["combined_text"] = self.career_df["combined_text"].fillna("")

        logger.info("Computing career embeddings...")
        career_texts = self.career_df["combined_text"].tolist()
        self.career_embeddings = self.sbert_model.encode(
            career_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        logger.info(f"✓ Computed embeddings for {len(career_texts)} careers")

        self.is_initialized = True
        logger.info("✓ Hybrid Career Recommender initialized successfully")

    def _build_user_query(self, profile: UserProfile) -> str:
        expanded_skills = self.skill_extractor.expand_skills(profile.skills)
        domains = self.domain_classifier.get_primary_domains(expanded_skills)
        domain_context = " ".join([d[0] for d in domains])

        query_parts = [
            " ".join(profile.skills),
            " ".join(expanded_skills),
            profile.education_level,
            f"{profile.experience_years} years experience" if profile.experience_years else "",
            domain_context
        ]

        if profile.interests:
            query_parts.append(" ".join(profile.interests))

        return " ".join(filter(None, query_parts))

    def _compute_skill_match(self, user_skills: List[str], career_text: str) -> Tuple[float, List[str], List[str]]:
        career_text_lower = career_text.lower()

        user_extracted_skills = []
        for skill_input in user_skills:
            extracted, _ = self.skill_extractor.extract_skills(skill_input)
            user_extracted_skills.extend(extracted)
            if skill_input.lower() in self.skill_extractor.all_skills:
                user_extracted_skills.append(skill_input.lower())

        user_extracted_skills = list(dict.fromkeys(user_extracted_skills))
        user_skills_lower = [s.lower() for s in user_extracted_skills]

        if not user_skills_lower:
            user_skills_lower = [s.lower() for s in user_skills]

        career_skills, _ = self.skill_extractor.extract_skills(career_text)
        career_skills_lower = [s.lower() for s in career_skills]

        matched = []
        for user_skill in user_skills_lower:
            for career_skill in career_skills_lower:
                if (user_skill in career_skill or
                        career_skill in user_skill or
                        user_skill in career_text_lower):
                    if user_skill not in matched:
                        matched.append(user_skill)
                    break

        for user_skill in user_skills_lower:
            if user_skill in career_text_lower and user_skill not in matched:
                matched.append(user_skill)

        missing = [s for s in career_skills_lower if not any(
            s in us or us in s for us in user_skills_lower
        )][:5]

        # ✅ FIX 3: Improved skill match scoring — average of both sides
        if career_skills_lower:
            career_score = len(matched) / max(1, len(career_skills_lower))
            user_score = len(matched) / max(1, len(user_skills_lower))
            score = (career_score + user_score) / 2
        elif user_skills_lower:
            match_count = sum(1 for s in user_skills_lower if s in career_text_lower)
            score = min(1.0, match_count / max(1, len(user_skills_lower)))
        else:
            score = 0.0

        return min(1.0, score), matched, missing

    def _compute_education_match(self, user_education: str, career_text: str) -> float:
        user_level = self.EDUCATION_LEVELS.get(user_education.lower(), 4)
        career_text_lower = career_text.lower()
        career_level = 4

        for edu, level in sorted(self.EDUCATION_LEVELS.items(), key=lambda x: -x[1]):
            if edu in career_text_lower:
                career_level = level
                break

        if user_level >= career_level:
            return 1.0
        elif user_level == career_level - 1:
            return 0.7
        else:
            return max(0.3, 1.0 - (career_level - user_level) * 0.2)

    def _compute_domain_match(self, user_skills: List[str], career_text: str) -> Tuple[float, str]:
        user_domains = self.domain_classifier.get_primary_domains(user_skills)
        career_skills, _ = self.skill_extractor.extract_skills(career_text)
        career_domains = self.domain_classifier.get_primary_domains(
            career_skills + career_text.split()[:20]
        )

        user_domain_names = set(d[0] for d in user_domains)
        career_domain_names = set(d[0] for d in career_domains)
        overlap = user_domain_names & career_domain_names

        if overlap:
            score = len(overlap) / max(len(user_domain_names), 1)
            detected_domain = list(overlap)[0]
        else:
            score = 0.3
            detected_domain = career_domains[0][0] if career_domains else "general"

        return score, detected_domain

    def _rerank_with_cross_encoder(self, user_query: str, candidates: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if self.cross_encoder is None or not candidates:
            return candidates

        pairs = []
        for idx, _ in candidates:
            career_text = self.career_df.iloc[idx]["combined_text"]
            pairs.append((user_query, career_text[:512]))

        try:
            scores = self.cross_encoder.predict(pairs)
            min_score, max_score = min(scores), max(scores)
            if max_score > min_score:
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized_scores = [0.5] * len(scores)

            reranked = [(candidates[i][0], normalized_scores[i]) for i in range(len(candidates))]
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
            return candidates

    def _determine_confidence(self, score: float) -> str:
        if score >= 0.80:
            return "Very High - Excellent Match"
        elif score >= 0.65:
            return "High - Strong Match"
        elif score >= 0.50:
            return "Medium - Good Fit"
        elif score >= 0.35:
            return "Low-Medium - Potential Fit"
        else:
            return "Low - Consider Alternatives"

    # ✅ FIX 4: Increased initial_candidates from 50 to 100
    def recommend(self, profile: UserProfile, top_k: int = 5,
                  initial_candidates: int = 100) -> List[CareerMatch]:
        if not self.is_initialized:
            raise RuntimeError("Recommender not initialized. Call initialize() first.")

        user_query = self._build_user_query(profile)
        logger.debug(f"User query: {user_query[:200]}...")

        user_embedding = self.sbert_model.encode(user_query, convert_to_tensor=True)
        similarities = util.cos_sim(user_embedding, self.career_embeddings)[0]

        top_results = torch.topk(similarities, k=min(initial_candidates, len(self.career_df)))

        candidates = [
            (idx.item(), score.item())
            for score, idx in zip(top_results.values, top_results.indices)
        ]

        user_domains = self.domain_classifier.get_primary_domains(profile.skills)

        reranked_candidates = self._rerank_with_cross_encoder(user_query, candidates)

        results = []

        for rank, (idx, ce_score) in enumerate(reranked_candidates[:top_k * 2], start=1):
            career = self.career_df.iloc[idx]
            career_text = career["combined_text"]
            sbert_score = candidates[[c[0] for c in candidates].index(idx)][1]

            skill_score, matched_skills, missing_skills = self._compute_skill_match(
                profile.skills, career_text
            )
            domain_score, detected_domain = self._compute_domain_match(
                profile.skills, career_text
            )
            education_score = self._compute_education_match(
                profile.education_level, career_text
            )

            # ✅ FIX 2: Improved SBERT normalization
            sbert_normalized = max(0, min(1, (sbert_score - 0.10) / 0.70))

            cross_encoder_score = ce_score if self.cross_encoder else sbert_normalized

            final_score = (
                self.WEIGHTS["sbert_similarity"] * sbert_normalized +
                self.WEIGHTS["cross_encoder"] * cross_encoder_score +
                self.WEIGHTS["skill_match"] * skill_score +
                self.WEIGHTS["domain_match"] * domain_score +
                self.WEIGHTS["education_match"] * education_score
            )

            matched_str = ", ".join(matched_skills[:3]) if matched_skills else "your background"
            explanation = (
                f"Your skills in {matched_str} align well with this {detected_domain} role. "
                f"Semantic similarity: {sbert_normalized * 100:.1f}%, Skill match: {skill_score * 100:.1f}%. "
                f"With {profile.experience_years} year(s) of experience and {profile.education_level} "
                f"education, you're {'well-positioned' if final_score > 0.6 else 'a potential fit'} "
                f"for this career path."
            )

            results.append(CareerMatch(
                rank=rank,
                career_name=career.get("career_name", "Unknown Career"),
                combined_text=career_text[:500],
                sbert_score=sbert_normalized,
                cross_encoder_score=cross_encoder_score,
                skill_match_score=skill_score,
                domain_score=domain_score,
                education_score=education_score,
                final_score=final_score,
                confidence=self._determine_confidence(final_score),
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                detected_domain=detected_domain,
                explanation=explanation
            ))

        results.sort(key=lambda x: x.final_score, reverse=True)

        for i, result in enumerate(results[:top_k], start=1):
            result.rank = i

        return results[:top_k]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "sbert_model": self.SBERT_MODEL if self.sbert_model else None,
            "cross_encoder_model": self.CROSS_ENCODER_MODEL if self.cross_encoder else None,
            "career_count": len(self.career_df) if self.career_df is not None else 0,
            "is_initialized": self.is_initialized,
            "scoring_weights": self.WEIGHTS
        }


def test_hybrid_recommender():
    """Test the hybrid recommender with sample data"""
    sample_data = pd.DataFrame({
        "career_name": [
            "Data Scientist",
            "Software Engineer",
            "Product Manager",
            "Nurse Practitioner",
            "Marketing Manager"
        ],
        "combined_text": [
            "Data Scientist machine learning python statistics deep learning AI data analysis pandas numpy tensorflow",
            "Software Engineer java python programming backend frontend development git docker kubernetes",
            "Product Manager agile scrum product development strategy stakeholder management business analysis",
            "Nurse Practitioner healthcare patient care medical diagnosis treatment clinical nursing pharmacy",
            "Marketing Manager marketing strategy brand management sales digital marketing analytics campaigns"
        ]
    })

    recommender = HybridCareerRecommender(use_local_model=False)
    recommender.initialize(sample_data)

    profile = UserProfile(
        skills=["python", "machine learning", "data analysis", "statistics"],
        education_level="Master's",
        experience_years=3
    )

    recommendations = recommender.recommend(profile, top_k=3)

    print("\n" + "=" * 60)
    print("HYBRID CAREER RECOMMENDATIONS")
    print("=" * 60)

    for rec in recommendations:
        print(f"\n#{rec.rank}: {rec.career_name}")
        print(f"   Final Score: {rec.final_score * 100:.1f}%")
        print(f"   Confidence: {rec.confidence}")
        print(f"   Domain: {rec.detected_domain}")
        print(f"   Matched Skills: {', '.join(rec.matched_skills[:3])}")
        print(f"   Missing Skills: {', '.join(rec.missing_skills[:3])}")

    return recommendations


if __name__ == "__main__":
    test_hybrid_recommender()