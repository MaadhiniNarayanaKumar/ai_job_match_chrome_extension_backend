from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import openai
import pdfplumber
from docx import Document
import re
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

app = FastAPI()

class JobMatchRequest(BaseModel):
    resume_text: str
    page_text: str
    user_skills: List[str] = []

class AnalyzeRequest(BaseModel):
    text: str

# -------------------- Skill Extraction Functions --------------------
def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills from resume/job text using simple keyword matching and NLP"""
    
    # Common technical and soft skills to match
    common_skills = {
        # Programming languages
        "python", "java", "javascript", "typescript", "csharp", "c#", "cpp", "c++", "ruby", "php", "swift", "kotlin",
        "golang", "go", "rust", "scala", "r", "matlab", "perl",
        # Web frameworks
        "react", "vue", "angular", "django", "flask", "fastapi", "express", "nextjs", "next.js", "laravel", "rails",
        "spring", "asp.net", "dotnet", ".net",
        # Databases
        "sql", "mysql", "postgresql", "mongodb", "firebase", "redis", "elasticsearch", "cassandra", "dynamodb",
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "gitlab", "github", "terraform", "ansible",
        # Data & ML
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "spark", "hadoop",
        # Soft skills
        "leadership", "communication", "teamwork", "problem solving", "project management", "agile", "scrum",
        "ui design", "ux design", "graphic design", "content writing", "copywriting",
        # Other technical
        "git", "rest api", "graphql", "microservices", "api", "sql", "nosql", "html", "css", "scss", "sass",
        "webpack", "npm", "yarn", "linux", "unix", "windows", "macos", "deployment", "ci/cd"
    }
    
    text_lower = text.lower()
    found_skills = set()
    
    for skill in common_skills:
        # Match whole words to avoid partial matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    
    return sorted(list(found_skills))

def analyze_resume_content(resume_text: str) -> dict:
    """Analyze resume and extract key information"""
    skills = extract_skills_from_text(resume_text)
    
    # Extract years of experience pattern
    exp_pattern = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?experience'
    exp_matches = re.findall(exp_pattern, resume_text.lower())
    experience_years = int(exp_matches[0]) if exp_matches else 0
    
    return {
        "skills": skills,
        "experience_years": experience_years,
        "skill_count": len(skills)
    }

# -------------------- File Parsing Functions --------------------
def parse_pdf(file_content: bytes) -> str:
    """Extract text from PDF"""
    try:
        import io
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing PDF: {str(e)}")

def parse_docx(file_content: bytes) -> str:
    """Extract text from DOCX"""
    try:
        import io
        doc = Document(io.BytesIO(file_content))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing DOCX: {str(e)}")

# -------------------- API Endpoints --------------------

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload and parse resume (PDF, DOCX, or TXT)"""
    try:
        content = await file.read()
        filename = file.filename.lower()
        
        if filename.endswith(".pdf"):
            text = parse_pdf(content)
        elif filename.endswith(".docx") or filename.endswith(".doc"):
            text = parse_docx(content)
        elif filename.endswith(".txt"):
            text = content.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT")
        
        resume_info = analyze_resume_content(text)
        
        return {
            "success": True,
            "resume_text": text,
            "analysis": resume_info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/match-job")
def match_job(request: JobMatchRequest):
    """
    Match resume skills against job posting.
    Returns match score, matched skills, missing skills, and AI suggestions
    """
    resume_text = request.resume_text
    page_text = request.page_text
    user_skills = request.user_skills
    
    # Extract skills from resume
    resume_skills = extract_skills_from_text(resume_text)
    
    # Combine with user manually entered skills
    all_candidate_skills = list(set(resume_skills + user_skills))
    
    # Extract job requirements from page
    job_skills = extract_skills_from_text(page_text)
    
    # Calculate match
    matched_skills = [s for s in all_candidate_skills if s in job_skills]
    missing_skills = [s for s in all_candidate_skills if s not in job_skills]
    
    # Calculate match percentage
    skill_match_score = 0
    if all_candidate_skills:
        skill_match_score = round((len(matched_skills) / len(all_candidate_skills)) * 100)
    
    # Calculate required match (what % of job requirements do we have)
    required_match_score = 0
    if job_skills:
        required_match_score = round((len(matched_skills) / len(job_skills)) * 100)
    
    # Make decision
    decision = "DO NOT APPLY"
    decision_color = "red"
    confidence = "low"
    
    if skill_match_score >= 70:
        decision = "APPLY"
        decision_color = "green"
        confidence = "high"
    elif skill_match_score >= 50:
        decision = "MAYBE APPLY"
        decision_color = "orange"
        confidence = "medium"
    
    return {
        "success": True,
        "decision": decision,
        "decision_color": decision_color,
        "confidence": confidence,
        "skill_match_score": skill_match_score,
        "required_match_score": required_match_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "your_skills": all_candidate_skills,
        "job_requirements": job_skills,
        "matched_count": len(matched_skills),
        "missing_count": len(missing_skills),
        "job_skills_count": len(job_skills)
    }

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Use AI to analyze job posting and provide detailed insights"""
    prompt = f"""
    Analyze this job posting and provide:
    1. Key required skills
    2. Experience level needed
    3. Main responsibilities
    4. Red flags or concerns
    
    Job Posting:
    {request.text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"result": response.choices[0].message.content}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Backend is running"}
