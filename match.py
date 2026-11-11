import re
from typing import List, Optional

# Whole-word skills. Short English words ("go", "r", "api") are left out
# so a normal sentence is not counted as a technology.
COMMON_SKILLS = {
    "python", "java", "javascript", "typescript", "csharp", "c#", "cpp", "c++", "ruby", "php", "swift", "kotlin",
    "golang", "rust", "scala", "matlab", "perl",
    "react", "vue", "angular", "django", "flask", "fastapi", "express", "nextjs", "next.js", "laravel", "rails",
    "spring", "asp.net", "dotnet", ".net",
    "sql", "mysql", "postgresql", "mongodb", "firebase", "redis", "elasticsearch", "cassandra", "dynamodb",
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "gitlab", "github", "terraform", "ansible",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "spark", "hadoop",
    "leadership", "communication", "teamwork", "problem solving", "project management", "agile", "scrum",
    "ui design", "ux design", "graphic design", "content writing", "copywriting",
    "git", "rest api", "graphql", "microservices", "nosql", "html", "css", "scss", "sass",
    "webpack", "npm", "yarn", "linux", "unix", "windows", "macos", "deployment", "ci/cd",
}

APPLY_AT = 70
MAYBE_AT = 50


def extract_skills_from_text(text: str) -> List[str]:
    """Return known skills that appear as whole words."""
    text_lower = (text or "").lower()
    found = set()
    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return sorted(found)


def analyze_resume_content(resume_text: str) -> dict:
    skills = extract_skills_from_text(resume_text)
    exp_matches = re.findall(
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?experience",
        (resume_text or "").lower(),
    )
    experience_years = int(exp_matches[0]) if exp_matches else 0
    return {
        "skills": skills,
        "experience_years": experience_years,
        "skill_count": len(skills),
    }


def score_match(resume_text: str, page_text: str, user_skills: Optional[List[str]] = None) -> dict:
    """Score how much of the posting's skill list the candidate covers.

    missing_skills are posting skills the resume does not have.
    The decision uses that coverage, not the size of the resume.
    """
    extra = extract_skills_from_text(" ".join(user_skills or []))
    candidate = sorted(set(extract_skills_from_text(resume_text)) | set(extra))
    job_skills = extract_skills_from_text(page_text)
    matched = sorted(skill for skill in job_skills if skill in candidate)
    missing = sorted(skill for skill in job_skills if skill not in candidate)

    note = ""
    if not job_skills:
        decision, color, confidence, score = "MAYBE APPLY", "orange", "low", 0
        note = "No skills from the list were found in the posting."
    else:
        score = round((len(matched) / len(job_skills)) * 100)
        if score >= APPLY_AT:
            decision, color, confidence = "APPLY", "green", "high"
        elif score >= MAYBE_AT:
            decision, color, confidence = "MAYBE APPLY", "orange", "medium"
        else:
            decision, color, confidence = "DO NOT APPLY", "red", "low"

    return {
        "success": True,
        "decision": decision,
        "decision_color": color,
        "confidence": confidence,
        "skill_match_score": score,
        "required_match_score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "your_skills": candidate,
        "job_requirements": job_skills,
        "matched_count": len(matched),
        "missing_count": len(missing),
        "job_skills_count": len(job_skills),
        "note": note,
    }
