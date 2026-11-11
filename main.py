import json
import os
import threading
import time
from typing import List, Optional

import pdfplumber
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

from match import analyze_resume_content, score_match

load_dotenv()

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ANALYZE_CHARS = 8000
ANALYZE_LIMIT = 8
ANALYZE_WINDOW_SECONDS = 60

app = FastAPI(docs_url=None, redoc_url=None)

_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_rate_lock = threading.Lock()
_analyze_hits = {}


def split_skills(raw: str) -> List[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def parse_pdf(file_content: bytes) -> str:
    try:
        import io

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            return "".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error parsing PDF: {exc}") from exc


def parse_docx(file_content: bytes) -> str:
    try:
        import io

        doc = Document(io.BytesIO(file_content))
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error parsing DOCX: {exc}") from exc


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def check_analyze_rate(request: Request) -> None:
    """Per-process cap so one client cannot spend the Gemini key unchecked."""
    now = time.time()
    ip = client_ip(request)
    with _rate_lock:
        recent = [stamp for stamp in _analyze_hits.get(ip, []) if now - stamp < ANALYZE_WINDOW_SECONDS]
        if len(recent) >= ANALYZE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Too many analyze requests. Try again in a minute.",
            )
        recent.append(now)
        _analyze_hits[ip] = recent


_gemini: Optional[genai.Client] = None
FALLBACK_MODELS = ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite")


def gemini_client() -> genai.Client:
    global _gemini
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not set")
    if _gemini is None:
        _gemini = genai.Client(api_key=key)
    return _gemini


def text_from_upload(file: UploadFile) -> str:
    content = file.file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Resume file is larger than 5 MB.")

    filename = (file.filename or "").lower()
    if filename.endswith(".doc") and not filename.endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="DOC is not supported. Save the resume as DOCX, PDF, or TXT.",
        )
    if filename.endswith(".pdf"):
        return parse_pdf(content)
    if filename.endswith(".docx"):
        return parse_docx(content)
    if filename.endswith(".txt"):
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="TXT file is not valid UTF-8.") from exc
    raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT.")


@app.post("/upload-resume")
async def upload_resume(
    file: Optional[UploadFile] = File(default=None, description="PDF, DOCX, or TXT, up to 5 MB."),
    resume_text: str = Form(
        default="",
        description="Paste the resume here. If this box has text, it is used instead of the file.",
    ),
):
    """Paste a resume or upload a PDF, DOCX, or TXT file."""
    text = resume_text.strip()
    if not text:
        if file is None or not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Paste resume text or upload a PDF, DOCX, or TXT file.",
            )
        text = text_from_upload(file)

    return {
        "success": True,
        "resume_text": text,
        "analysis": analyze_resume_content(text),
    }


@app.post("/match-job")
def match_job(
    resume_text: str = Form(..., description="Paste the full resume."),
    page_text: str = Form(..., description="Paste the full job page."),
    user_skills: str = Form(
        default="",
        description="Optional extra skills, comma separated.",
    ),
):
    """Score the posting's skill list against the resume."""
    return score_match(resume_text, page_text, split_skills(user_skills))


ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer"},
                    "skill": {"type": "string"},
                    "need": {"type": "string", "enum": ["required", "preferred"]},
                },
                "required": ["rank", "skill", "need"],
            },
        },
        "expectations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["skills", "expectations"],
}


def analyze_models() -> List[str]:
    chosen = os.getenv("GEMINI_MODEL", "").strip() or FALLBACK_MODELS[0]
    models = [chosen]
    for name in FALLBACK_MODELS:
        if name not in models:
            models.append(name)
    return models


def gemini_busy(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code in (429, 500, 503):
        return True
    text = str(exc).upper()
    return "UNAVAILABLE" in text or "HIGH DEMAND" in text or "RESOURCE_EXHAUSTED" in text


def generate_analysis(client: genai.Client, prompt: str) -> str:
    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_json_schema=ANALYZE_SCHEMA,
    )
    last_error: Optional[Exception] = None
    for model in analyze_models():
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except HTTPException:
            raise
        except Exception as exc:
            last_error = exc
            if gemini_busy(exc):
                time.sleep(0.3)
                continue
            raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc
        raw = (getattr(response, "text", None) or "").strip()
        if raw:
            return raw
        last_error = RuntimeError(f"{model} returned an empty response")
    if last_error is not None and gemini_busy(last_error):
        raise HTTPException(
            status_code=503,
            detail="Gemini is busy right now. Try again in a moment.",
        ) from last_error
    raise HTTPException(status_code=502, detail="Gemini returned an empty response.")


def format_analysis(skills: list, expectations: list) -> str:
    skill_lines = []
    for item in skills:
        skill_lines.append(f"{item['rank']}. {item['skill']} — {item['need']}")
    expectation_lines = [f"- {line}" for line in expectations]
    return "Skills\n" + "\n".join(skill_lines) + "\n\nExpectations\n" + "\n".join(expectation_lines)


def parse_analysis(raw: str) -> dict:
    data = json.loads(raw)
    skills = []
    for item in data.get("skills") or []:
        skill = str(item.get("skill", "")).strip()
        need = str(item.get("need", "")).strip().lower()
        if not skill or need not in ("required", "preferred"):
            continue
        skills.append({"rank": int(item.get("rank") or 0), "skill": skill, "need": need})
    skills.sort(key=lambda item: item["rank"])
    for index, item in enumerate(skills, start=1):
        item["rank"] = index
    expectations = [str(line).strip() for line in data.get("expectations") or [] if str(line).strip()]
    if not skills and not expectations:
        raise ValueError("empty analysis")
    return {
        "skills": skills,
        "expectations": expectations,
        "result": format_analysis(skills, expectations),
    }


@app.post("/analyze")
def analyze(
    http_request: Request,
    text: str = Form(..., description="Paste the job posting."),
):
    """Ask Gemini for a priority-ordered skill list and non-skill expectations."""
    check_analyze_rate(http_request)
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_ANALYZE_CHARS:
        text = text[:MAX_ANALYZE_CHARS]

    prompt = f"""
    Read the job posting and return only the JSON object described by the schema.

    skills: every concrete skill the posting asks for, highest priority first.
    Rank 1 is the skill that matters most for this role.
    Put a skill higher when the posting names it as a core requirement, repeats it, or the day-to-day work depends on it.
    A named preferred skill can outrank a generic required skill.
    need is "required" only when the posting says required, must, or proficient.
    Familiarity, nice-to-have, and the word preferred stay "preferred".
    skill is a short name, not a sentence. One skill per item. No soft skills here.

    expectations: requirements that are not skills, most important first.
    Include education, seniority, employment type, domain interest, and stated preferences such as prior research only when the posting states them.
    One short line each. No paragraphs, no red-flag essay, no responsibility essay.

    Job posting:
    {text}
    """
    raw = generate_analysis(gemini_client(), prompt)
    try:
        return parse_analysis(raw)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"Gemini returned an unusable analysis: {exc}") from exc


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Backend is running"}


@app.get("/docs", include_in_schema=False)
def custom_docs():
    page = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="AI Job Match — docs",
    )
    html = page.body.decode()
    style = """
    <style>
      textarea.body-param__text {
        display: block;
        box-sizing: border-box;
        min-height: 160px;
        width: 100%;
        margin-top: 8px;
        padding: 8px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 13px;
        line-height: 1.45;
      }
    </style>
    """
    script = """
    <script>
      function setNativeValue(el, value) {
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
        setter.call(el, value);
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }
      function attachLongText(input) {
        if (input.dataset.longText) return;
        const row = input.closest("tr[data-property-name]");
        if (!row || input.type !== "text") return;
        const name = row.getAttribute("data-property-name");
        if (!["resume_text", "page_text", "text", "user_skills"].includes(name)) return;
        input.dataset.longText = "1";
        const box = document.createElement("textarea");
        box.className = "body-param__text";
        box.placeholder = "Paste " + name.replaceAll("_", " ");
        const starter = input.value === "string" ? "" : input.value;
        box.value = starter;
        setNativeValue(input, starter);
        box.addEventListener("input", () => setNativeValue(input, box.value));
        input.hidden = true;
        input.after(box);
      }
      function scanLongText() {
        document.querySelectorAll(".opblock-section-request-body input[type='text']").forEach(attachLongText);
      }
      new MutationObserver(scanLongText).observe(document.body, { childList: true, subtree: true });
      scanLongText();
    </script>
    """
    html = html.replace("</head>", style + "</head>")
    return HTMLResponse(html.replace("</body>", script + "</body>"))
