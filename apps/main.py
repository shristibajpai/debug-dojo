
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from apps.llm_service import ask_qwen


# ==================================================
# 1. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Debug Dojo AI",
    description="AI-powered programming debugging coach",
    version="1.1.0",
)


# ==================================================
# 2. IN-MEMORY SESSION STORAGE
# ==================================================

sessions = {}
sessions_lock = Lock()


# ==================================================
# 3. REQUEST AND RESPONSE MODELS
# ==================================================

class DebugRequest(BaseModel):
    session_id: str | None = Field(
        default=None, min_length=1, max_length=100
    )
    code: str = Field(min_length=1, max_length=10000)
    error_message: str = Field(default="", max_length=3000)
    hint_level: int = Field(default=1, ge=1, le=3)


class DebugResponse(BaseModel):
    session_id: str
    hint_level: int
    bug_type: str
    hint: str


class InterviewRescueRequest(BaseModel):
    code: str = Field(min_length=1, max_length=10000)
    error_message: str = Field(default="", max_length=3000)
    failed_attempts: int = Field(ge=0, le=100)


class InterviewRescueResponse(BaseModel):
    activated: bool
    debugging_approach: str
    explanation_english: str
    explanation_hinglish: str


# ==================================================
# 4. BUG CLASSIFICATION
# ==================================================

def classify_bug(code: str, error_message: str) -> str:
    code_lower = code.lower()
    error_lower = error_message.lower()

    if (
        "indexerror" in error_lower
        or "out of range" in error_lower
        or "range(len(" in code_lower
    ):
        return "off_by_one"

    if "keyerror" in error_lower:
        return "missing_key"

    if (
        "zerodivisionerror" in error_lower
        or "/ 0" in code_lower
    ):
        return "division_by_zero"

    if (
        "nonetype" in error_lower
        or "attributeerror" in error_lower
    ):
        return "null_or_attribute_error"

    if (
        "typeerror" in error_lower
        or "unsupported operand" in error_lower
    ):
        return "type_mismatch"

    if (
        "syntaxerror" in error_lower
        or "indentationerror" in error_lower
    ):
        return "syntax_or_indentation"

    return "unknown"


# ==================================================
# 5. HOME AND HEALTH ENDPOINTS
# ==================================================

@app.get("/")
def home():
    return {
        "message": "Debug Dojo AI backend is running!",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "qwen2.5-coder:3b",
    }


# ==================================================
# 6. DEBUGGING AND PROGRESSIVE HINTS
# ==================================================

@app.post("/debug", response_model=DebugResponse)
def debug_code(request: DebugRequest):

    session_id = request.session_id or str(uuid4())

    # Validate session and hint progression.
    with sessions_lock:
        session = sessions.get(session_id)

        if session is None:
            if request.hint_level != 1:
                raise HTTPException(
                    status_code=409,
                    detail="Start with hint level 1.",
                )

            session = {
                "next_hint": 1,
                "code": request.code,
                "error_message": request.error_message,
            }
            sessions[session_id] = session

        if (
            session["code"] != request.code
            or session["error_message"] != request.error_message
        ):
            raise HTTPException(
                status_code=409,
                detail="Start a new session for a different problem.",
            )

        if request.hint_level != session["next_hint"]:
            raise HTTPException(
                status_code=409,
                detail=f"Request hint level {session['next_hint']} next.",
            )

        hint_level = session["next_hint"]

    prompt = f"""
You are Debug Dojo, a programming debugging coach.

Analyze only the visible code and supplied error evidence.

Rules:
- Never provide a complete corrected solution.
- Never reveal hidden tests or private expected outputs.
- Never invent test results or claim code was executed.
- Return one concise hint only.

Hint levels:
1. Explain the relevant concept.
2. Narrow down the likely cause.
3. Suggest a specific debugging check without corrected code.

Student code:
{request.code}

Visible error evidence:
{request.error_message or "No error message provided"}

Hint level: {hint_level}

Return only the hint.
"""

    try:
        hint = ask_qwen(prompt)

    except (RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The local AI service could not generate a hint.",
        ) from exc

    if not hint or not hint.strip():
        raise HTTPException(
            status_code=503,
            detail="The AI returned an empty hint.",
        )

    bug_type = classify_bug(
        request.code,
        request.error_message,
    )

    # Advance only after successfully generating a hint.
    with sessions_lock:
        sessions[session_id]["next_hint"] = min(
            hint_level + 1, 3
        )

    return DebugResponse(
        session_id=session_id,
        hint_level=hint_level,
        bug_type=bug_type,
        hint=hint.strip(),
    )


# ==================================================
# 7. INTERVIEW RESCUE
# ==================================================

@app.post(
    "/interview-rescue",
    response_model=InterviewRescueResponse,
)
def interview_rescue(request: InterviewRescueRequest):

    # Do not activate before two reported failed attempts.
    if request.failed_attempts < 2:
        return InterviewRescueResponse(
            activated=False,
            debugging_approach=(
                "Inspect the error, isolate the relevant code, "
                "and test one assumption at a time."
            ),
            explanation_english=(
                "I am still investigating the issue and "
                "checking the relevant boundary conditions."
            ),
            explanation_hinglish=(
                "Main abhi issue ko investigate kar rahi hoon "
                "aur relevant boundary conditions check kar rahi hoon."
            ),
        )

    prompt = f"""
You are Interview Rescue in Debug Dojo.

The student reports at least two failed attempts.

Analyze only the supplied code and visible error evidence.

Return exactly these three labeled sections:

DEBUGGING APPROACH:
Give a practical, step-by-step debugging approach.
Do not provide the complete corrected solution.

ENGLISH:
Give a concise, honest interview explanation.
Do not claim the student fixed the bug or ran tests
unless the supplied evidence establishes that.

HINGLISH:
Give the same explanation naturally in Hinglish.

Never invent test results.
Never reveal hidden tests or private expected outputs.
If evidence is insufficient, state that clearly.

Code:
{request.code}

Visible error evidence:
{request.error_message or "No error message provided"}
"""

    try:
        result = ask_qwen(prompt)

    except (RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Interview Rescue could not contact the AI service.",
        ) from exc

    if not result or not result.strip():
        raise HTTPException(
            status_code=503,
            detail="The AI returned an empty response.",
        )

    # Parse the required response sections.
    sections = {
        "DEBUGGING APPROACH:": [],
        "ENGLISH:": [],
        "HINGLISH:": [],
    }

    current_section = None

    for line in result.splitlines():
        normalized = line.strip().upper()

        if normalized in sections:
            current_section = normalized
        elif current_section is not None:
            sections[current_section].append(line)

    parsed = {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
    }

    if not all(parsed.values()):
        raise HTTPException(
            status_code=502,
            detail=(
                "The AI response did not contain all three "
                "required sections. Please retry."
            ),
        )

    return InterviewRescueResponse(
        activated=True,
        debugging_approach=parsed["DEBUGGING APPROACH:"],
        explanation_english=parsed["ENGLISH:"],
        explanation_hinglish=parsed["HINGLISH:"],
    )
