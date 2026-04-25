"""
Adversa — FastAPI Server (OpenEnv-compliant)

Endpoints:
    POST /reset           — Start new episode
    POST /step            — Execute action
    GET  /state           — Full internal state
    GET  /health          — Health check
    GET  /cases           — List available cases
    GET  /docs            — Auto-generated API docs
"""
from __future__ import annotations

import sys
import os

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional

from server.adversa_environment import AdversaEnvironment
from server.case_templates import ALL_CASES

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Adversa — Multi-Agent Courtroom Environment",
    description=(
        "A multi-agent adversarial courtroom trial simulation environment. "
        "Trains LLMs to learn WHEN to speak, not just WHAT to say. "
        "OpenEnv-compliant: /reset, /step, /state."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful per session)
_env = AdversaEnvironment()

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: int = 0
    options: Optional[dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "seed": 42,
                "options": {"case_id": "C1", "role": "prosecutor"}
            }
        }


class StepRequest(BaseModel):
    action: dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "action": {
                    "role": "prosecutor",
                    "action_type": "present_evidence",
                    "evidence_id": "E1",
                    "framing": "factual"
                }
            }
        }


class ResetResponse(BaseModel):
    observation: dict[str, Any]


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — required by OpenEnv spec."""
    return {
        "status": "healthy",
        "environment": "Adversa",
        "version": "1.0.0",
        "cases_available": len(ALL_CASES),
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    """
    Reset the environment to a new episode.

    - **seed**: Random seed for episode reproducibility
    - **options.case_id**: Which case to load (C1–C10). Random if omitted.
    - **options.role**: Role of the calling agent (prosecutor/defense/judge)
    """
    try:
        obs = _env.reset(seed=request.seed, options=request.options)
        return ResetResponse(observation=obs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Execute one action in the environment.

    Action types by phase:
    - **opening**: opening_statement, pass
    - **prosecution_case** (prosecutor): present_evidence, pass
    - **prosecution_case** (defense): object, cross_examine, pass
    - **defense_case** (defense): present_evidence, pass
    - **defense_case** (prosecutor): object, cross_examine, pass
    - **closing**: closing_argument, pass
    - **verdict** (judge): instruct_jury, pass
    """
    try:
        obs, reward, done, info = _env.step(request.action)
        return StepResponse(
            observation=obs,
            reward=float(reward),
            done=done,
            info=info,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state")
def state():
    """
    Get full internal environment state (for grading/debugging).
    Contains ground truth verdict — NOT visible to agents during training.
    """
    return _env.state()


@app.get("/cases")
def list_cases():
    """List all available case templates."""
    return {
        case_id: {
            "name": case.name,
            "charges": case.charges,
            "difficulty": case.difficulty,
            "ground_truth": case.ground_truth,
            "prosecution_evidence_count": len(case.prosecution_evidence),
            "defense_evidence_count": len(case.defense_evidence),
        }
        for case_id, case in ALL_CASES.items()
    }


@app.get("/cases/{case_id}")
def get_case_detail(case_id: str):
    """Get detailed case template by ID."""
    if case_id not in ALL_CASES:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    case = ALL_CASES[case_id]
    return {
        "case_id": case.case_id,
        "name": case.name,
        "charges": case.charges,
        "defendant": case.defendant,
        "difficulty": case.difficulty,
        "ground_truth": case.ground_truth,
        "summary": case.summary,
        "prosecution_evidence": [
            {
                "id": e.id,
                "description": e.description,
                "strength": e.strength,
                "emotional_impact": e.emotional_impact,
                "admissible": e.admissible,
            }
            for e in case.prosecution_evidence
        ],
        "defense_evidence": [
            {
                "id": e.id,
                "description": e.description,
                "strength": e.strength,
                "emotional_impact": e.emotional_impact,
                "admissible": e.admissible,
            }
            for e in case.defense_evidence
        ],
    }
