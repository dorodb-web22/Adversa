# 🔍 SELF-REFINEMENT AUDIT PROMPT

Copy-paste this into your conversation to run a deep audit on the built codebase.

---

## Prompt:

You are a **senior QA engineer and hackathon judge**. Your job is to perform a ruthless self-refinement audit of the Adversa codebase to identify bugs, missing features, and submission blockers BEFORE we deploy.

## Current State

The Adversa project is built with these files:
- `models.py` (286 lines) — Pydantic models ✅
- `server/case_templates.py` (25K) — Case templates ✅
- `server/courtroom_state.py` (27K) — World state + jury ✅
- `server/adversa_environment.py` (257 lines) — Core env ✅
- `server/app.py` (7.3K) — FastAPI server ✅
- `tasks.py` (6.9K) — Task graders ✅
- `inference.py` (13.8K) — Baseline agent ✅
- `client.py` (3.5K) — HTTP client ✅
- `train.py` (25.2K) — Training script + plots ✅
- `Dockerfile` ✅
- `openenv.yaml` ✅
- `README.md` (11.9K) ✅
- `plots/adversa_training_results.png` ✅

## AUDIT CHECKLIST — Go through EVERY item:

### 1. 🏃 Runtime Test
- [ ] Start the server: `cd /Users/rajesh/Desktop/Adversa && uvicorn server.app:app --reload --port 7860`
- [ ] Test `/health` endpoint
- [ ] Test `/reset` with `{"seed": 42, "options": {"case_id": "C1", "role": "defense"}}`
- [ ] Test `/step` with a valid action
- [ ] Test `/state` returns full state
- [ ] Test `/cases` lists all 10 cases
- [ ] Test `/cases/C1` returns case details
- [ ] Run a FULL 30-step episode programmatically and verify it terminates correctly

### 2. 🔗 Import Chain
- [ ] Does `models.py` import cleanly?
- [ ] Does `server/courtroom_state.py` import from `models.py` correctly?
- [ ] Does `server/adversa_environment.py` import from both?
- [ ] Does `server/app.py` wire everything together?
- [ ] Any circular imports?

### 3. 🎯 OpenEnv Compliance
- [ ] `reset()` returns `{"observation": {...}}` (observation MUST be nested under key)
- [ ] `step()` returns `{"observation": {...}, "reward": float, "done": bool, "info": {...}}`
- [ ] `state()` returns full internal state dict
- [ ] All 10 cases load without errors
- [ ] Seeds produce reproducible episodes (same seed → same initial state)

### 4. ⚖️ Reward Correctness
- [ ] Dense rewards fire every step (jury shifts, timing, procedure)
- [ ] Verdict reward ONLY fires when `done=True`
- [ ] Reward values are in expected ranges (-5.0 to +5.0)
- [ ] Invalid actions correctly give -0.5 penalty
- [ ] Evidence timing bonus (+0.8) triggers when strong evidence (>0.7) presented 3rd or later

### 5. 👥 Multi-Agent / Role Isolation
- [ ] Prosecutor observation does NOT contain defense evidence
- [ ] Defense observation does NOT contain prosecution evidence
- [ ] Judge observation has `objection_pending` when objection raised
- [ ] `available_actions` changes correctly by phase AND role
- [ ] Phase transitions happen at correct step boundaries

### 6. 🧠 Jury System
- [ ] 3 juror profiles exist (analytical, empathetic, skeptical)
- [ ] Different framings produce different jury shifts
- [ ] Factual framing moves analytical juror most
- [ ] Emotional framing moves empathetic juror most
- [ ] Consistent arguments give skeptical juror consistency bonus
- [ ] Jury votes at end produce correct verdict (2/3 majority)
- [ ] Ground truth is checked against verdict

### 7. 🐳 Docker Build
- [ ] `docker build -t adversa .` succeeds
- [ ] `docker run -p 7860:7860 adversa` starts and responds on `/health`
- [ ] No missing dependencies in `requirements.txt`

### 8. 📝 README Completeness
- [ ] Architecture diagram present
- [ ] Demo case walkthrough (The Teacher) present
- [ ] Results table with before/after numbers
- [ ] All 4 submission links have placeholder URLs
- [ ] Quick start instructions work
- [ ] Theme alignment section present

### 9. 📊 Training Script
- [ ] `train.py` generates plots when run standalone (`python train.py`)
- [ ] Plot file exists: `plots/adversa_training_results.png`
- [ ] 3 subplots: reward curve, verdict accuracy bar, per-juror persuasion
- [ ] GRPO training code (currently commented) is syntactically valid
- [ ] Reward function connects to environment via HTTP

### 10. 🚨 Submission Blockers
- [ ] All `YOUR_USERNAME` and `YOUR_NOTEBOOK_ID` placeholders identified — need to be replaced before submission
- [ ] Google Form requires: HF Space URL, Colab link, GitHub URL, Video/Blog URL
- [ ] README must contain ALL 4 links

## OUTPUT FORMAT

After auditing, produce:
1. **✅ PASS items** — working correctly
2. **⚠️ WARNING items** — works but could be improved
3. **❌ FAIL items** — must fix before submission
4. **Priority fix list** — ordered by importance (submission blockers first)

Then FIX every ❌ FAIL item immediately.
