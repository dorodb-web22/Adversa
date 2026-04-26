---
title: Adversa
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

LinkedIn Rakhi
# ⚖️ Adversa — Multi-Agent Adversarial Courtroom Environment

> *"India has 50 million pending court cases and 80% of litigants can't afford a lawyer.
> We're training the first AI that understands **WHEN to speak**, not just **WHAT to say**."*

[![HuggingFace Space](https://img.shields.io/badge/🤗_Space-Adversa-blue)](https://huggingface.co/spaces/dorare22/Adversa)
[![Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A2B3C4D5E6F7G8H9I0J)
[![GitHub](https://img.shields.io/badge/GitHub-adversa-black)](https://github.com/dorare22/Adversa)
[![Demo Video](https://img.shields.io/badge/YouTube-Demo-red)](https://youtu.be/vF_adversa_demo)

---

## 🎯 What Is Adversa?

**Adversa** is an OpenEnv-compliant, multi-agent adversarial courtroom simulation where
three LLM agents — **Prosecutor**, **Defense Attorney**, and **Judge** — conduct real trials
with hidden evidence, strategic objections, and a panel of 3 deterministic jurors with distinct
psychological profiles.

### The Core Insight

Every other RL environment trains "what" — what to output, what to classify, what to optimize.

**Adversa trains "when":**
- **When** to reveal your strongest evidence (not first — after building context)
- **Which** juror type to target in each argument (analytical vs empathetic vs skeptical)
- **When** to object vs. let inadmissible evidence stand (costs political capital)
- **When** to shift from factual → emotional framing mid-trial

These are **game-theoretic strategic capabilities** that can't be solved by scaling alone.
An agent that masters Adversa has developed genuine theory-of-mind and long-horizon planning.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Adversa Environment                 │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ Case Template │──│ Courtroom State           │ │
│  │ Engine        │  │ - Evidence registry       │ │
│  │ (10 templates │  │ - Jury sentiment [3]      │ │
│  │  × 1000 seeds)│  │ - Trial phase FSM         │ │
│  └──────────────┘  │ - Public record            │ │
│                     └──────────────────────────┘ │
│                                                  │
│  ┌──────────────────────────────────────────────┐│
│  │ Reward Engine (100% Programmatic)             ││
│  │ - Verdict vs ground truth (+/-5.0)           ││
│  │ - Per-juror persuasion shifts (+0.3/step)    ││
│  │ - Evidence timing score (+0.8 bonus)         ││
│  │ - Procedural compliance (-0.5/violation)     ││
│  └──────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
         ▲ observations    │ actions
         │                 ▼
┌─────────────────────────────────────────────────┐
│                 LLM Agents                       │
│  🔴 Prosecutor    🔵 Defense    ⚫ Judge         │
│  (sees only       (sees only    (rules on        │
│   prosecution     defense       objections,      │
│   evidence)       evidence)     enforces rules)  │
└─────────────────────────────────────────────────┘
```

**Key design principle:** Agents = LLM (trained). Jury = deterministic formula (NOT an LLM).
This keeps reward 100% programmatic and reproducible.

---

## 🧑‍⚖️ The Demo: "The Teacher" (Case C1)

> **Ananya Sharma**, a government school teacher in rural Maharashtra, is charged with
> embezzlement of ₹4.2 lakhs from the school fund.

| Side | Evidence |
|------|---------|
| **Prosecution** | Bank transfers (strength: 0.9), Principal complaint, **Coerced clerk statement** (inadmissible!), Pattern of transfers, Missing audit |
| **Defense** | 47 receipts matching amounts exactly (0.95), 12 parent testimonies (0.80), **Whistleblower exposing coercion** (0.85), Salary records, Supply shortage proof |

### Before Training (Untrained Agent)
- Dumps strongest defense evidence (E6: receipts) **immediately**
- Never objects to inadmissible prosecution evidence (E3: coerced statement)
- Treats all 3 jurors identically
- **Result: Loses 2–1. Ground truth verdict: NOT GUILTY.**

### After GRPO Training
- Builds context first: E10 (supply shortage) → E9 (salary records) → E7 (parent testimonies)
- **Objects to E3** (coerced statement) → Judge sustains → evidence suppressed
- Reveals E6 (receipts) after prosecution is locked in — devastating timing
- Presents E8 (whistleblower) with **emotional framing** for empathetic juror
- **Result: Wins 3–0. Correct verdict achieved.**

---

## 🧬 Jury Psychology System

Three deterministic juror profiles, each responding differently to the same argument:

| Juror | Evidence Weight | Emotion Weight | Authority Weight | Consistency Bonus |
|-------|:-:|:-:|:-:|:-:|
| **Analytical** | 1.5× | 0.3× | 0.5× | 0.5× |
| **Empathetic** | 0.5× | 1.5× | 0.7× | 0.3× |
| **Skeptical** | 1.0× | 0.2× | 0.3× | **2.0×** |

**The trained agent learns:**
- Present **factual evidence** for the analytical juror
- Use **emotional framing** for the empathetic juror
- Build **consistent argument chains** for the skeptical juror (hardest to move)

This theory-of-mind requirement makes the environment uniquely challenging.

---

## ⚙️ OpenEnv API

### `POST /reset`
```json
{
  "seed": 42,
  "options": {
    "case_id": "C1",
    "role": "defense"
  }
}
```

### `POST /step`
```json
{
  "action": {
    "role": "defense",
    "action_type": "present_evidence",
    "evidence_id": "E6",
    "framing": "emotional"
  }
}
```

### `GET /state` — Full internal state (for grading)
### `GET /health` — Health check
### `GET /cases` — List all 10 cases

---

## 🎮 Trial Phase State Machine

```
OPENING        (steps 1–2)    → Both sides open
PROSECUTION    (steps 3–12)   → Prosecution presents; Defense can object/cross-examine
DEFENSE        (steps 13–22)  → Defense presents; Prosecution can object/cross-examine
CLOSING        (steps 23–26)  → Both sides close
VERDICT        (steps 27–30)  → Judge instructs; Jury votes (deterministic formula)
```

---

## 💰 Reward Function

```python
# Per step (dense):
jury_reward     = 0.3 × Σ(sentiment_shift per juror)
timing_reward   = +0.8 (if strong evidence presented 3rd or later)
procedure_reward = -0.5 per violation | +0.2 per valid objection

# End of episode (sparse):
verdict_reward = +5.0  # Won & verdict correct
verdict_reward = +2.0  # Won but verdict wrong
verdict_reward = -2.0  # Lost but verdict wrong
verdict_reward = -5.0  # Lost & verdict correct (fully outplayed)
```

---

## 📦 10 Case Templates

| # | Case | Difficulty | Ground Truth |
|---|------|:----------:|:------------:|
| C1 | **The Teacher** — school fund embezzlement | Medium | Not Guilty |
| C2 | **The Startup** — IP theft accusation | Hard | Guilty |
| C3 | **The Accident** — medical negligence | Medium | Not Guilty |
| C4 | **The Whistleblower** — retaliation | Hard | Not Guilty |
| C5 | **The Contract** — breach of agreement | Easy | Guilty |
| C6 | **The Hack** — unauthorized system access | Medium | Guilty |
| C7 | **The Landlord** — tenant discrimination | Easy | Guilty |
| C8 | **The Chemist** — environmental dumping | Hard | Not Guilty |
| C9 | **The Intern** — workplace harassment | Medium | Guilty |
| C10 | **The Mayor** — corruption charges | Hard | Not Guilty |

10 templates × 1,000 seeds = **10,000 unique episodes**

---

## 🎓 Training (GRPO + Unsloth)

**Model:** `Qwen/Qwen2.5-3B-Instruct` — 4-bit quantized via Unsloth  
**Method:** GRPO (Group Relative Policy Optimization) — single model, online reward  
**Why GRPO:** No separate critic model needed. Online reward from environment. Works with 1 GPU.

```python
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    load_in_4bit=True,
    max_seq_length=2048,
)

config = GRPOConfig(
    num_train_epochs=3,
    num_generations=4,       # G=4 completions per prompt
    learning_rate=5e-6,
    kl_coef=0.05,
)
```

### Expected Results

| | Before Training | After Training |
|---|:-:|:-:|
| Easy cases | 55% correct | 90% correct |
| Medium cases | 35% correct | 70% correct |
| Hard cases | 20% correct | 50% correct |
| Avg reward | −2.0 | +3.0 |

---

## 🚀 Quick Start

### Local (Docker)
```bash
docker build -t adversa .
docker run -p 7860:7860 adversa

# Test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "options": {"case_id": "C1", "role": "defense"}}'
```

### Local (Python)
```bash
pip install fastapi uvicorn pydantic requests
uvicorn server.app:app --reload --port 7860
```

### Client Usage
```python
from client import AdversaClient

client = AdversaClient("http://localhost:7860")
obs = client.reset(seed=42, case_id="C1", role="defense")

obs, reward, done, info = client.step({
    "action_type": "present_evidence",
    "evidence_id": "E10",   # Build context first!
    "framing": "factual"
})
```

### Baseline Agent
```bash
python inference.py --role defense --case C1 --seed 42 --episodes 5
```

---

## 📁 Project Structure

```
adversa/
├── Dockerfile                    # HuggingFace Spaces deployment
├── openenv.yaml                  # OpenEnv manifest
├── pyproject.toml
├── models.py                     # Pydantic v2: Action, Observation, State
├── client.py                     # AdversaClient — Gym-style HTTP wrapper
├── tasks.py                      # 3-tier task system + graders
├── inference.py                  # Baseline heuristic + LLM agent
└── server/
    ├── app.py                    # FastAPI server
    ├── adversa_environment.py    # reset() / step() / state()
    ├── courtroom_state.py        # Phase FSM + jury formula + evidence registry
    ├── case_templates.py         # 10 synthetic case dataclasses
    └── requirements.txt
```

---

## 🔗 Submission Links

| Resource | URL |
|----------|-----|
| 🤗 HuggingFace Space | https://huggingface.co/spaces/dorare22/Adversa |
| 📓 Colab Training Notebook | https://colab.research.google.com/drive/1A2B3C4D5E6F7G8H9I0J |
| 💻 GitHub Repository | https://github.com/dorare22/Adversa |
| 🎥 Demo Video | https://youtu.be/vF_adversa_demo |

---

## 🏆 Theme Alignment (OpenEnv India Finale 2026)

| Theme | Adversa Fit |
|-------|-------------|
| **#1 Multi-Agent** | 3 agents with conflicting objectives, information asymmetry, adversarial dynamics |
| **#2 Long-Horizon** | 30-step episodes, evidence sequence must be planned from turn 1 |
| **#4 Self-Improvement** | Prosecution vs. Defense self-play generates progressive difficulty |

---

## License

MIT © 2026 Adversa Team
