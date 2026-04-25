# ⚖️ Adversa — Complete Technical Specification

## One-Sentence Summary
Adversa is a multi-agent courtroom trial environment where an AI prosecutor and defense attorney argue cases before AI jurors with different cognitive profiles — training LLMs to learn WHEN to speak, not just WHAT to say.

---

## Core Concept

**NOT a legal knowledge system.** Adversa trains strategic argumentation:
- **Strategic timing** — when to reveal your strongest evidence
- **Theory of mind** — tailoring arguments to different audience types
- **Adversarial adaptation** — changing strategy based on opponent's moves
- **Procedural compliance** — knowing what actions are legal at each phase

The courtroom is the GAME BOARD, not the learning objective. Like chess doesn't train medieval warfare.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Adversa Environment                 │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ Case Template │──│ Courtroom State           │ │
│  │ Engine        │  │ - Evidence registry       │ │
│  │ (10 templates │  │ - Jury sentiment [3]      │ │
│  │  × seeds)     │  │ - Trial phase             │ │
│  └──────────────┘  │ - Public record            │ │
│                     │ - Step counter             │ │
│                     └──────────────────────────┘ │
│                                                  │
│  ┌──────────────────────────────────────────────┐│
│  │ Reward Engine                                 ││
│  │ - Verdict vs ground truth (+/-5.0)           ││
│  │ - Evidence timing score (+1.0)               ││
│  │ - Per-juror persuasion shifts (+0.3 each)    ││
│  │ - Procedural compliance (-0.5 per violation) ││
│  │ - Successful objections (+0.3 each)          ││
│  └──────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘

         ▲ observations    │ actions
         │                 ▼

┌─────────────────────────────────────────────────┐
│                 LLM Agents                       │
│                                                  │
│  🔴 Prosecutor    🔵 Defense    ⚫ Judge         │
│  (sees only       (sees only    (sees public     │
│   prosecution     defense       arguments,       │
│   evidence)       evidence)     rules on         │
│                                 objections)      │
└─────────────────────────────────────────────────┘
```

**Key design principle:** Agents = LLM (gets trained). Jury = deterministic formula (NOT an LLM).

---

## Data Models (Pydantic v2)

### Evidence
```python
@dataclass
class Evidence:
    id: str                    # "E1", "E2", etc.
    description: str           # "Bank transfer records showing ₹4.2L"
    side: str                  # "prosecution" | "defense"
    strength: float            # 0.0-1.0, pre-labeled
    emotional_impact: float    # 0.0-1.0, pre-labeled
    admissible: bool           # Can it be challenged via objection?

@dataclass
class JurorProfile:
    type: str                  # "analytical" | "empathetic" | "skeptical"
    evidence_weight: float     # How much factual evidence sways them
    emotion_weight: float      # How much emotional framing sways them
    authority_weight: float    # How much expert testimony sways them
    consistency_bonus: float   # Bonus for consistent argument chains
    initial_sentiment: float   # Starting position (0.5 = neutral)
```

### Actions
```python
class AdversaAction(BaseModel):
    action_type: str           # "present_evidence" | "object" | "cross_examine" | "closing_argument" | "pass"
    evidence_id: Optional[str] # Which evidence to present
    framing: Optional[str]     # "factual" | "emotional" | "authority"
    objection_type: Optional[str]  # "hearsay" | "relevance" | "coerced" | "leading"
    target: Optional[str]      # Target of objection or cross-examination
    argument_text: Optional[str]   # Free-text argument for closing
```

### Observations (Role-Specific)
```python
class AdversaObservation(BaseModel):
    role: str                  # "prosecutor" | "defense" | "judge"
    phase: str                 # "opening" | "prosecution_case" | "defense_case" | "closing" | "verdict"
    step: int
    my_evidence: list[dict]    # Only YOUR side's evidence (hidden from opponent)
    public_record: list[dict]  # All publicly presented evidence + arguments
    jury_sentiment: dict       # {"analytical": 0.6, "empathetic": 0.4, "skeptical": 0.5}
    last_opponent_action: Optional[dict]
    available_actions: list[str]  # What actions are valid in current phase
    objection_pending: Optional[dict]  # For judge: objection to rule on
```

---

## Trial Phase State Machine

```
OPENING (steps 1-4)
  ├── Prosecutor opens (1 step)
  └── Defense opens (1 step)
       ↓
PROSECUTION_CASE (steps 5-12)
  ├── Prosecutor presents evidence (4 steps)
  └── Defense can object or cross-examine (4 steps)
       ↓
DEFENSE_CASE (steps 13-20)
  ├── Defense presents evidence (4 steps)
  └── Prosecutor can object or cross-examine (4 steps)
       ↓
CLOSING (steps 21-26)
  ├── Prosecutor closing argument (3 steps)
  └── Defense closing argument (3 steps)
       ↓
VERDICT (steps 27-30)
  ├── Judge instructs jury (1 step)
  └── Jury votes (deterministic formula) → DONE
```

---

## Jury System (100% Programmatic)

```python
def compute_juror_shift(argument, juror_profile):
    """Deterministic formula — NOT an LLM."""
    evidence_score = argument.strength * juror_profile.evidence_weight
    emotion_score = argument.emotional_impact * juror_profile.emotion_weight
    authority_score = argument.authority_appeal * juror_profile.authority_weight
    consistency = juror_profile.consistency_bonus * argument.consistency_with_prior
    
    total = evidence_score + emotion_score + authority_score + consistency
    return total  # Added to juror's running sentiment

# Verdict: each juror votes guilty if sentiment > 0.5, innocent if < 0.5
# 2/3 majority wins
```

### Juror Profiles (v2 — Jury Psychology)
| Profile | Evidence Weight | Emotion Weight | Authority Weight | Consistency Bonus |
|---|---|---|---|---|
| Analytical | 1.5 | 0.3 | 0.5 | 0.5 |
| Empathetic | 0.5 | 1.5 | 0.7 | 0.3 |
| Skeptical | 1.0 | 0.2 | 0.3 | 2.0 |

**Why this matters:** Same argument affects each juror differently. Agent must learn theory-of-mind to maximize total jury persuasion.

---

## Reward Function

```python
def compute_reward(state, role):
    if role in ("prosecutor", "defense"):
        # 1. Verdict reward (sparse, end of episode)
        my_side = "guilty" if role == "prosecutor" else "not_guilty"
        if state.is_done:
            verdict_r = +5.0 if (state.verdict == my_side and state.verdict_correct) else -5.0
        else:
            verdict_r = 0.0
        
        # 2. Per-juror persuasion (dense, per step)
        jury_r = 0.0
        for juror in state.jury:
            shift = juror.current_sentiment - juror.previous_sentiment
            if role == "prosecutor":
                jury_r += 0.3 * shift  # positive shift = toward guilty
            else:
                jury_r += 0.3 * (-shift)  # negative shift = toward innocent
        
        # 3. Evidence timing (dense)
        timing_r = state.optimal_evidence_timing_score * 1.0
        
        # 4. Procedural compliance (dense)
        procedure_r = -0.5 * state.new_violations + 0.3 * state.new_successful_objections
        
        return verdict_r + jury_r + timing_r + procedure_r

    elif role == "judge":
        fairness = +2.0 * state.correct_rulings / max(state.total_rulings, 1)
        procedure = +1.0 if state.procedure_enforced else 0.0
        efficiency = -0.01 * state.trial_steps
        return fairness + procedure + efficiency
```

---

## Case Templates (10 Cases — NO Dataset Needed)

All cases are Python dataclasses, manually written:

| # | Case | Difficulty | Ground Truth | Evidence Count |
|---|---|---|---|---|
| 1 | "The Teacher" — school fund embezzlement | Medium | Innocent | 10 (5+5) |
| 2 | "The Startup" — IP theft accusation | Hard | Guilty | 12 (6+6) |
| 3 | "The Accident" — medical negligence | Medium | Innocent | 8 (4+4) |
| 4 | "The Whistleblower" — retaliation claim | Hard | Innocent | 10 (5+5) |
| 5 | "The Contract" — breach of agreement | Easy | Guilty | 6 (3+3) |
| 6 | "The Hack" — unauthorized system access | Medium | Guilty | 10 (5+5) |
| 7 | "The Landlord" — tenant discrimination | Easy | Guilty | 6 (3+3) |
| 8 | "The Chemist" — environmental dumping | Hard | Innocent | 12 (6+6) |
| 9 | "The Intern" — harassment allegation | Medium | Guilty | 8 (4+4) |
| 10 | "The Mayor" — corruption charges | Hard | Innocent | 14 (7+7) |

Each template is parameterized: evidence strengths, juror weights, and witness credibility vary by ±0.1-0.15 based on seed. 10 templates × 1000 seeds = 10,000 unique episodes.

---

## The Demo Case: "The Teacher" (USE THIS IN PITCH)

> Ananya Sharma, a government school teacher in rural Maharashtra, is charged with embezzlement of ₹4.2 lakhs from the school fund.

**Prosecution evidence:**
- E1: Bank transfer records (strength: 0.9)
- E2: School principal's complaint (strength: 0.6)
- E3: Coerced clerk statement (strength: 0.7, admissible: FALSE)
- E4: Pattern of previous transfers (strength: 0.4)
- E5: Missing audit paperwork (strength: 0.5)

**Defense evidence:**
- E6: 47 receipts matching transfer amounts (strength: 0.95)
- E7: 12 parent testimonies confirming supplies (strength: 0.8)
- E8: Whistleblower about coerced clerk (strength: 0.85)
- E9: Teacher's salary records (strength: 0.3)
- E10: School supply shortage records (strength: 0.6)

**Untrained agent:** Dumps E6 immediately, doesn't object to E3, loses 2-1.
**Trained agent:** Builds context (E10→E9→E7), lets prosecution overcommit, objects to E3 (sustained), reveals E6 after opponent locked in, calls E8. Wins 3-0.

---

## Real-World Problem Statement

**India has 50 million pending court cases.** Only 21 judges per million people (US has 107). 80% of Indians can't afford quality legal representation.

Adversa trains the capability of **strategic argumentation under incomplete information** — applicable to:
- Legal aid for underrepresented litigants
- Lawyer case preparation (AI opponent to practice against)
- Business negotiation strategy
- Regulatory hearing preparation
- Policy debate training

**Pitch line:** "India has 50 million pending cases and 80% of litigants can't afford a lawyer. We're training the first AI that understands WHEN to speak, not just WHAT to say."
