# ⏰ Adversa — 48-Hour Execution Timeline

## Phase 1: Foundation (Hours 0–10)

| Hour | Task | Deliverable |
|---|---|---|
| 0–1 | Scaffold OpenEnv project. Set up `models.py`, `openenv.yaml`, `server/app.py` | Skeleton that starts, `/health` responds |
| 1–3 | Build case template system: `CaseTemplate`, `Evidence`, `JurorProfile` dataclasses + 10 case templates | `server/case_templates.py` complete |
| 3–5 | Build `CourtroomState` — evidence registry, public record, jury sentiment tracker, phase state machine | `server/courtroom_state.py` |
| 5–7 | Implement `AdversaObservation` and `AdversaAction` Pydantic models with role-specific fields | `models.py` complete |
| 7–10 | Implement `AdversaEnvironment.reset()` and `.step()` with phase transitions | Environment runs end-to-end |

## Phase 2: Core Logic (Hours 10–20)

| Hour | Task | Deliverable |
|---|---|---|
| 10–13 | Role-specific action processing: evidence presentation, objection handling, cross-examination, verdict | Full step() logic for all 3 roles |
| 13–15 | Reward function with all components (verdict, evidence timing, procedural, jury shifts) | `tasks.py` with 3 task difficulties |
| 15–17 | Build `inference.py` — baseline agent with role-specific system prompts | Baseline runs all tasks |
| 17–20 | Docker build + test + deploy to HuggingFace Spaces | Live at HF Space URL |

## Phase 3: Training (Hours 20–34)

| Hour | Task | Deliverable |
|---|---|---|
| 20–23 | Write GRPO training script in Colab. Connect to HF environment. | `train.ipynb` |
| 23–25 | Generate baseline trajectories (untrained). Record metrics. | Baseline scores + distributions |
| 25–31 | **TRAIN** with HF compute credits. Qwen2.5-3B via Unsloth 4-bit. ~500 steps. | Trained checkpoint |
| 31–34 | Generate post-training trajectories. Create comparison plots (matplotlib). | Before/after `.png` plots |

## Phase 4: Polish (Hours 34–48)

| Hour | Task | Deliverable |
|---|---|---|
| 34–37 | Write README with motivation, architecture, results, all links | README.md |
| 37–39 | Record <2min video OR write HF blog post | Submission media |
| 39–42 | Edge cases, error handling, code cleanup | Production quality |
| 42–45 | Final deployment with all assets to HF Space | Complete submission |
| 45–48 | Buffer + fill Google Form + pitch rehearsal | SUBMITTED ✅ |

---

## Build Order (File-by-File)

```
1. models.py                    ← First (everything depends on this)
2. server/case_templates.py     ← Second (environment needs cases)
3. server/courtroom_state.py    ← Third (environment needs state)
4. server/adversa_environment.py ← Fourth (core logic)
5. server/app.py                ← Fifth (FastAPI wrapper)
6. openenv.yaml                 ← Sixth (manifest)
7. tasks.py                     ← Seventh (graders)
8. inference.py                 ← Eighth (baseline agent)
9. Dockerfile                   ← Ninth (deployment)
10. README.md                   ← Last (needs results)
```

---

## Risk Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Case templates too simple | Low innovation | Pre-build 10 diverse templates with rich evidence sets |
| Multi-agent training doesn't converge | Can't show improvement | Train prosecution role only if needed (simpler) |
| Reward too sparse | Slow learning | Dense per-step signals (jury shifts, objection quality) |
| HF Space deployment fails | No demo | Test Docker locally first, deploy early |
| Not enough training time | Weak results | Start training by Hour 20, not later |
