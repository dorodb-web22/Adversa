# 📋 Official Hackathon Rules — OpenEnv India Finale (Round 2)

## Event: April 25-26, 2026 (48 hours, in-person, Bangalore)

---

## Round 2 Themes

### Theme #1 — Multi-Agent Interactions
Environments involving cooperation, competition, negotiation, and coalition formation. Agents must model beliefs and incentives of others in partially observable settings. Drives theory-of-mind reasoning and emergent strategic behavior.

**Expected Outcome:** An environment that can be used to train multi-agent task handling in an LLM.

**Example environments:** Market simulations, compute-allocation negotiations, collaborative puzzle worlds, mixed cooperative/competitive strategy games.

### Theme #2 — (Super) Long-Horizon Planning & Instruction Following
Environments requiring deep, multi-step reasoning with sparse or delayed rewards. Agents must decompose goals, track state over extended trajectories, and recover from early mistakes.

**Expected Outcome:** An environment that captures and improves LLM behavior on challenging long-horizon tasks beyond context memory limits.

**Example environments:** Research-planning simulators, large-scale codebase refactoring, strategic resource management, long-horizon logistics optimization, extremely complicated long-horizon instruction following (e.g., 300 instructions scattered around).

### Theme #3 — World Modeling

#### #3.1 Professional Tasks
Environments requiring real interaction with tools, APIs, or dynamic systems where the model does real hard work instead of exploiting shortcuts. Strengthens causal reasoning and persistent world models.

**Expected Outcome:** An environment capturing nuances of a defined partially observable world and improve LLM interaction with it.

**Example environments:** Dynamic browser/API ecosystems, enterprise applications, scientific workflow loops, economic simulations with feedback, tool-discovery benchmarks.

#### #3.2 Personalized Tasks
Environments offering real personalized task handling — replying to personal messages, handling dinner conflicts due to work conflicts, replying to tough emails. Any personal assistant tasks.

**Expected Outcome:** An environment giving the model a realistic simulation of handling personal tasks, conflicts and managing them as delegations.

**Example environments:** Executive Assistant Meeting Planner, dinner and drive planning, email and message replying, shopping.

### Theme #4 — Self-Improvement
Environments where agents learn to generate new challenges, escalate difficulty, and improve through self-play or adaptive curricula. Rather than optimizing fixed tasks, agents learn to drive their own capability growth. Recursive skill amplification.

**Expected Outcome:** An environment for improving self-play of an LLM over a defined set of tasks.

**Example environments:** Self-play negotiation arenas, auto-generated math/proof tasks, evolving coding competitions, adaptive RL curricula.

### Theme #5 — Wild Card — Impress Us!
Not limited to the above. Will reward out-of-box tasks. Be creative but must meaningfully add value to LLM training on a certain task.

---

## Judging Criteria (Official Weights)

| Criterion | Weight | Description |
|---|---|---|
| **Environment Innovation** | **40%** | Is the environment novel, creative, or genuinely challenging? Does it meaningfully test agent behavior? |
| **Storytelling & Presentation** | **30%** | Can you clearly explain the problem and what the agent learned? Is the demo engaging for non-technical audience? |
| **Showing Improvement in Rewards** | **20%** | Observable evidence of training progress — reward curves, before/after behavior, comparison against baseline |
| **Reward & Training Pipeline** | **10%** | Is the reward logic coherent? Does the pipeline produce meaningful improvement in agent behavior? |

---

## Submission Requirements

A Google Form shared on April 26th requires:
1. **HuggingFace Space URL** — deployed, working environment
2. **Colab Notebook link** — training script
3. **Code repository link** — public GitHub
4. **YouTube video URL or HuggingFace blog post URL** — demo
5. **README file** — MUST contain all URLs and links above

---

## Important Guidelines
- It is NOT mandatory to choose the same problem statement as Round 1
- Only choose Round 1 problem if it aligns with Round 2 themes
- You can start working on your problem statement once finalized
- Post-training to be done onsite on 25th & 26th when you receive HuggingFace compute credits

---

## Key Quote from Organizers
> "A messy but ambitious environment with real training evidence beats a polished but boring one."
> "Pick a problem that excites you (that energy comes through in the pitch)."
> "Does this environment exist to teach an LLM something it currently can't do well?"

---

## Schedule (Day 1 — April 25)
- 7:00-10:30 AM — Registration & Arrival
- 8:00-9:15 AM — Opening Ceremony
- 10:00-10:15 AM — Problem Themes Overview & Briefing
- 10:15-10:30 AM — Move to Build Zones
- 10:30-11:00 AM — Address by META Team
- 11:30 AM — Hacking Begins
- 1:00 PM — Mentor Round 1
- 3:30-4:30 PM — Mentor Round 2
- 5:00-5:30 PM — Talk + High Tea
- 8:00-10:00 PM — Dinner
- 2:00 AM — Midnight Snacks

## Schedule (Day 2 — April 26)
- 8:00 AM — Breakfast
- 10:00 AM-12:00 PM — Mentor Round 3 (Final)
- 12:00 PM — Lunch
- 2:00 PM — 5-Hour Reminder: Submission Deadline
- 3:00 PM — ⏰ 2-Hour Reminder
- 3:30-4:30 PM — Submission Deadline
- 5:00 PM — Closing Remarks
- 5:15 PM — Event Concludes

---

## Prize Pool: $30,000
- 1st place: $7,500
- 2nd place: $5,000
- 3rd place: $3,500
- 4th-8th: $2,000 each
- 9th-15th: $650 each

**Winners get direct interview opportunity with Meta and HuggingFace AI teams.**
