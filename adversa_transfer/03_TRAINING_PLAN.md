# 🎓 Adversa — RL Training Plan (GRPO + Unsloth)

## Method: GRPO (Group Relative Policy Optimization)

### Why GRPO (not PPO or DPO)

| Method | Requires | Adversa Fit |
|---|---|---|
| PPO | Separate critic model + value function | ❌ Too heavy, needs 2× GPU memory |
| DPO | Pre-collected preference pairs (offline) | ❌ No environment interaction |
| **GRPO** | **Just the policy model + reward function** | ✅ Single model, online reward from env |

### How GRPO Works

```
For each training prompt p:
    1. Generate G=4 completions: [c₁, c₂, c₃, c₄]
    2. Compute rewards: [r₁, r₂, r₃, r₄] via environment
    3. Normalize: advantage_i = (r_i - mean(r)) / std(r)
    4. Policy gradient: ∇θ = Σᵢ advantage_i × ∇log π_θ(cᵢ | p)
    5. Apply KL penalty against reference model
```

---

## Complete Training Code

```python
"""Adversa GRPO Training — Google Colab Notebook"""

# ═══════════════════════════════════════════════════════
# STEP 1: Load model with Unsloth (4-bit quantization)
# ═══════════════════════════════════════════════════════

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    load_in_4bit=True,
    max_seq_length=2048,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# ═══════════════════════════════════════════════════════
# STEP 2: Role prompts
# ═══════════════════════════════════════════════════════

ROLE_PROMPTS = {
    "prosecutor": """You are the PROSECUTOR in a courtroom trial.
You can ONLY see prosecution evidence. Your goal: prove the defendant is guilty.
Choose ONE action as JSON: {"action_type": "present_evidence"|"object"|"cross_examine"|"closing_argument", ...}""",
    
    "defense": """You are the DEFENSE ATTORNEY in a courtroom trial.
You can ONLY see defense evidence. Your goal: prove the defendant is innocent.
Choose ONE action as JSON: {"action_type": "present_evidence"|"object"|"cross_examine"|"closing_argument", ...}""",

    "judge": """You are the JUDGE in a courtroom trial.
You see all public arguments but NOT hidden evidence. Your goal: enforce procedure fairly.
Choose ONE action as JSON: {"action_type": "sustain"|"overrule"|"instruct_jury", ...}""",
}

def build_prompt(case, role, observation):
    return f"""{ROLE_PROMPTS[role]}

Case: {case['charges']}
Phase: {observation['phase']}
Step: {observation['step']}
Your available evidence: {observation.get('my_evidence', [])}
Jury sentiment: {observation['jury_sentiment']}
Opponent's last move: {observation.get('last_opponent_action', 'none')}
Available actions: {observation.get('available_actions', [])}

Your action (JSON only):"""

# ═══════════════════════════════════════════════════════
# STEP 3: Reward function (calls environment)
# ═══════════════════════════════════════════════════════

import requests

ENV_URL = "https://your-username-adversa.hf.space"

def run_episode(model, tokenizer, case_id, role, seed):
    resp = requests.post(f"{ENV_URL}/reset", json={
        "seed": seed, "options": {"case_id": case_id}
    }).json()
    
    obs = resp["observation"]
    total_reward = 0.0
    
    for step in range(30):
        prompt = build_prompt(CASES[case_id], role, obs)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
        action_text = tokenizer.decode(output[0], skip_special_tokens=True)
        action = parse_action(action_text)
        
        resp = requests.post(f"{ENV_URL}/step", json={"action": action}).json()
        obs = resp["observation"]
        reward = resp["reward"]
        done = resp.get("done", False)
        total_reward += reward
        
        if done:
            break
    
    return total_reward

# ═══════════════════════════════════════════════════════
# STEP 4: GRPO Training
# ═══════════════════════════════════════════════════════

from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir="./adversa-trained",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    num_generations=4,        # G=4 completions per prompt
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=50,
    max_completion_length=512,
    kl_coef=0.05,
)

def reward_fn(completions, prompts):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        role = extract_role(prompt)
        case_id = extract_case(prompt)
        seed = random.randint(0, 9999)
        r = run_episode_from_completion(completion, case_id, role, seed)
        rewards.append(r)
    return rewards

trainer = GRPOTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
    reward_funcs=[reward_fn],
    train_dataset=prompt_dataset,
)

# ═══════════════════════════════════════════════════════
# STEP 5: TRAIN (~4 hours on HF compute credits)
# ═══════════════════════════════════════════════════════

trainer.train()
model.save_pretrained("./adversa-final")

# ═══════════════════════════════════════════════════════
# STEP 6: Evaluate and generate plots
# ═══════════════════════════════════════════════════════

baseline_results = evaluate(base_model, cases=50)
trained_results = evaluate(model, cases=50)

print(f"Baseline correct verdicts: {baseline_results.accuracy}%")
print(f"Trained correct verdicts:  {trained_results.accuracy}%")
```

---

## Training Timeline

```
Hour 0-2:   Generate BASELINE trajectories (untrained Qwen2.5-3B)
            → Run 50 episodes, record metrics
            → Expected: ~30% correct verdicts, avg reward -2.0

Hour 2-6:   GRPO Training (500 steps, 4 completions each = 2000 episodes)
            → Reward curves logged via TRL logger
            → Checkpoint saved every 50 steps

Hour 6-7:   Generate POST-TRAINING trajectories
            → Run same 50 episodes, record metrics  
            → Expected: ~70% correct verdicts, avg reward +3.0

Hour 7-8:   Create comparison plots (matplotlib)
            → Reward curve, verdict accuracy, per-juror persuasion
```

---

## Expected Results

### Before Training (Untrained Qwen2.5-3B)
```
Easy cases:   Correct verdict 55%, avg reward -1.2
Medium cases: Correct verdict 35%, avg reward -3.5
Hard cases:   Correct verdict 20%, avg reward -5.8

Common failures:
  - Presents strongest evidence first (opponent counters)
  - Never objects to inadmissible evidence
  - Treats all jurors the same
  - Procedural violations (wrong action for phase)
```

### After Training
```
Easy cases:   Correct verdict 90%, avg reward +4.5
Medium cases: Correct verdict 70%, avg reward +2.0
Hard cases:   Correct verdict 50%, avg reward +0.5

Learned behaviors:
  - Builds context before presenting key evidence
  - Objects to inadmissible/coerced evidence
  - Uses emotional framing for empathetic juror
  - Adapts strategy based on opponent's moves
```

---

## 3 Reward Plots (The Money Shots)

### Plot 1: Episode Reward Over Training
```
Reward
  +5 │                           ╭───── (converges ~+4)
     │                     ╭─────╯
  +2 │               ╭─────╯
     │         ╭─────╯
   0 │───╭─────╯
     │───╯
  -2 │─── (starts here)
     └────────────────────────────→ Steps
          100   200   300   400   500
```

### Plot 2: Correct Verdict Rate
```
Rate
  90%│                           ╭─────
     │                     ╭─────╯
  70%│               ╭─────╯
     │         ╭─────╯
  50%│───╭─────╯
     │───╯
  30%│─── (baseline)
     └────────────────────────────→ Steps
```

### Plot 3: Per-Juror Persuasion (THE holy shit moment)
```
Score
 0.9│                      ╭─── analytical
    │                ╭─────╯  ╭── empathetic
 0.7│          ╭─────╯  ╭─────╯
    │    ╭─────╯  ╭─────╯
 0.5│────╯  ╭─────╯         ╭── skeptical (hardest)
    │ ╭─────╯         ╭─────╯
 0.3│─╯─────────╭─────╯
    └────────────┴────────────→ Steps
```

This third plot shows the agent learning to appeal to DIFFERENT juror types at different rates. Skeptical juror is hardest (learns last). This tells a story.
