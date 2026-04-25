"""
Adversa — OpenEnv Client

Thin Python wrapper around the Adversa HTTP API.
Provides a clean Gym-like interface for training scripts.

Usage:
    from client import AdversaClient
    client = AdversaClient("https://your-hf-space.hf.space")
    obs = client.reset(seed=42, case_id="C1", role="defense")
    obs, reward, done, info = client.step({
        "action_type": "present_evidence",
        "evidence_id": "E6",
        "framing": "emotional"
    })
"""
from __future__ import annotations

import requests
from typing import Any, Optional


class AdversaClient:
    """HTTP client for the Adversa environment."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._role = "defense"

    def health(self) -> dict:
        return self._get("/health")

    def reset(
        self,
        seed: int = 0,
        case_id: Optional[str] = None,
        role: str = "defense",
    ) -> dict:
        self._role = role
        options: dict[str, Any] = {"role": role}
        if case_id:
            options["case_id"] = case_id
        resp = self._post("/reset", {"seed": seed, "options": options})
        return resp["observation"]

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        action.setdefault("role", self._role)
        resp = self._post("/step", {"action": action})
        return (
            resp["observation"],
            float(resp["reward"]),
            bool(resp["done"]),
            resp.get("info", {}),
        )

    def state(self) -> dict:
        return self._get("/state")

    def cases(self) -> dict:
        return self._get("/cases")

    def _post(self, path: str, data: dict) -> dict:
        r = requests.post(f"{self.base_url}{path}", json=data, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict:
        r = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ── Gym-style helpers ────────────────────────────────────────────────────

    def run_episode(
        self,
        role: str = "defense",
        case_id: Optional[str] = None,
        seed: int = 0,
        action_fn=None,
        max_steps: int = 30,
    ) -> dict:
        """Run a full episode with a given action function."""
        obs = self.reset(seed=seed, case_id=case_id, role=role)
        trajectory = []
        total_reward = 0.0

        for _ in range(max_steps):
            if action_fn:
                action = action_fn(obs, role)
            else:
                action = {"action_type": "pass", "role": role}

            obs, reward, done, info = self.step(action)
            total_reward += reward
            trajectory.append({
                "obs": obs,
                "reward": reward,
                "done": done,
                "info": info,
            })
            if done:
                break

        final_state = self.state()
        return {
            "trajectory": trajectory,
            "total_reward": total_reward,
            "verdict": final_state.get("verdict"),
            "verdict_correct": final_state.get("verdict_correct"),
            "jury_sentiment": final_state.get("jury_sentiment"),
        }
