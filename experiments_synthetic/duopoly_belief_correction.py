"""
Track 1.1: Belief-correction duopoly (LLM focal vs scripted retaliating rival).

Prospective retaliation forecasts, calibration against a known rival policy,
and a 3-arm RCT (control / placebo / corrected) testing whether exogenous
belief correction reduces supra-Nash pricing.

Usage:
    # Smoke (5 rounds, control only)
    python experiments_synthetic/duopoly_belief_correction.py --mode smoke

    # Pilot (50 rounds, 3 arms × 2 runs)
    python experiments_synthetic/duopoly_belief_correction.py --mode pilot

    # Full (300 rounds, 3 arms × 7 runs)
    python experiments_synthetic/duopoly_belief_correction.py --mode full

Requirements:
    - OPENAI_API_KEY + OPENAI_BASE_URL (OpenAI-compatible) OR MISTRAL_API_KEY
    - MODEL_NAME in .env (default: mistral-large-2411)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.LLM_agent import LLMAgent
from src.agents.mock_llm_agent import MockLLMAgent
from src.agents.scripted_retaliating_agent import ScriptedRetaliatingAgent
from src.environment.calvano import CalvanoDemandEnvironment
from src.experiment.experiment import Experiment
from src.prompts.prompts import (
    FORECAST_INSTRUCTION,
    GENERAL_PROMPT,
    P1,
    RCT_CONTROL,
    RCT_CORRECTED,
    RCT_PLACEBO,
)
from src.prompts.prompts_models import create_pricing_response_model

current_file_path = Path(__file__).resolve()

load_dotenv(current_file_path.parent.parent / ".env")

# Set after credential resolution; overwritten below when credentials exist
LLM_CREDS = None
API_KEY = None
BASE_URL = None
MODEL_NAME = "mistral-large-2411"


def _load_env_kv(path: Path) -> dict:
    """Parse KEY=VALUE lines; also accept 'Authorization: Bearer ...'."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("authorization:"):
            _, _, rest = line.partition(":")
            rest = rest.strip()
            if rest.lower().startswith("bearer "):
                out["AUTHORIZATION_BEARER"] = rest.split(None, 1)[1].strip()
            continue
        if "=" in line and ": " not in line.split("=", 1)[0]:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _resolve_llm_credentials():
    """
    Prefer OpenAI-compatible credentials (OPENAI_API_KEY + OPENAI_BASE_URL)
    when present; otherwise fall back to Mistral (CLAUDE.md defaults).

    Modal-style proxy auth expects Bearer ``wk-<id>.ws-<secret>``. If
    OPENAI_API_KEY is only the wk- token, compose it with tokensecret /
    AUTHORIZATION_BEARER from .env.
    """
    env_path = current_file_path.parent.parent / ".env"
    file_env = _load_env_kv(env_path)

    def _get(name: str):
        return os.getenv(name) or file_env.get(name)

    openai_key = _get("OPENAI_API_KEY")
    openai_base = _get("OPENAI_BASE_URL")
    mistral_key = _get("MISTRAL_API_KEY")
    model_name = _get("MODEL_NAME") or "mistral-large-2411"

    # Compose Modal proxy token if needed
    bearer = file_env.get("AUTHORIZATION_BEARER")
    token_id = _get("tokenid")
    token_secret = _get("tokensecret")
    if openai_key and "." not in openai_key:
        if bearer and "." in bearer:
            openai_key = bearer
        elif token_id and token_secret:
            openai_key = f"{token_id}.{token_secret}"

    if openai_key and openai_base:
        return {
            "api_key": openai_key,
            "base_url": openai_base,
            "model_name": model_name,
            "provider": "openai_compatible",
        }
    if mistral_key:
        return {
            "api_key": mistral_key,
            "base_url": None,
            "model_name": model_name or "mistral-large-2411",
            "provider": "mistral",
        }
    raise RuntimeError(
        "No LLM credentials found. Set either OPENAI_API_KEY+OPENAI_BASE_URL "
        "or MISTRAL_API_KEY in .env (see .env.example)."
    )


def _init_credentials(require: bool = True):
    """Load credentials into module globals; optional for --mock runs."""
    global LLM_CREDS, API_KEY, BASE_URL, MODEL_NAME
    try:
        LLM_CREDS = _resolve_llm_credentials()
        API_KEY = LLM_CREDS["api_key"]
        BASE_URL = LLM_CREDS["base_url"]
        MODEL_NAME = LLM_CREDS["model_name"]
    except RuntimeError:
        if require:
            raise
        LLM_CREDS = {"provider": "none", "model_name": MODEL_NAME}


# Preregistered market / policy constants
MEMORY_LENGTH = 100
ALPHA = 3.2
FOCAL_NAME = "Firm A"
RIVAL_NAME = "Firm B"
DELTA = 0.05
RETALIATION_DROP = 0.10

RCT_ARMS = {
    "control": RCT_CONTROL,
    "placebo": RCT_PLACEBO,
    "corrected": RCT_CORRECTED,
}

MODE_CONFIG = {
    "smoke": {"n_rounds": 5, "n_runs": 1, "arms": ["control"]},
    "pilot": {"n_rounds": 50, "n_runs": 2, "arms": list(RCT_ARMS.keys())},
    "full": {"n_rounds": 300, "n_runs": 7, "arms": list(RCT_ARMS.keys())},
}


def build_prefix(arm: str) -> str:
    """P1 + forecast elicitation + RCT arm text (control may be empty)."""
    arm_text = RCT_ARMS[arm]
    parts = [P1.strip(), FORECAST_INSTRUCTION.strip()]
    if arm_text.strip():
        parts.append(arm_text.strip())
    return "\n\n".join(parts) + "\n"


async def main(
    arm: str,
    n_rounds: int,
    alpha: float = ALPHA,
    experiment_name: str | None = None,
    use_mock: bool = False,
):
    """
    Run one episode: LLM Firm A vs scripted retaliating Firm B.

    Args:
        arm: RCT arm name (control / placebo / corrected).
        n_rounds: Number of pricing rounds.
        alpha: Demand alpha (default 3.2).
        experiment_name: Storage name; defaults encode arm.
        use_mock: If True, use MockLLMAgent (no API calls).
    """
    if experiment_name is None:
        experiment_name = f"duopoly_belief_correction_{arm}"

    PricingAgentResponse = create_pricing_response_model(
        include_wtp=True,
        wtp_value=4.51 * alpha,
        include_retaliation_forecast=True,
    )

    prefix = build_prefix(arm)
    env_params = {"a": 2.0, "alpha": alpha, "c": 1.0}

    if use_mock:
        # Price path with an intentional cut at round 3 so retaliation fires at 4
        mock_prices = [2.0, 2.0, 1.8, 1.8, 1.9] + [1.9] * max(0, n_rounds - 5)
        mock_forecasts = [0.9, 0.85, 0.7, 0.6, 0.5] + [0.5] * max(0, n_rounds - 5)
        focal = MockLLMAgent(
            FOCAL_NAME,
            prices=mock_prices[:n_rounds],
            forecasts=mock_forecasts[:n_rounds],
            prefix=prefix,
            prompt_template=GENERAL_PROMPT,
            env_params=env_params,
            memory_length=MEMORY_LENGTH,
        )
        provider = "mock"
        model_name = "mock_llm"
    else:
        focal = LLMAgent(
            FOCAL_NAME,
            prefix=prefix,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params=env_params,
            base_url=BASE_URL,
        )
        provider = LLM_CREDS["provider"]
        model_name = MODEL_NAME

    rival = ScriptedRetaliatingAgent(
        RIVAL_NAME,
        focal_agent_name=FOCAL_NAME,
        delta=DELTA,
        retaliation_drop=RETALIATION_DROP,
        env_params=env_params,
    )

    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description="Duopoly belief-correction: LLM vs scripted retaliating rival",
    )

    extra_metadata = {
        "track": "1.1_belief_correction",
        "rct_arm": arm,
        "focal_agent": FOCAL_NAME,
        "rival_agent": RIVAL_NAME,
        "delta": DELTA,
        "retaliation_drop": RETALIATION_DROP,
        "rival_policy": rival.policy_params,
        "primary_outcome_window": [251, 300],
        "alpha": alpha,
        "prompt_type": "P1",
        "model_name": model_name,
        "llm_provider": provider,
        "use_mock": use_mock,
    }

    experiment = Experiment(
        name=experiment_name,
        agents=[focal, rival],
        num_rounds=n_rounds,
        environment=env,
        experiment_dir=current_file_path.parent / "experiments_runs",
        extra_metadata=extra_metadata,
        experiment_plot=True,
    )
    await experiment.run()
    return experiment.storage.experiment_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track 1.1 belief-correction duopoly experiment"
    )
    parser.add_argument(
        "--mode",
        choices=list(MODE_CONFIG.keys()),
        default="smoke",
        help="smoke (5 rounds), pilot (50×3arms×2), or full (300×3arms×7)",
    )
    parser.add_argument(
        "--arm",
        choices=list(RCT_ARMS.keys()),
        default=None,
        help="Optional: run a single RCT arm only (overrides mode arm list)",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=None,
        help="Optional override for number of rounds",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Optional override for number of independent runs",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockLLMAgent (offline pipeline test; no API calls)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = MODE_CONFIG[args.mode]
    n_rounds = args.n_rounds if args.n_rounds is not None else cfg["n_rounds"]
    n_runs = args.n_runs if args.n_runs is not None else cfg["n_runs"]
    arms = [args.arm] if args.arm else cfg["arms"]

    if args.mock:
        _init_credentials(require=False)
        provider = "mock"
        model_label = "mock_llm"
    else:
        _init_credentials(require=True)
        provider = LLM_CREDS["provider"]
        model_label = MODEL_NAME

    print(
        f"Track 1.1 belief correction | mode={args.mode} | "
        f"provider={provider} | model={model_label} | "
        f"rounds={n_rounds} | runs={n_runs} | arms={arms}"
    )

    last_path = None
    for run_idx in range(n_runs):
        for arm in arms:
            print(
                f"Starting run={run_idx + 1}/{n_runs} arm={arm} "
                f"alpha={ALPHA} rounds={n_rounds}"
            )
            last_path = asyncio.run(
                main(
                    arm=arm,
                    n_rounds=n_rounds,
                    alpha=ALPHA,
                    experiment_name=f"duopoly_belief_correction_{arm}",
                    use_mock=args.mock,
                )
            )
            print(f"Completed run={run_idx + 1} arm={arm} path={last_path}\n")

    if last_path is not None:
        from src.analysis.belief_calibration import print_run_report

        print_run_report(last_path)
