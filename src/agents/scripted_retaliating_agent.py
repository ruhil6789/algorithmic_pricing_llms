"""
Scripted retaliating rival with known response policy.

Provides ground-truth rival behavior for Track 1.1 belief-calibration
experiments: play Nash by default; after a focal price cut, retaliate for
exactly one period by lowering price by a fixed drop.
"""

from typing import Any, Dict, Optional

from src.agents.base_agent import Agent


class ScriptedRetaliatingAgent(Agent):
    """
    Reactive scripted rival with a preregistered retaliation rule.

    Policy (ground truth):
        - Default: post the Nash price for this agent's env index.
        - If the focal agent cut price last period by more than ``delta``,
          post ``nash_price - retaliation_drop`` for one period, then return
          to Nash.

    True P(retaliation | cut) = 1; P(retaliation | no cut) = 0.

    Args:
        name: Unique identifier for the agent.
        focal_agent_name: Name of the LLM firm whose cuts trigger retaliation.
        nash_price: Optional fixed Nash price. If None, read from
            ``environment.nash_prices[env_index]`` at act time.
        delta: Absolute price-cut threshold (preregistered).
        retaliation_drop: Absolute price reduction when retaliating.
        **kwargs: Passed to ``Agent`` (must include ``env_params``).
    """

    def __init__(
        self,
        name: str,
        focal_agent_name: str,
        nash_price: Optional[float] = None,
        delta: float = 0.05,
        retaliation_drop: float = 0.10,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.model_name = "scripted_retaliating"
        self.memory_length = -1
        self.focal_agent_name = focal_agent_name
        self._nash_price_override = nash_price
        self.delta = delta
        self.retaliation_drop = retaliation_drop

        self._prev_focal_price: Optional[float] = None
        self._retaliate_next: bool = False
        self._last_own_price: Optional[float] = None

    @property
    def policy_params(self) -> Dict[str, Any]:
        """Serializable policy parameters for experiment metadata."""
        return {
            "policy": "nash_default_one_period_retaliation",
            "focal_agent_name": self.focal_agent_name,
            "delta": self.delta,
            "retaliation_drop": self.retaliation_drop,
            "nash_price_override": self._nash_price_override,
        }

    def _nash_price(self) -> float:
        if self._nash_price_override is not None:
            return float(self._nash_price_override)
        env = getattr(self, "environment", None)
        if env is not None and getattr(env, "nash_prices", None) is not None:
            return float(env.nash_prices[self.env_index])
        raise RuntimeError(
            f"ScriptedRetaliatingAgent {self.name}: Nash price unavailable. "
            "Pass nash_price=... or attach an environment with nash_prices."
        )

    async def act(self, prompt: str) -> dict:
        """Choose Nash or one-period retaliatory price from prior observations."""
        nash = self._nash_price()
        if self._retaliate_next:
            chosen_price = max(nash - self.retaliation_drop, 0.01)
            self._retaliate_next = False
        else:
            chosen_price = nash

        self._last_own_price = float(chosen_price)
        return {
            "agent_name": self.name,
            "content": {"chosen_price": float(chosen_price)},
        }

    def observe_market(self, prices: Dict[str, float], round_num: int) -> None:
        """
        Update retaliation state after market clearing.

        Flags retaliation for the *next* act if the focal firm cut this round
        by more than ``delta`` relative to its previous posted price.
        """
        if self.focal_agent_name not in prices:
            return

        focal_price = float(prices[self.focal_agent_name])
        if (
            self._prev_focal_price is not None
            and focal_price < self._prev_focal_price - self.delta
        ):
            self._retaliate_next = True
        self._prev_focal_price = focal_price

    @property
    def requires_prompt(self) -> bool:
        return False

    @property
    def type(self) -> str:
        return "scripted_retaliating"
