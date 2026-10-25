"""
Deterministic LLM stand-in for offline Track 1.1 pipeline tests.

Returns structured pricing + retaliation forecasts without calling an API.
Useful when credentials or upstream models are unavailable.
"""

from typing import Any, Dict, Optional

from src.agents.base_agent import Agent


class MockLLMAgent(Agent):
    """
    Offline focal agent that posts a scripted price path and forecasts.

    Args:
        name: Agent name.
        prices: Sequence of chosen prices (cycled if shorter than n rounds).
        forecasts: Optional sequence of p_retaliation values in [0, 1].
        default_forecast: Forecast used when forecasts is None.
        **kwargs: Passed to Agent (requires env_params).
    """

    def __init__(
        self,
        name: str,
        prices: list,
        forecasts: Optional[list] = None,
        default_forecast: float = 0.8,
        model_name: str = "mock_llm",
        memory_length: int = 100,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.model_name = model_name
        self.memory_length = memory_length
        self.prices = list(prices)
        self.forecasts = list(forecasts) if forecasts is not None else None
        self.default_forecast = float(default_forecast)
        self._index = 0

    async def act(self, prompt: str) -> Dict[str, Any]:
        price = float(self.prices[self._index % len(self.prices)])
        if self.forecasts is not None:
            forecast = float(self.forecasts[self._index % len(self.forecasts)])
        else:
            forecast = self.default_forecast
        self._index += 1
        return {
            "agent_name": self.name,
            "content": {
                "observations": "mock observations",
                "plans": "mock plans",
                "insights": "mock insights",
                "chosen_price": price,
                "p_retaliation": forecast,
            },
        }

    @property
    def requires_prompt(self) -> bool:
        """Offline mock does not need prompt construction."""
        return False

    @property
    def type(self) -> str:
        return "mock_llm"
