"""Budget policy and errors for deterministic context assembly."""

from __future__ import annotations

from dataclasses import dataclass


class ContextBudgetExceeded(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    model_context_limit: int = 200_000
    reserved_response_tokens: int = 8_000
    safety_margin_tokens: int = 2_000
    recent_turn_fraction: float = 0.35
    memory_fraction: float = 0.35
    tool_context_fraction: float = 0.20
    recent_message_limit: int = 8

    def __post_init__(self) -> None:
        if min(self.model_context_limit, self.reserved_response_tokens, self.safety_margin_tokens, self.recent_message_limit) < 0:
            raise ValueError("token limits cannot be negative")
        for name in ("recent_turn_fraction", "memory_fraction", "tool_context_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.available_input_tokens <= 0:
            raise ValueError("reserved response and safety margin leave no input budget")

    @property
    def available_input_tokens(self) -> int:
        return self.model_context_limit - self.reserved_response_tokens - self.safety_margin_tokens

