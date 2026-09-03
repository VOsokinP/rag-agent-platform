"""What one agent run accumulates."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Step:
    """One tool invocation, recorded for the response trace."""

    tool: str
    input: str
    result: str


@dataclass
class Usage:
    """Token and cost accounting for a run.

    Carried because unbounded spend is the first practical objection to any
    agent loop; `cost_usd` is an estimate from configured rates, not a bill.
    """

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    citations: list = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    budget_exhausted: bool = False
