"""superagent-playground: a tiny playground for classic agent paradigms."""

from __future__ import annotations

from .agent import FALLBACK_ANSWER, ReActAgent, ReActResult, Step
from .llm import LLM, LLMConfig, LLMError, OpenAILLM
from .parser import (
    Action,
    ParsedOutput,
    ToolInvocation,
    parse_action,
    parse_output,
    parse_plan,
)
from .plan_and_solve import PlanAndSolveAgent, PlanAndSolveResult, PlanStep
from .tools import (
    Tool,
    ToolRegistry,
    build_default_registry,
    calculate,
    serpapi_search,
)

__version__ = "0.2.0"

__all__ = [
    "Action",
    "FALLBACK_ANSWER",
    "LLM",
    "LLMConfig",
    "LLMError",
    "OpenAILLM",
    "ParsedOutput",
    "PlanAndSolveAgent",
    "PlanAndSolveResult",
    "PlanStep",
    "ReActAgent",
    "ReActResult",
    "Step",
    "Tool",
    "ToolInvocation",
    "ToolRegistry",
    "build_default_registry",
    "calculate",
    "parse_action",
    "parse_output",
    "parse_plan",
    "serpapi_search",
]
