"""superagent-playground: a tiny playground for classic agent paradigms."""

from __future__ import annotations

from .agent import FALLBACK_ANSWER, ReActAgent, ReActResult, Step
from .llm import LLM, LLMConfig, LLMError, OpenAILLM
from .parser import Action, ParsedOutput, ToolInvocation, parse_action, parse_output
from .tools import (
    Tool,
    ToolRegistry,
    build_default_registry,
    calculate,
    serpapi_search,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "FALLBACK_ANSWER",
    "LLM",
    "LLMConfig",
    "LLMError",
    "OpenAILLM",
    "ParsedOutput",
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
    "serpapi_search",
]
