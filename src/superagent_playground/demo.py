"""A deterministic, offline ReAct demo that needs no API key."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .agent import ReActAgent, ReActResult
from .tools import build_default_registry

DEMO_QUESTION = "请计算 (12 + 8) * 3，并告诉我结果是否大于 50。"

DEMO_RESPONSES = [
    (
        "Thought: 这是四则运算，我可以用 Calculator 工具保证计算准确。\n"
        "Action: Calculator[(12 + 8) * 3]"
    ),
    (
        "Thought: 计算结果是 60，60 大于 50，因此可以给出最终答案。\n"
        "Action: Finish[(12 + 8) * 3 = 60，结果大于 50。]"
    ),
]


class ScriptedLLM:
    """An ``LLM`` implementation that replays a fixed list of replies."""

    def __init__(self, responses: Sequence[str]) -> None:
        if not responses:
            raise ValueError("responses 不能为空。")
        self._responses = list(responses)
        self._index = 0
        self.calls: list[list[Mapping[str, str]]] = []

    def think(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
    ) -> str:
        self.calls.append(list(messages))
        if self._index >= len(self._responses):
            return self._responses[-1]
        response = self._responses[self._index]
        self._index += 1
        return response


def run_offline_demo(
    *,
    verbose: bool = True,
    logger: Callable[[str], None] = print,
) -> ReActResult:
    """Run the scripted calculator example through the real ReAct loop."""

    llm = ScriptedLLM(DEMO_RESPONSES)
    agent = ReActAgent(
        llm,
        build_default_registry(),
        max_steps=3,
        verbose=verbose,
        logger=logger,
    )
    return agent.run(DEMO_QUESTION)
