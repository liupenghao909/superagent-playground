"""The ReAct agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .llm import LLM
from .parser import ParsedOutput, parse_output
from .prompt import REACT_PROMPT_TEMPLATE, build_react_prompt
from .tools import ToolRegistry


@dataclass(frozen=True)
class Step:
    """One ``Thought -> Action -> Observation`` turn."""

    index: int
    thought: str | None
    action: str | None
    observation: str | None


@dataclass
class ReActResult:
    """The outcome of one ``ReActAgent.run`` call."""

    question: str
    answer: str | None
    steps: list[Step] = field(default_factory=list)
    reason: str = "finish"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.answer is not None

    @property
    def final_answer(self) -> str | None:
        return self.answer


class ReActAgent:
    """A minimal ReAct agent: reason, act, observe, repeat.

    Args:
        llm: Any object implementing the :class:`~superagent_playground.llm.LLM`
            protocol.
        tools: Registry of tools the model may call.
        max_steps: Safety valve against loops; the loop stops after this many
            model calls.
        prompt_template: ReAct prompt, overridable for experiments.
        verbose: Print a readable trace of every step.
        logger: Callable used when ``verbose`` is true (defaults to ``print``).
    """

    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        *,
        max_steps: int = 5,
        prompt_template: str = REACT_PROMPT_TEMPLATE,
        verbose: bool = False,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0。")
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.prompt_template = prompt_template
        self.verbose = verbose
        self._logger = logger or print

    def run(self, question: str) -> ReActResult:
        """Run the ReAct loop until ``Finish`` or ``max_steps``."""

        question = (question or "").strip()
        if not question:
            raise ValueError("question 不能为空。")

        history: list[str] = []
        steps: list[Step] = []
        tools_description = self.tools.describe()

        for index in range(1, self.max_steps + 1):
            self._log(f"--- 第 {index}/{self.max_steps} 步 ---")
            prompt = build_react_prompt(
                tools=tools_description,
                question=question,
                history="\n".join(history),
                template=self.prompt_template,
            )
            try:
                response = self.llm.think([{"role": "user", "content": prompt}])
            except Exception as exc:
                message = f"LLM 调用失败: {exc}"
                self._log(message)
                return ReActResult(
                    question=question,
                    answer=None,
                    steps=steps,
                    reason="llm_error",
                    error=message,
                )

            if not response or not response.strip():
                message = "LLM 返回了空响应。"
                self._log(message)
                return ReActResult(
                    question=question,
                    answer=None,
                    steps=steps,
                    reason="llm_error",
                    error=message,
                )

            parsed = parse_output(response)
            self._log_parsed(parsed)

            if parsed.action is None:
                observation = (
                    "无法解析 Action。请严格按 `Thought: ...` 和 `Action: ...` 的格式回复，"
                    "Action 使用 `Tool[input]` 或 `Finish[answer]`。"
                )
                steps.append(
                    Step(
                        index=index,
                        thought=parsed.thought,
                        action=parsed.action_text,
                        observation=observation,
                    )
                )
                self._log(f"Observation: {observation}")
                history.append(f"Action: {parsed.action_text or '(缺失)'}")
                history.append(f"Observation: {observation}")
                continue

            if parsed.action.kind == "finish":
                answer = parsed.finish_answer or ""
                steps.append(
                    Step(
                        index=index,
                        thought=parsed.thought,
                        action=parsed.action_text,
                        observation=None,
                    )
                )
                return ReActResult(
                    question=question,
                    answer=answer,
                    steps=steps,
                    reason="finish",
                )

            if parsed.action.kind == "tool":
                invocation = parsed.action.invocation
                assert invocation is not None
                observation = self._execute(invocation.tool_name, invocation.tool_input)
            else:
                observation = (
                    f"Action 格式无效: {parsed.action.raw!r}。"
                    "请使用 `Tool[input]` 或 `Finish[answer]`。"
                )

            steps.append(
                Step(
                    index=index,
                    thought=parsed.thought,
                    action=parsed.action_text,
                    observation=observation,
                )
            )
            self._log(f"Observation: {observation}")
            history.append(f"Action: {parsed.action_text}")
            history.append(f"Observation: {observation}")

        self._log(f"已达到最大步数 {self.max_steps}，流程终止。")
        return ReActResult(
            question=question,
            answer=None,
            steps=steps,
            reason="max_steps",
        )

    def _execute(self, tool_name: str, tool_input: str) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            available = ", ".join(self.tools.names()) or "(无)"
            return f"错误: 未找到名为 '{tool_name}' 的工具。可用工具: {available}。"
        try:
            observation = tool.func(tool_input)
        except Exception as exc:
            return f"错误: 工具 '{tool_name}' 执行失败: {exc}"
        if observation is None:
            return ""
        return str(observation)

    def _log_parsed(self, parsed: ParsedOutput) -> None:
        if parsed.thought:
            self._log(f"Thought: {parsed.thought}")
        if parsed.action_text:
            self._log(f"Action: {parsed.action_text}")

    def _log(self, message: str) -> None:
        if self.verbose:
            self._logger(message)
