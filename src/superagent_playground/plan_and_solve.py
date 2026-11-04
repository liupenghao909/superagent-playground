"""The Plan-and-Solve agent: plan first, then solve step by step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .agent import FALLBACK_ANSWER
from .llm import LLM
from .parser import parse_plan
from .prompt import (
    EXECUTOR_PROMPT_TEMPLATE,
    PLANNER_PROMPT_TEMPLATE,
    build_executor_prompt,
    build_planner_prompt,
)


@dataclass(frozen=True)
class PlanStep:
    """One planned subtask and, once executed, its result."""

    index: int
    description: str
    result: str | None = None


@dataclass
class PlanAndSolveResult:
    """The outcome of one ``PlanAndSolveAgent.run`` call.

    ``reason`` is ``"finish"``, ``"plan_error"``, ``"max_steps"`` or
    ``"llm_fallback"``. Only ``finish`` carries a usable answer: the result of
    the last plan step. ``plan_error`` means the planner never produced a
    parseable step list; ``max_steps`` means the plan was longer than the
    configured execution budget; ``llm_fallback`` means a model call failed.
    """

    question: str
    answer: str | None
    plan: list[str] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)
    reason: str = "finish"
    error: str | None = None
    raw_plan: str | None = None

    @property
    def ok(self) -> bool:
        return self.answer is not None and self.reason == "finish"

    @property
    def final_answer(self) -> str | None:
        return self.answer


@dataclass
class _PlanningOutcome:
    """Internal result of the planning phase."""

    plan: list[str]
    raw: str | None
    error: str | None
    llm_failure: bool = False


class PlanAndSolveAgent:
    """A minimal Plan-and-Solve agent.

    Planning decomposes the question into an ordered list of subtasks. Solving
    then executes those subtasks one by one, passing the completed steps and
    their results as context for the next step. The result of the last step is
    the final answer, so the classic paradigm needs no separate synthesis call.

    Args:
        llm: Any object implementing the :class:`~superagent_playground.llm.LLM`
            protocol.
        max_steps: Maximum number of plan steps to execute. A longer plan is
            rejected as ``max_steps`` instead of being silently truncated.
        plan_retries: Extra planner calls after an unparseable plan.
        planner_prompt_template: Planner prompt, overridable for experiments.
        executor_prompt_template: Executor prompt, overridable for experiments.
        verbose: Print a readable trace of the plan and every step.
        logger: Callable used when ``verbose`` is true (defaults to ``print``).
    """

    def __init__(
        self,
        llm: LLM,
        *,
        max_steps: int = 8,
        plan_retries: int = 1,
        planner_prompt_template: str = PLANNER_PROMPT_TEMPLATE,
        executor_prompt_template: str = EXECUTOR_PROMPT_TEMPLATE,
        verbose: bool = False,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0。")
        if plan_retries < 0:
            raise ValueError("plan_retries 不能为负数。")
        self.llm = llm
        self.max_steps = max_steps
        self.plan_retries = plan_retries
        self.planner_prompt_template = planner_prompt_template
        self.executor_prompt_template = executor_prompt_template
        self.verbose = verbose
        self._logger = logger or print

    def run(self, question: str) -> PlanAndSolveResult:
        """Plan the question, execute every step, and return the final answer."""

        question = (question or "").strip()
        if not question:
            raise ValueError("question 不能为空。")

        planning = self._make_plan(question)
        if not planning.plan:
            if planning.llm_failure:
                return self._fallback_result(
                    question=question,
                    plan=[],
                    steps=[],
                    message=planning.error or "LLM 调用失败。",
                    raw_plan=planning.raw,
                )
            return PlanAndSolveResult(
                question=question,
                answer=None,
                reason="plan_error",
                error=planning.error,
                raw_plan=planning.raw,
            )

        plan = planning.plan
        if len(plan) > self.max_steps:
            message = (
                f"计划包含 {len(plan)} 个步骤，超过 max_steps={self.max_steps}。"
            )
            self._log(message)
            return PlanAndSolveResult(
                question=question,
                answer=None,
                plan=plan,
                reason="max_steps",
                error=message,
                raw_plan=planning.raw,
            )

        self._log_plan(plan)
        steps: list[PlanStep] = []
        history: list[str] = []
        for index, description in enumerate(plan, start=1):
            self._log(f"--- 执行第 {index}/{len(plan)} 步 ---")
            prompt = build_executor_prompt(
                question=question,
                plan=plan,
                history="\n\n".join(history),
                current_step=description,
                template=self.executor_prompt_template,
            )
            try:
                response = self.llm.think([{"role": "user", "content": prompt}])
            except Exception as exc:
                message = f"LLM 调用失败: {exc}"
                self._log(message)
                return self._fallback_result(
                    question=question,
                    plan=plan,
                    steps=steps,
                    message=message,
                    raw_plan=planning.raw,
                )

            if not response or not response.strip():
                message = "LLM 返回了空响应。"
                self._log(message)
                return self._fallback_result(
                    question=question,
                    plan=plan,
                    steps=steps,
                    message=message,
                    raw_plan=planning.raw,
                )

            result = response.strip()
            steps.append(
                PlanStep(index=index, description=description, result=result)
            )
            self._log(f"当前步骤: {description}")
            self._log(f"结果: {result}")
            history.append(f"步骤 {index}: {description}\n结果: {result}")

        return PlanAndSolveResult(
            question=question,
            answer=steps[-1].result,
            plan=plan,
            steps=steps,
            reason="finish",
            raw_plan=planning.raw,
        )

    def _make_plan(self, question: str) -> _PlanningOutcome:
        """Ask the planner for steps, retrying once after a format failure."""

        self._log("--- 规划阶段 ---")
        attempts = self.plan_retries + 1
        feedback = ""
        raw_response: str | None = None
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                self._log(f"--- 重新规划（第 {attempt}/{attempts} 次） ---")
            prompt = build_planner_prompt(
                question=question,
                feedback=feedback,
                template=self.planner_prompt_template,
            )
            try:
                raw_response = self.llm.think(
                    [{"role": "user", "content": prompt}]
                )
            except Exception as exc:
                message = f"LLM 调用失败: {exc}"
                self._log(message)
                return _PlanningOutcome(
                    plan=[], raw=raw_response, error=message, llm_failure=True
                )

            if not raw_response or not raw_response.strip():
                error = "LLM 返回了空响应。"
                self._log(error)
                feedback = error
                continue

            plan = parse_plan(raw_response)
            if plan:
                return _PlanningOutcome(plan=plan, raw=raw_response, error=None)

            error = "无法从模型输出中解析出步骤列表。"
            self._log(error)
            feedback = error

        return _PlanningOutcome(
            plan=[],
            raw=raw_response,
            error=f"规划失败（已尝试 {attempts} 次）: {feedback}",
        )

    def _fallback_result(
        self,
        *,
        question: str,
        plan: list[str],
        steps: list[PlanStep],
        message: str,
        raw_plan: str | None,
    ) -> PlanAndSolveResult:
        """Return a user-facing answer when the model stays unreachable."""

        return PlanAndSolveResult(
            question=question,
            answer=FALLBACK_ANSWER,
            plan=plan,
            steps=steps,
            reason="llm_fallback",
            error=message,
            raw_plan=raw_plan,
        )

    def _log_plan(self, plan: list[str]) -> None:
        self._log(f"--- 规划完成，共 {len(plan)} 步 ---")
        for index, step in enumerate(plan, start=1):
            self._log(f"{index}. {step}")

    def _log(self, message: str) -> None:
        if self.verbose:
            self._logger(message)
