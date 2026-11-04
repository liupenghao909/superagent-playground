import pytest

from superagent_playground.agent import FALLBACK_ANSWER
from superagent_playground.plan_and_solve import PlanAndSolveAgent
from superagent_playground.prompt import (
    build_executor_prompt,
    build_planner_prompt,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def think(self, messages, *, temperature=None):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeLLM 被调用了太多次")
        return self.responses.pop(0)


class ExplodingLLM:
    def think(self, messages, *, temperature=None):
        raise RuntimeError("network down")


def test_agent_plans_then_executes_each_step():
    llm = FakeLLM(
        [
            '```python\n["计算周二销量", "计算周三销量", "计算总销量"]\n```',
            "30",
            "25",
            "70",
        ]
    )
    result = PlanAndSolveAgent(llm, max_steps=5).run("水果店问题")

    assert result.ok
    assert result.reason == "finish"
    assert result.answer == "70"
    assert result.final_answer == "70"
    assert result.plan == ["计算周二销量", "计算周三销量", "计算总销量"]
    assert [step.index for step in result.steps] == [1, 2, 3]
    assert [step.result for step in result.steps] == ["30", "25", "70"]
    assert len(llm.calls) == 4


def test_executor_prompt_contains_problem_plan_history_and_current_step():
    llm = FakeLLM(['["计算周二销量", "计算周三销量"]', "30", "25"])
    PlanAndSolveAgent(llm).run("水果店问题")

    planner_prompt = llm.calls[0][0]["content"]
    assert "Question: 水果店问题" in planner_prompt
    assert "Python 字符串列表" in planner_prompt

    executor_prompt = llm.calls[2][0]["content"]
    assert "水果店问题" in executor_prompt
    assert "1. 计算周二销量" in executor_prompt
    assert "2. 计算周三销量" in executor_prompt
    assert "步骤 1: 计算周二销量" in executor_prompt
    assert "结果: 30" in executor_prompt
    assert "当前步骤\n计算周三销量" in executor_prompt


def test_agent_retries_planner_after_unparseable_plan():
    llm = FakeLLM(
        [
            "我不会做计划",
            '["唯一一步"]',
            "答案",
        ]
    )
    result = PlanAndSolveAgent(llm, plan_retries=1).run("问题")

    assert result.ok
    assert result.answer == "答案"
    assert len(llm.calls) == 3
    retry_prompt = llm.calls[1][0]["content"]
    assert "上一次输出无法解析为步骤列表" in retry_prompt
    assert "请严格按照上面的格式重新输出计划" in retry_prompt


def test_agent_returns_plan_error_after_retries_are_exhausted():
    llm = FakeLLM(["不是计划", "仍然不是", "也不"])
    result = PlanAndSolveAgent(llm, plan_retries=2).run("问题")

    assert not result.ok
    assert result.reason == "plan_error"
    assert result.answer is None
    assert result.plan == []
    assert result.steps == []
    assert result.raw_plan == "也不"
    assert "规划失败（已尝试 3 次）" in result.error
    assert len(llm.calls) == 3


def test_agent_retries_planner_after_empty_response():
    llm = FakeLLM(["   ", '["第一步"]', "答案"])
    result = PlanAndSolveAgent(llm, plan_retries=1).run("问题")

    assert result.ok
    assert result.answer == "答案"
    assert "LLM 返回了空响应" in llm.calls[1][0]["content"]


def test_agent_returns_fallback_when_planner_llm_fails():
    result = PlanAndSolveAgent(ExplodingLLM()).run("问题")

    assert not result.ok
    assert result.reason == "llm_fallback"
    assert result.answer == FALLBACK_ANSWER
    assert "network down" in result.error
    assert result.plan == []
    assert result.steps == []


def test_agent_returns_fallback_when_executor_llm_fails():
    class FailingAfterPlanLLM:
        def __init__(self):
            self.calls = 0

        def think(self, messages, *, temperature=None):
            self.calls += 1
            if self.calls == 1:
                return '["第一步", "第二步"]'
            if self.calls == 2:
                return "结果一"
            raise RuntimeError("network down")

    result = PlanAndSolveAgent(FailingAfterPlanLLM()).run("问题")

    assert not result.ok
    assert result.reason == "llm_fallback"
    assert result.answer == FALLBACK_ANSWER
    assert "network down" in result.error
    assert result.plan == ["第一步", "第二步"]
    assert len(result.steps) == 1
    assert result.steps[0].result == "结果一"


def test_agent_returns_fallback_when_executor_returns_empty_text():
    result = PlanAndSolveAgent(FakeLLM(['["第一步"]', "   "])).run("问题")

    assert not result.ok
    assert result.reason == "llm_fallback"
    assert result.answer == FALLBACK_ANSWER
    assert "空响应" in result.error
    assert result.steps == []


def test_agent_rejects_plan_longer_than_max_steps():
    llm = FakeLLM(['["第一步", "第二步", "第三步"]'])
    result = PlanAndSolveAgent(llm, max_steps=2).run("问题")

    assert not result.ok
    assert result.reason == "max_steps"
    assert result.answer is None
    assert result.plan == ["第一步", "第二步", "第三步"]
    assert result.steps == []
    assert "超过 max_steps=2" in result.error
    assert len(llm.calls) == 1


def test_agent_rejects_empty_question():
    with pytest.raises(ValueError):
        PlanAndSolveAgent(FakeLLM(['["第一步"]'])).run("  ")


def test_agent_rejects_invalid_limits():
    llm = FakeLLM(['["第一步"]'])

    with pytest.raises(ValueError):
        PlanAndSolveAgent(llm, max_steps=0)
    with pytest.raises(ValueError):
        PlanAndSolveAgent(llm, plan_retries=-1)


def test_agent_prints_a_readable_trace():
    lines = []
    llm = FakeLLM(['["第一步", "第二步"]', "中间结果", "最终结果"])
    agent = PlanAndSolveAgent(
        llm,
        verbose=True,
        logger=lines.append,
    )

    agent.run("问题")

    assert "--- 规划阶段 ---" in lines
    assert "--- 规划完成，共 2 步 ---" in lines
    assert "1. 第一步" in lines
    assert "--- 执行第 1/2 步 ---" in lines
    assert "当前步骤: 第一步" in lines
    assert "结果: 中间结果" in lines
    assert "--- 执行第 2/2 步 ---" in lines
    assert "结果: 最终结果" in lines


def test_agent_resets_plan_between_runs():
    llm = FakeLLM(
        [
            '["第一次计划"]',
            "一",
            '["第二次计划"]',
            "二",
        ]
    )
    agent = PlanAndSolveAgent(llm)

    first = agent.run("第一个问题")
    second = agent.run("第二个问题")

    assert first.answer == "一"
    assert first.plan == ["第一次计划"]
    assert second.answer == "二"
    assert second.plan == ["第二次计划"]
    assert "第一个问题" in llm.calls[0][0]["content"]
    assert "第二个问题" in llm.calls[2][0]["content"]


def test_planner_prompt_includes_question_and_list_format():
    prompt = build_planner_prompt(question="问题")

    assert "Question: 问题" in prompt
    assert "Python 字符串列表" in prompt
    assert "```python" in prompt


def test_planner_prompt_includes_retry_feedback():
    prompt = build_planner_prompt(question="问题", feedback="缺少列表")

    assert "缺少列表" in prompt
    assert "请严格按照上面的格式重新输出计划" in prompt


def test_executor_prompt_formats_plan_and_empty_history():
    prompt = build_executor_prompt(
        question="问题",
        plan=["第一步", "第二步"],
        current_step="第一步",
    )

    assert "1. 第一步" in prompt
    assert "2. 第二步" in prompt
    assert "已完成的步骤和结果\n(暂无)" in prompt
    assert "当前步骤\n第一步" in prompt
