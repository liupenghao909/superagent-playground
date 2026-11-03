from types import SimpleNamespace

import pytest

from superagent_playground.agent import FALLBACK_ANSWER, ReActAgent
from superagent_playground.llm import OpenAILLM
from superagent_playground.prompt import build_react_prompt
from superagent_playground.tools import ToolRegistry


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


def make_registry():
    registry = ToolRegistry()
    registry.register("Search", "搜索工具。", lambda query: f"搜索结果: {query}")
    registry.register("Calculator", "计算工具。", lambda expression: "60")
    return registry


def test_agent_runs_tool_then_finishes():
    llm = FakeLLM(
        [
            "Thought: 先计算。\nAction: Calculator[(12 + 8) * 3]",
            "Thought: 结果是 60，大于 50。\nAction: Finish[60，大于 50。]",
        ]
    )
    result = ReActAgent(llm, make_registry(), max_steps=3).run("计算并比较")

    assert result.ok
    assert result.answer == "60，大于 50。"
    assert result.reason == "finish"
    assert len(result.steps) == 2
    assert result.steps[0].observation == "60"
    assert result.steps[1].observation is None

    second_prompt = llm.calls[1][0]["content"]
    assert "Question: 计算并比较" in second_prompt
    assert "Action: Calculator[(12 + 8) * 3]" in second_prompt
    assert "Observation: 60" in second_prompt


def test_agent_reports_unknown_tool_and_recovers():
    llm = FakeLLM(
        [
            "Thought: 查一下天气。\nAction: Weather[北京]",
            "Thought: 没有天气工具。\nAction: Finish[无法查询天气]",
        ]
    )
    result = ReActAgent(llm, make_registry(), max_steps=3).run("北京天气")

    assert result.ok
    assert "未找到名为 'Weather' 的工具" in result.steps[0].observation
    assert "Search" in result.steps[0].observation


def test_agent_recovers_from_unparseable_response():
    llm = FakeLLM(
        [
            "Thought: 我需要搜索一下。",
            "Thought: 现在搜索。\nAction: Search[测试]",
            "Thought: 有结果了。\nAction: Finish[完成]",
        ]
    )
    result = ReActAgent(llm, make_registry(), max_steps=4).run("问题")

    assert result.ok
    assert "无法解析 Action" in result.steps[0].observation
    assert result.answer == "完成"


def test_agent_recovers_from_invalid_action_format():
    llm = FakeLLM(
        [
            "Thought: 搜索。\nAction: Search 测试",
            "Thought: 重试。\nAction: Finish[完成]",
        ]
    )
    result = ReActAgent(llm, make_registry(), max_steps=3).run("问题")

    assert result.ok
    assert "Action 格式无效" in result.steps[0].observation


def test_agent_stops_at_max_steps():
    llm = FakeLLM(
        [
            "Thought: 继续搜索。\nAction: Search[a]",
            "Thought: 继续搜索。\nAction: Search[b]",
        ]
    )
    result = ReActAgent(llm, make_registry(), max_steps=2).run("问题")

    assert not result.ok
    assert result.reason == "max_steps"
    assert len(result.steps) == 2


def test_agent_returns_fallback_answer_when_llm_fails():
    result = ReActAgent(ExplodingLLM(), make_registry()).run("问题")

    assert not result.ok
    assert result.answer == FALLBACK_ANSWER
    assert result.reason == "llm_fallback"
    assert "network down" in result.error
    assert result.steps == []


def test_agent_returns_fallback_when_llm_returns_empty_text():
    result = ReActAgent(FakeLLM(["   "]), make_registry()).run("问题")

    assert not result.ok
    assert result.answer == FALLBACK_ANSWER
    assert result.reason == "llm_fallback"
    assert "空响应" in result.error


class FailingAfterOneStepLLM:
    def __init__(self):
        self.calls = 0

    def think(self, messages, *, temperature=None):
        self.calls += 1
        if self.calls == 1:
            return "Thought: 先搜索。\nAction: Search[x]"
        raise RuntimeError("network down")


def test_agent_fallback_keeps_steps_completed_before_failure():
    result = ReActAgent(FailingAfterOneStepLLM(), make_registry()).run("问题")

    assert result.reason == "llm_fallback"
    assert len(result.steps) == 1
    assert result.steps[0].observation == "搜索结果: x"


def test_agent_retries_model_call_before_giving_up():
    class FlakyCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("timeout")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Thought: 直接回答。\nAction: Finish[完成]"
                        )
                    )
                ]
            )

    completions = FlakyCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = OpenAILLM(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
        client=client,
    )
    result = ReActAgent(llm, make_registry()).run("问题")

    assert result.ok
    assert result.answer == "完成"
    assert completions.calls == 2


def test_agent_reports_tool_exception_as_observation():
    registry = make_registry()

    def boom(_):
        raise RuntimeError("tool broke")

    registry.register("Boom", "会失败的工具。", boom)
    llm = FakeLLM(
        [
            "Thought: 调用。\nAction: Boom[x]",
            "Thought: 跳过。\nAction: Finish[完成]",
        ]
    )
    result = ReActAgent(llm, registry, max_steps=3).run("问题")

    assert result.ok
    assert "工具 'Boom' 执行失败" in result.steps[0].observation


def test_agent_rejects_empty_question():
    with pytest.raises(ValueError):
        ReActAgent(FakeLLM(["Action: Finish[ok]"]), make_registry()).run("  ")


def test_agent_rejects_invalid_max_steps():
    with pytest.raises(ValueError):
        ReActAgent(FakeLLM(["Action: Finish[ok]"]), make_registry(), max_steps=0)


def test_agent_prints_a_readable_trace():
    lines = []
    llm = FakeLLM(["Thought: 直接回答。\nAction: Finish[42]"])
    agent = ReActAgent(
        llm,
        make_registry(),
        verbose=True,
        logger=lines.append,
    )

    agent.run("问题")

    assert "--- 第 1/5 步 ---" in lines
    assert "Thought: 直接回答。" in lines
    assert "Action: Finish[42]" in lines


def test_agent_resets_history_between_runs():
    llm = FakeLLM(
        [
            "Thought: 第一次。\nAction: Finish[一]",
            "Thought: 第二次。\nAction: Finish[二]",
        ]
    )
    agent = ReActAgent(llm, make_registry())

    assert agent.run("第一个问题").answer == "一"
    assert agent.run("第二个问题").answer == "二"
    assert "History: (暂无)" in llm.calls[1][0]["content"]


def test_prompt_includes_tools_question_and_history():
    prompt = build_react_prompt(
        tools="- Echo: 返回输入。",
        question="问题",
        history="Action: Echo[x]\nObservation: x",
    )

    assert "- Echo: 返回输入。" in prompt
    assert "Question: 问题" in prompt
    assert "Action: Echo[x]" in prompt
    assert "Finish[" in prompt
