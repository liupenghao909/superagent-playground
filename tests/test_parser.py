import pytest

from superagent_playground.parser import (
    ToolInvocation,
    parse_action,
    parse_output,
)


def test_parses_thought_and_tool_action():
    parsed = parse_output(
        "Thought: 需要搜索最新的信息。\nAction: Search[华为最新手机]"
    )

    assert parsed.thought == "需要搜索最新的信息。"
    assert parsed.action_text == "Search[华为最新手机]"
    assert parsed.invocation == ToolInvocation("Search", "华为最新手机")
    assert parsed.finish_answer is None
    assert not parsed.is_finish


def test_parses_multiline_thought_and_finish_answer():
    parsed = parse_output(
        "Thought: 第一行\n第二行。\nAction: Finish[答案第一行\n答案第二行]"
    )

    assert parsed.thought == "第一行\n第二行。"
    assert parsed.finish_answer == "答案第一行\n答案第二行"
    assert parsed.is_finish


def test_parses_markdown_bold_labels():
    parsed = parse_output(
        "**Thought:** 先计算。\n**Action:** Calculator[1 + 2]"
    )

    assert parsed.thought == "先计算。"
    assert parsed.invocation == ToolInvocation("Calculator", "1 + 2")


def test_supports_fullwidth_colons():
    parsed = parse_output("Thought：先思考。\nAction：Finish[42]")

    assert parsed.thought == "先思考。"
    assert parsed.finish_answer == "42"


def test_action_only_response_is_supported():
    parsed = parse_output("Action: Finish[直接答案]")

    assert parsed.thought is None
    assert parsed.finish_answer == "直接答案"


def test_missing_action_returns_no_action():
    parsed = parse_output("Thought: 我还不知道答案。")

    assert parsed.thought == "我还不知道答案。"
    assert parsed.action_text is None
    assert parsed.action is None


def test_invalid_action_is_marked_invalid():
    parsed = parse_output("Thought: 试试看。\nAction: Search 华为")

    assert parsed.action is not None
    assert parsed.action.kind == "invalid"
    assert parsed.invocation is None


def test_parse_action_handles_whitespace_and_nested_brackets():
    action = parse_action(" Search [ 查找 [嵌套] 内容 ] ")

    assert action.kind == "tool"
    assert action.invocation == ToolInvocation("Search", "查找 [嵌套] 内容")


def test_parse_action_rejects_empty_finish():
    action = parse_action("Finish[]")

    assert action.kind == "invalid"


def test_code_fenced_response_is_unwrapped():
    parsed = parse_output(
        "```\nThought: 思考\nAction: Finish[答案]\n```"
    )

    assert parsed.thought == "思考"
    assert parsed.finish_answer == "答案"


def test_empty_response_yields_nothing():
    parsed = parse_output("   ")

    assert parsed.thought is None
    assert parsed.action_text is None
    assert parsed.action is None


def test_parse_action_supports_unicode_tool_names():
    action = parse_action("搜索[天气]")

    assert action.kind == "tool"
    assert action.invocation == ToolInvocation("搜索", "天气")


def test_parse_action_rejects_empty_text():
    assert parse_action("").kind == "invalid"


def test_fabricated_observation_does_not_pollute_tool_input():
    parsed = parse_output(
        "Thought: 搜索。\nAction: Search[华为手机]\nObservation: 搜索结果 [1]"
    )

    assert parsed.action_text == "Search[华为手机]"
    assert parsed.invocation == ToolInvocation("Search", "华为手机")


def test_second_action_is_ignored():
    parsed = parse_output(
        "Thought: 两次。\nAction: Calculator[1 + 1]\nAction: Finish[2]"
    )

    assert parsed.action_text == "Calculator[1 + 1]"
    assert parsed.invocation == ToolInvocation("Calculator", "1 + 1")


def test_same_line_fabricated_observation_does_not_pollute_tool_input():
    parsed = parse_output(
        "Thought: 搜索。\nAction: Search[华为手机] Observation: 搜索结果 [1]"
    )

    assert parsed.action_text == "Search[华为手机]"
    assert parsed.invocation == ToolInvocation("Search", "华为手机")


def test_multiline_finish_answer_may_contain_an_observation_label():
    parsed = parse_output(
        "Thought: 总结。\nAction: Finish[结果如下：\nObservation: 实验表明 ...]"
    )

    assert parsed.finish_answer == "结果如下：\nObservation: 实验表明 ..."


def test_multiline_tool_input_may_contain_an_observation_label():
    parsed = parse_output(
        "Thought: 搜索。\nAction: Search[第一行\nObservation: 第二行]"
    )

    assert parsed.invocation == ToolInvocation("Search", "第一行\nObservation: 第二行")


@pytest.mark.parametrize(
    "action",
    [
        "`Calculator[1 + 1]`",
        "**Calculator[1 + 1]**",
        "__Calculator[1 + 1]__",
        "*Calculator[1 + 1]*",
    ],
)
def test_action_markdown_wrapper_is_removed(action):
    parsed = parse_output(f"Thought: 计算。\nAction: {action}")

    assert parsed.action_text == "Calculator[1 + 1]"
    assert parsed.invocation == ToolInvocation("Calculator", "1 + 1")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Thought: 完成。\nAction: Finish[42]。", "42"),
        ("Thought: 完成。\nAction: Finish[42].", "42"),
        ("Thought: 完成。\nAction: Finish[42]！", "42"),
    ],
)
def test_finish_tolerates_trailing_punctuation(text, expected):
    assert parse_output(text).finish_answer == expected


def test_tool_action_tolerates_trailing_punctuation():
    parsed = parse_output("Thought: 搜索。\nAction: Search[天气]。")

    assert parsed.invocation == ToolInvocation("Search", "天气")


def test_trailing_code_fence_is_tolerated():
    parsed = parse_output("Thought: 完成。\nAction: Finish[42]\n```")

    assert parsed.finish_answer == "42"


def test_prose_before_a_fenced_response_is_tolerated():
    parsed = parse_output(
        "好的，下面开始：\n```\nThought: 完成。\nAction: Finish[42]\n```"
    )

    assert parsed.finish_answer == "42"
