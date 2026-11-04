from superagent_playground.parser import (
    ToolInvocation,
    parse_action,
    parse_output,
    parse_plan,
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


def test_parse_plan_reads_python_list_inside_code_fence():
    plan = parse_plan(
        "```python\n"
        '["计算周二销量", "计算周三销量", "计算总销量"]\n'
        "```"
    )

    assert plan == ["计算周二销量", "计算周三销量", "计算总销量"]


def test_parse_plan_reads_bare_python_list_with_prose():
    plan = parse_plan('计划如下：["第一步", "第二步"]。请开始执行。')

    assert plan == ["第一步", "第二步"]


def test_parse_plan_reads_numbered_markdown_list():
    plan = parse_plan("## 计划\n1. 第一步\n2. 第二步\n3. 第三步")

    assert plan == ["第一步", "第二步", "第三步"]


def test_parse_plan_joins_indented_continuation_lines():
    plan = parse_plan("1. 计算总量\n   并保留两位小数\n2. 给出答案")

    assert plan == ["计算总量 并保留两位小数", "给出答案"]


def test_parse_plan_reads_bulleted_list():
    plan = parse_plan("- 第一步\n- 第二步")

    assert plan == ["第一步", "第二步"]


def test_parse_plan_ignores_non_string_and_empty_items():
    plan = parse_plan('["第一步", 42, "", " ", "第二步"]')

    assert plan == ["第一步", "第二步"]


def test_parse_plan_returns_empty_for_empty_or_unparseable_text():
    assert parse_plan("") == []
    assert parse_plan("   ") == []
    assert parse_plan("我不知道该怎么计划。") == []


def test_parse_plan_accepts_a_single_step_plan():
    assert parse_plan('["直接回答"]') == ["直接回答"]


def test_parse_plan_prefers_numbered_steps_over_embedded_list():
    plan = parse_plan('1. 输出 ["a", "b"]\n2. 解释结果')

    assert plan == ['输出 ["a", "b"]', "解释结果"]
