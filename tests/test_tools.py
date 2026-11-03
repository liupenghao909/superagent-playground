import pytest

from superagent_playground.tools import (
    Tool,
    ToolError,
    ToolRegistry,
    build_default_registry,
    calculate,
    format_search_results,
    serpapi_search,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", "3"),
        ("(12 + 8) * 3", "60"),
        ("2 ** 10", "1024"),
        ("7 // 2", "3"),
        ("7 % 2", "1"),
        ("-3 + 5", "2"),
        ("sqrt(16)", "4"),
        ("round(3.14159, 2)", "3.14"),
        ("abs(-5)", "5"),
    ],
)
def test_calculate_returns_expected_value(expression, expected):
    assert calculate(expression) == expected


def test_calculate_supports_constants():
    assert calculate("pi").startswith("3.14159")
    assert calculate("e").startswith("2.71828")


def test_calculate_rejects_division_by_zero():
    assert "除数不能为 0" in calculate("1 / 0")


def test_calculate_rejects_unsafe_expressions():
    result = calculate("__import__('os').system('echo hacked')")

    assert result.startswith("错误:")
    assert "不支持的表达式" in result or "未知常量" in result or "只支持" in result


def test_calculate_rejects_huge_powers():
    assert "幂运算超出安全范围" in calculate("9 ** 9 ** 9")


def test_calculate_rejects_overlong_expressions():
    assert "不能超过" in calculate("1 + " * 100 + "1")


def test_calculate_rejects_empty_expression():
    assert calculate("   ") == "错误: 计算表达式不能为空。"


def test_calculate_accepts_numeric_strings():
    assert calculate("1.5 * 2") == "3"


def test_registry_registers_and_describes_tools():
    registry = ToolRegistry()
    tool = registry.register("Echo", "原样返回输入。", lambda text: text)

    assert tool.name == "Echo"
    assert registry.get("Echo") is tool
    assert registry.get("Echo").func("hi") == "hi"
    assert registry.names() == ["Echo"]
    assert "- Echo: 原样返回输入。" in registry.describe()
    assert "Echo" in registry
    assert len(registry) == 1


def test_registry_refuses_duplicate_names_by_default():
    registry = ToolRegistry()
    registry.register("Echo", "第一次", lambda text: text)

    with pytest.raises(ToolError):
        registry.register("Echo", "第二次", lambda text: text)


def test_registry_allows_explicit_overwrite():
    registry = ToolRegistry()
    registry.register("Echo", "第一次", lambda text: text)
    registry.register("Echo", "第二次", lambda text: text, overwrite=True)

    assert "第二次" in registry.describe()


def test_registry_validates_tool_definitions():
    registry = ToolRegistry()

    with pytest.raises(ToolError):
        registry.register("", "空名称", lambda text: text)
    with pytest.raises(ToolError):
        registry.register("Bad", "不是函数", "not-callable")  # type: ignore[arg-type]


def test_registry_accepts_tool_instances():
    registry = ToolRegistry([Tool("Echo", "返回输入。", lambda text: text)])

    assert registry.get("Echo").func("ok") == "ok"


def test_unknown_tool_returns_none():
    assert ToolRegistry().get("Missing") is None


def test_default_registry_has_search_and_calculator():
    registry = build_default_registry()

    assert {"Search", "Calculator"} <= set(registry.names())


def test_format_search_results_prefers_answer_box():
    result = format_search_results({"answer_box": {"answer": "42"}})

    assert result == "42"


def test_format_search_results_supports_answer_box_list():
    result = format_search_results({"answer_box_list": ["A", "B"]})

    assert result == "A\nB"


def test_format_search_results_uses_knowledge_graph():
    result = format_search_results(
        {"knowledge_graph": {"description": "知识图谱摘要"}}
    )

    assert result == "知识图谱摘要"


def test_format_search_results_returns_top_three_organic_results():
    results = {
        "organic_results": [
            {"title": f"标题{i}", "snippet": f"摘要{i}"} for i in range(1, 5)
        ]
    }

    text = format_search_results(results, query="测试")

    assert "[1] 标题1" in text
    assert "[2] 标题2" in text
    assert "[3] 标题3" in text
    assert "标题4" not in text


def test_format_search_results_handles_no_results():
    assert "未找到" in format_search_results({}, query="测试")


def test_format_search_results_truncates_long_observations():
    text = format_search_results({"answer_box": {"answer": "x" * 5000}})

    assert len(text) <= 2100
    assert text.endswith("...(结果已截断)")


def test_search_without_api_key_returns_config_error(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.setattr("superagent_playground.tools.load_dotenv", lambda: None)

    result = serpapi_search("华为最新手机")

    assert "SERPAPI_API_KEY" in result
    assert result.startswith("错误:")
