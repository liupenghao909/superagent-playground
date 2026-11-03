"""Tools exposed to the ReAct agent."""

from __future__ import annotations

import ast
import math
import operator
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from dotenv import load_dotenv

MAX_EXPRESSION_LENGTH = 200
MAX_POWER_EXPONENT = 1000
MAX_POWER_BASE = 1_000_000
MAX_OBSERVATION_LENGTH = 2000
_SECRET_RE = re.compile(r"(api_key=)[^&\s]+", re.IGNORECASE)

SEARCH_DESCRIPTION = (
    "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，"
    "应使用此工具。输入为搜索关键词。"
)
CALCULATOR_DESCRIPTION = (
    "一个安全的算术计算器。支持 + - * / // % **、括号，以及 sqrt、abs、round、min、max、"
    "log、log10、sin、cos、tan、floor、ceil、pi、e。输入为算术表达式。"
)


class ToolError(ValueError):
    """Raised for invalid tool definitions or calculations."""


ToolFunc = Callable[[str], str]


@dataclass(frozen=True)
class Tool:
    """A named tool with a human-readable description for the prompt."""

    name: str
    description: str
    func: ToolFunc


class ToolRegistry:
    """Registers tools and renders the tool list used in the ReAct prompt."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register_tool(tool)

    def register(
        self,
        name: str,
        description: str,
        func: ToolFunc,
        *,
        overwrite: bool = False,
    ) -> Tool:
        """Register a tool; refuse to silently replace an existing one by default."""

        name = (name or "").strip()
        if not name:
            raise ToolError("工具名称不能为空。")
        if not callable(func):
            raise ToolError(f"工具 '{name}' 的执行函数必须可调用。")
        if name in self._tools and not overwrite:
            raise ToolError(f"工具 '{name}' 已注册；如需替换请传入 overwrite=True。")
        tool = Tool(name=name, description=(description or "").strip(), func=func)
        self._tools[name] = tool
        return tool

    def register_tool(self, tool: Tool, *, overwrite: bool = False) -> Tool:
        return self.register(
            tool.name, tool.description, tool.func, overwrite=overwrite
        )

    def get(self, name: str) -> Tool | None:
        return self._tools.get((name or "").strip())

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        if not self._tools:
            return "(无可用工具)"
        return "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self._tools.values()
        )

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip() in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}
_SAFE_CONSTANTS = {"pi": math.pi, "e": math.e}


def _evaluate_expression(node: ast.expr) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("只支持数字常量。")
        return node.value

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)
        if operator_type not in _BINARY_OPERATORS:
            raise ToolError(f"不支持的运算符: {operator_type.__name__}")
        left = _evaluate_expression(node.left)
        right = _evaluate_expression(node.right)
        if operator_type is ast.Pow and (
            abs(right) > MAX_POWER_EXPONENT or abs(left) > MAX_POWER_BASE
        ):
            raise ToolError("幂运算超出安全范围。")
        try:
            return _BINARY_OPERATORS[operator_type](left, right)
        except ZeroDivisionError:
            raise ToolError("除数不能为 0。") from None
        except (OverflowError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)
        if operator_type not in _UNARY_OPERATORS:
            raise ToolError(f"不支持的运算符: {operator_type.__name__}")
        return _UNARY_OPERATORS[operator_type](_evaluate_expression(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
            raise ToolError("只支持预置的数学函数。")
        if node.keywords:
            raise ToolError("数学函数不支持关键字参数。")
        args = [_evaluate_expression(arg) for arg in node.args]
        try:
            return _SAFE_FUNCTIONS[node.func.id](*args)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ToolError(str(exc)) from exc

    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ToolError(f"未知常量: {node.id}")

    raise ToolError(f"不支持的表达式: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate a small arithmetic expression without using ``eval``."""

    expression = (expression or "").strip()
    if not expression:
        return "错误: 计算表达式不能为空。"
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return f"错误: 表达式不能超过 {MAX_EXPRESSION_LENGTH} 个字符。"
    try:
        tree = ast.parse(expression, mode="eval")
        value = _evaluate_expression(tree.body)
    except SyntaxError:
        return f"错误: 无法解析表达式: {expression}"
    except ToolError as exc:
        return f"错误: {exc}"

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "错误: 计算结果不是有限数值。"
        if value.is_integer():
            return str(int(value))
    return str(value)


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_OBSERVATION_LENGTH:
        return text
    return text[:MAX_OBSERVATION_LENGTH].rstrip() + "\n...(结果已截断)"


def format_search_results(
    results: Mapping[str, Any], *, query: str = ""
) -> str:
    """Turn a SerpApi response into a compact observation for the model."""

    answer_box_list = results.get("answer_box_list")
    if isinstance(answer_box_list, list) and answer_box_list:
        return _truncate("\n".join(str(item) for item in answer_box_list))

    answer_box = results.get("answer_box")
    if isinstance(answer_box, Mapping):
        for key in ("answer", "snippet", "title"):
            value = answer_box.get(key)
            if value:
                return _truncate(str(value))

    knowledge_graph = results.get("knowledge_graph")
    if isinstance(knowledge_graph, Mapping):
        for key in ("description", "title"):
            value = knowledge_graph.get(key)
            if value:
                return _truncate(str(value))

    organic_results = results.get("organic_results")
    if isinstance(organic_results, list) and organic_results:
        snippets: list[str] = []
        for index, result in enumerate(organic_results[:3], start=1):
            if not isinstance(result, Mapping):
                continue
            title = str(result.get("title") or "").strip()
            snippet = str(result.get("snippet") or "").strip()
            block = f"[{index}] {title}\n{snippet}".strip()
            if block:
                snippets.append(block)
        if snippets:
            return _truncate("\n\n".join(snippets))

    if query:
        return f"未找到关于 '{query}' 的搜索结果。"
    return "未找到搜索结果。"


def serpapi_search(query: str) -> str:
    """Search the web through SerpApi and return a short answer or snippets."""

    load_dotenv()
    query = (query or "").strip()
    if not query:
        return "错误: 搜索关键词不能为空。"

    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        return "错误: SERPAPI_API_KEY 未配置，无法使用 Search 工具。"
    try:
        from serpapi import SerpApiClient
    except ImportError:
        return "错误: 未安装 google-search-results，无法使用 Search 工具。"

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "gl": "cn",
        "hl": "zh-cn",
    }
    try:
        results = SerpApiClient(params).get_dict()
    except Exception as exc:
        detail = _SECRET_RE.sub(r"\1***", str(exc))
        return f"错误: 搜索失败: {detail}"
    if not isinstance(results, Mapping):
        return "错误: 搜索服务返回了无法识别的结果。"
    return format_search_results(results, query=query)


def build_default_registry() -> ToolRegistry:
    """Create a registry with the tools used by the demo and CLI."""

    registry = ToolRegistry()
    registry.register("Search", SEARCH_DESCRIPTION, serpapi_search)
    registry.register("Calculator", CALCULATOR_DESCRIPTION, calculate)
    return registry
