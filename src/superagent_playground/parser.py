"""Parsing helpers for the ReAct and Plan-and-Solve textual protocols.

ReAct replies contain a ``Thought`` followed by an ``Action``, where the
action is either ``Tool[input]`` or ``Finish[answer]``. Plan-and-Solve replies
contain an ordered list of plan steps, preferably as a Python list literal.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_LABEL_SEP = r"\s*(?:\*\*|__)?\s*[:：]\s*(?:\*\*|__)?\s*"
_THOUGHT_RE = re.compile(
    rf"Thought{_LABEL_SEP}(.*?)(?=\s*(?:\*\*|__)?\s*Action{_LABEL_SEP}|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_ACTION_RE = re.compile(
    rf"Action{_LABEL_SEP}(.*)\Z",
    re.DOTALL | re.IGNORECASE,
)
_FINISH_RE = re.compile(r"^Finish\s*\[(.*)\]\s*$", re.DOTALL | re.IGNORECASE)
_INVOCATION_RE = re.compile(
    r"^([A-Za-z_][\w.-]*)\s*\[(.*)\]\s*$", re.DOTALL
)
_FENCE_RE = re.compile(r"^\s*```[^\n]*\n(.*?)\n?```\s*$", re.DOTALL)
_LIST_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_PLAN_ITEM_RE = re.compile(
    r"^\s*(?:\d+\s*[.)、]|第\s*\d+\s*步\s*[:：.]?|step\s*\d+\s*[:：.]?)\s*(.+?)\s*$",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+?)\s*$")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")


@dataclass(frozen=True)
class ToolInvocation:
    """A call to a registered tool, e.g. ``Search[华为最新手机]``."""

    tool_name: str
    tool_input: str


@dataclass(frozen=True)
class Action:
    """The parsed action of one ReAct step."""

    raw: str
    invocation: ToolInvocation | None = None
    finish_answer: str | None = None

    @property
    def kind(self) -> str:
        """One of ``finish``, ``tool`` or ``invalid``."""

        if self.finish_answer is not None:
            return "finish"
        if self.invocation is not None:
            return "tool"
        return "invalid"


@dataclass(frozen=True)
class ParsedOutput:
    """A model reply split into its thought and action."""

    thought: str | None
    action_text: str | None
    action: Action | None

    @property
    def invocation(self) -> ToolInvocation | None:
        return self.action.invocation if self.action is not None else None

    @property
    def finish_answer(self) -> str | None:
        return self.action.finish_answer if self.action is not None else None

    @property
    def is_finish(self) -> bool:
        return self.finish_answer is not None


def _strip_code_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_action(action_text: str) -> Action:
    """Parse ``Tool[input]`` or ``Finish[answer]``; invalid text yields kind=invalid."""

    raw = (action_text or "").strip()
    if not raw:
        return Action(raw=raw)

    finish_match = _FINISH_RE.match(raw)
    if finish_match:
        answer = finish_match.group(1).strip()
        if answer:
            return Action(raw=raw, finish_answer=answer)
        return Action(raw=raw)

    invocation_match = _INVOCATION_RE.match(raw)
    if invocation_match:
        tool_input = invocation_match.group(2).strip()
        return Action(
            raw=raw,
            invocation=ToolInvocation(
                tool_name=invocation_match.group(1),
                tool_input=tool_input,
            ),
        )

    return Action(raw=raw)


def parse_output(text: str) -> ParsedOutput:
    """Extract the thought and action from a raw model reply."""

    if not text or not text.strip():
        return ParsedOutput(thought=None, action_text=None, action=None)

    cleaned = _strip_code_fence(text)
    thought_match = _THOUGHT_RE.search(cleaned)
    action_match = _ACTION_RE.search(cleaned)

    thought = thought_match.group(1).strip() if thought_match else None
    action_text = action_match.group(1).strip() if action_match else None

    return ParsedOutput(
        thought=thought or None,
        action_text=action_text or None,
        action=parse_action(action_text) if action_text else None,
    )


def _plan_items_from_literal(text: str, *, fenced_only: bool = False) -> list[str]:
    """Return plan steps from a Python list literal, if one is present."""

    candidates = [
        match.group(1).strip() for match in _LIST_FENCE_RE.finditer(text)
    ]
    if not fenced_only:
        body = _strip_code_fence(text)
        candidates.append(body)
        start, end = body.find("["), body.rfind("]")
        if start != -1 and end > start:
            candidates.append(body[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, (list, tuple)):
            continue
        items = [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
        if items:
            return items
    return []


def _plan_items_from_lines(text: str) -> list[str]:
    """Return plan steps from a Markdown-style numbered or bulleted list."""

    items: list[str] = []
    for raw_line in _strip_code_fence(text).splitlines():
        line = raw_line.strip()
        if not line or _HEADING_RE.match(line):
            continue
        match = _PLAN_ITEM_RE.match(line) or _BULLET_RE.match(line)
        if match:
            items.append(match.group(1).strip())
        elif items and raw_line[:1].isspace():
            # An indented line continues the previous step.
            items[-1] = f"{items[-1]} {line}"
    return [item for item in items if item]


def parse_plan(text: str) -> list[str]:
    """Extract the ordered steps of a planner reply.

    The planner is asked for a Python list inside a code fence because that is
    unambiguous. Replies that use a numbered or bulleted Markdown list are
    accepted too, which keeps the agent tolerant of small format slips.
    """

    if not text or not text.strip():
        return []

    # A fenced Python list is the requested format. Only fall back to a list
    # wrapped in prose after checking Markdown lists, so a step like
    # ``1. 输出 ["a", "b"]`` is not mistaken for a bare plan list.
    return (
        _plan_items_from_literal(text, fenced_only=True)
        or _plan_items_from_lines(text)
        or _plan_items_from_literal(text)
    )
