"""Parsing helpers for the ReAct textual protocol.

The protocol is intentionally small: the model replies with a ``Thought``
followed by an ``Action``, and the action is either ``Tool[input]`` or
``Finish[answer]``.
"""

from __future__ import annotations

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
_TRAILING_PUNCT_RE = r"[。．.，,；;！!？?]"
_FINISH_RE = re.compile(
    rf"^Finish\s*\[(.*)\]\s*{_TRAILING_PUNCT_RE}?\s*$",
    re.DOTALL | re.IGNORECASE,
)
# Tool names may be any Unicode word start (plus ``_``); a leading digit is
# rejected so that ``1[2]`` is never mistaken for a tool invocation.
_INVOCATION_RE = re.compile(
    rf"^([^\W\d][\w.-]*)\s*\[(.*)\]\s*{_TRAILING_PUNCT_RE}?\s*$",
    re.DOTALL,
)
# A model may append a fabricated ``Observation:`` (or a second ``Action:``)
# after the real action, either on the next line or on the same line. Only cut
# when the action already has a closing bracket, so a legitimate multiline
# Finish answer is left untouched.
_TRAILING_LABEL_RE = re.compile(
    rf"(?:\n[ \t]*|(?<=\])[ \t]+)(?:\*\*|__)?[ \t]*(?:Observation|Action){_LABEL_SEP}",
    re.IGNORECASE,
)
_TRAILING_FENCE_RE = re.compile(r"\n[ \t]*```[^\n]*\s*$")
_INLINE_WRAPPERS = (("**", "**"), ("__", "__"), ("`", "`"), ("*", "*"))
_FENCE_RE = re.compile(r"^\s*```[^\n]*\n(.*?)\n?```\s*$", re.DOTALL)


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
    cleaned = text.strip()
    if _TRAILING_FENCE_RE.search(cleaned):
        cleaned = _TRAILING_FENCE_RE.sub("", cleaned).strip()
    return cleaned


def _strip_inline_wrapper(text: str) -> str:
    """Remove a matched Markdown wrapper around the whole action value."""

    text = text.strip()
    # ``Action: **Calculator[...]**`` loses its opening ``**`` to the label
    # separator, so the closing marker is left dangling after the brackets.
    for marker in ("**", "__"):
        if (
            text.endswith(marker)
            and not text.startswith(marker)
            and "]" in text[: -len(marker)]
        ):
            text = text[: -len(marker)].rstrip()
    changed = True
    while changed:
        changed = False
        for left, right in _INLINE_WRAPPERS:
            if (
                len(text) > len(left) + len(right)
                and text.startswith(left)
                and text.endswith(right)
            ):
                inner = text[len(left) : -len(right)].strip()
                if inner:
                    text = inner
                    changed = True
                    break
    return text


def _cut_trailing_labels(text: str) -> str:
    """Drop a fabricated ``Observation:``/``Action:`` after the real action."""

    match = _TRAILING_LABEL_RE.search(text)
    if match is None or "]" not in text[: match.start()]:
        return text
    return text[: match.start()].rstrip()


def parse_action(action_text: str) -> Action:
    """Parse ``Tool[input]`` or ``Finish[answer]``; invalid text yields kind=invalid."""

    raw = _strip_inline_wrapper(action_text or "")
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
    if action_text:
        action_text = _strip_inline_wrapper(_cut_trailing_labels(action_text))

    return ParsedOutput(
        thought=thought or None,
        action_text=action_text or None,
        action=parse_action(action_text) if action_text else None,
    )
