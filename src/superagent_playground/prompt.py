"""Prompt template for the ReAct loop."""

from __future__ import annotations

REACT_PROMPT_TEMPLATE = """请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`: 调用一个可用工具。
- `Finish[最终答案]`: 当你认为已经获得最终答案时。

规则:
1. 每次回复必须包含 Thought 和 Action 两个字段。
2. 一次只能调用一个工具，且必须使用可用工具列表中的名称；不要编造 Observation。
3. 当收集到足够的信息、能够回答最终问题时，必须使用 Finish[最终答案] 结束。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


def build_react_prompt(
    *,
    tools: str,
    question: str,
    history: str = "",
    template: str = REACT_PROMPT_TEMPLATE,
) -> str:
    """Fill the ReAct prompt with the available tools and current trajectory."""

    return template.format(
        tools=tools.strip() or "(无可用工具)",
        question=question.strip(),
        history=history.strip() or "(暂无)",
    )
