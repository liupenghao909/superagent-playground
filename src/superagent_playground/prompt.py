"""Prompt templates for the agent loops."""

from __future__ import annotations

from typing import Sequence

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


PLANNER_PROMPT_TEMPLATE = """你是一个顶级的 AI 规划专家。你的任务是先理解用户的问题，再把它分解成一个按逻辑顺序排列的分步计划。

要求:
1. 每个步骤都是一个独立、可执行的子任务，包含足够的信息，便于后续逐步执行。
2. 只输出计划，不要直接回答问题，也不要提前执行任何步骤。
3. 最后一步必须能够直接得出问题的最终答案。
4. 输出必须是一个 Python 字符串列表，每个元素是一个步骤，并使用 ```python 代码块包裹。

示例输出:
```python
["第一步：...", "第二步：...", "第三步：..."]
```

Question: {question}
{feedback}"""


EXECUTOR_PROMPT_TEMPLATE = """你是一个顶级的 AI 执行专家。你的任务是严格按照给定计划，一步一步地解决问题。
你将收到原始问题、完整计划、已完成的步骤和结果，以及当前需要执行的步骤。
请只解决“当前步骤”，直接输出该步骤的结果；不要重复计划，也不要输出与当前步骤无关的解释或对话。

# 原始问题
{question}

# 完整计划
{plan}

# 已完成的步骤和结果
{history}

# 当前步骤
{current_step}

请输出“当前步骤”的结果:"""


def build_planner_prompt(
    *,
    question: str,
    feedback: str = "",
    template: str = PLANNER_PROMPT_TEMPLATE,
) -> str:
    """Fill the planner prompt; ``feedback`` explains a failed plan attempt."""

    feedback_text = ""
    if feedback.strip():
        feedback_text = (
            "\n注意：上一次输出无法解析为步骤列表。"
            f"原因：{feedback.strip()}\n"
            "请严格按照上面的格式重新输出计划。"
        )
    return template.format(
        question=question.strip(),
        feedback=feedback_text,
    )


def build_executor_prompt(
    *,
    question: str,
    plan: Sequence[str],
    history: str = "",
    current_step: str,
    template: str = EXECUTOR_PROMPT_TEMPLATE,
) -> str:
    """Fill the executor prompt for one plan step."""

    plan_text = "\n".join(
        f"{index}. {step}" for index, step in enumerate(plan, start=1)
    ) or "(空计划)"
    return template.format(
        question=question.strip(),
        plan=plan_text,
        history=history.strip() or "(暂无)",
        current_step=current_step.strip(),
    )
