# superagent-playground

一个用于学习和实验经典智能体（Agent）范式的 Python playground。当前实现了两个范式：**ReAct（Reasoning + Acting）** 与 **Plan-and-Solve（先规划，再逐步求解）**，参考 [Hello-Agents 第四章：智能体经典范式构建](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA.md)。

项目刻意保持“手写、可读、少抽象”：提示词、输出解析、工具注册和执行、循环控制都在一个很小的代码库里，方便观察每种范式的每一步，而不是直接套用现成框架。

> 这是教学与实验用途的沙箱项目，不是生产代码。

## 特性

- 两个可切换的经典范式：
  - **ReAct**：`Thought -> Action -> Observation -> ... -> Finish`
  - **Plan-and-Solve**：先由规划器生成分步计划，再按计划逐步执行
- 兼容 OpenAI Chat Completions 接口，可接官方服务或任意兼容网关 / 本地模型
- 内置两个工具：`Search`（SerpApi 网页搜索）与 `Calculator`（安全的算术计算器），供 ReAct 使用
- 可注册自定义工具，工具描述会自动进入提示词
- `max_steps` 安全阀、工具异常兜底、无法解析 Action 时自动把错误反馈给模型重试
- Plan-and-Solve 的计划解析同时支持 Python 列表和 Markdown 编号 / 列表；计划格式错误时会自动重新规划
- LLM 调用失败自动重试 1 次；重试后仍失败则返回兜底答案，而不是中断整个流程
- 结构化返回 `ReActResult` / `PlanAndSolveResult` / `Step` / `PlanStep`，方便测试和二次开发
- 无需 API Key 的离线演示；单元测试覆盖解析、工具、两个循环、配置与 CLI

## 快速开始

要求 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY、LLM_MODEL_ID 等配置
```

### 不配置 API Key，先跑离线演示

ReAct 演示使用一个脚本化的假模型，真实地走完 ReAct 循环（工具调用、Observation 回填、Finish）：

```bash
superagent --demo
# 或者
python examples/offline_demo.py
```

输出：

```text
--- 第 1/3 步 ---
Thought: 这是四则运算，我可以用 Calculator 工具保证计算准确。
Action: Calculator[(12 + 8) * 3]
Observation: 60
--- 第 2/3 步 ---
Thought: 计算结果是 60，60 大于 50，因此可以给出最终答案。
Action: Finish[(12 + 8) * 3 = 60，结果大于 50。]

最终答案: (12 + 8) * 3 = 60，结果大于 50。
```

Plan-and-Solve 演示用同一套脚本化模型走完“规划 -> 逐步执行”：

```bash
superagent --demo --paradigm plan-and-solve
# 或者
python examples/plan_and_solve_demo.py
```

输出：

```text
--- 规划阶段 ---
--- 规划完成，共 3 步 ---
1. 计算周二销量：15 × 2
2. 计算周三销量：周二销量 - 5
3. 计算三天总销量：周一 + 周二 + 周三
--- 执行第 1/3 步 ---
当前步骤: 计算周二销量：15 × 2
结果: 30
--- 执行第 2/3 步 ---
当前步骤: 计算周三销量：周二销量 - 5
结果: 25
--- 执行第 3/3 步 ---
当前步骤: 计算三天总销量：周一 + 周二 + 周三
结果: 70

最终答案: 70
```

### 连接真实模型

```bash
# 默认使用 ReAct
superagent "华为最新的手机是哪一款？它的主要卖点是什么？"

# 使用 Plan-and-Solve
superagent --paradigm plan-and-solve "水果店周一卖出 15 个苹果，周二是周一的 2 倍，周三比周二少 5 个。三天一共卖出多少个？"
```

也可以以模块方式运行：

```bash
python -m superagent_playground "华为最新的手机是哪一款？它的主要卖点是什么？"
```

常用参数：

```bash
superagent --paradigm plan-and-solve "..."  # 切换范式，默认 react
superagent --max-steps 8 "..."              # 放宽步数上限（ReAct 默认 5，Plan-and-Solve 默认 8）
superagent --stream "..."                   # 流式打印模型原始输出
superagent --quiet "..."                    # 只输出最终答案
```

> `Search` 工具需要 [SerpApi](https://serpapi.com/) 的 `SERPAPI_API_KEY`。没有配置时，工具会把配置错误作为 Observation 返回给模型，而不是让程序崩溃；此时 ReAct 循环仍可正常结束。

## 配置

所有配置均可通过环境变量或项目根目录的 `.env` 提供：

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | 是 | OpenAI 兼容服务的 API Key |
| `LLM_MODEL_ID` | 是 | 模型 ID，例如 `gpt-4o-mini` |
| `LLM_BASE_URL` | 否 | 兼容接口地址，默认 `https://api.openai.com/v1` |
| `LLM_TIMEOUT` | 否 | 请求超时秒数，默认 `60` |
| `LLM_TEMPERATURE` | 否 | 采样温度，默认 `0` |
| `SERPAPI_API_KEY` | 否 | `Search` 工具使用的 SerpApi Key |

## 它是怎么工作的

### ReAct

一次 `run` 就是一个受限的循环：

```text
问题
  │
  ▼
构建提示词（工具描述 + 问题 + 历史 Action/Observation）
  │
  ▼
调用 LLM ──► 解析 Thought / Action
                │
      ┌─────────┴──────────┐
      ▼                    ▼
 Finish[答案]          Tool[输入]
      │                    │
      ▼                    ▼
   返回答案          执行工具得到 Observation
                           │
                           └──► 追加到历史，进入下一轮
```

- **Thought**：模型的推理，说明为什么下一步这样做。
- **Action**：`Tool[input]` 或 `Finish[answer]`。
- **Observation**：工具真实执行结果；由代码追加回历史，模型不能编造。

循环最多执行 `max_steps` 次（默认 5），避免无限循环和不可控的 API 消耗。

### Plan-and-Solve

Plan-and-Solve 把一次任务拆成两个阶段，规划阶段只产出计划，执行阶段只解决当前步骤：

```text
问题
  │
  ▼
规划阶段：LLM 生成计划（Python 字符串列表）
  │
  ▼
解析为有序步骤列表
  │
  ▼  对每一步 i
执行阶段：提示词 = 原始问题 + 完整计划 + 已完成步骤和结果 + 当前步骤
  │
  ▼
LLM 输出第 i 步结果 ──► 追加到历史，作为下一步的输入
  │
  ▼  所有步骤完成后
最后一步的结果 = 最终答案
```

- **规划器（Planner）**：只负责拆解问题，要求输出 Python 字符串列表；最后一步要能直接得出最终答案。
- **执行器（Executor）**：每次只执行当前步骤，并把前面步骤的结果作为上下文。
- **计划解析**：优先解析用 `python` 代码块包裹的 Python 字符串列表；也兼容 Markdown 编号列表和 `-` 列表，避免模型的小格式偏差直接导致失败。
- **重新规划**：计划无法解析时会带着错误说明再请求一次（`plan_retries=1`）；仍失败则返回 `reason="plan_error"`，不会把无效计划拿去执行。
- **步数上限**：计划步骤数超过 `max_steps`（默认 8）时直接终止并报告 `max_steps`，不会静默截断计划。

Plan-and-Solve 在本项目中保持经典形态：执行阶段是纯 LLM 推理，不调用外部工具，也不会根据执行结果动态改计划。

### 代码结构

```text
src/superagent_playground/
├── agent.py           # ReActAgent：提示词 -> LLM -> 解析 -> 工具 -> 历史
├── plan_and_solve.py  # PlanAndSolveAgent：规划 -> 计划解析 -> 逐步执行
├── prompt.py          # ReAct / Planner / Executor 提示词模板
├── parser.py          # ReAct Action 解析与 Plan-and-Solve 计划解析
├── tools.py           # 工具注册表、Search、Calculator
├── llm.py             # OpenAI 兼容客户端与配置
├── demo.py            # 两个范式的离线脚本化模型演示
└── cli.py             # 命令行入口（--paradigm 切换范式）
tests/                 # pytest 单元测试
examples/              # 可运行示例
```

### 失败重试与兜底

`OpenAILLM` 会对失败的模型调用自动重试 1 次（构造时可用 `retries` 调整，`retries=0` 表示不重试）。网络错误、服务端错误和空响应都会触发重试；流式模式下，如果已经向调用方输出过 token，则不会重试，以免重复打印。

重试后仍然失败时，两个循环都不会抛错，而是返回兜底结果：

```python
result = agent.run(question)
if result.reason == "llm_fallback":
    print(result.answer)  # 抱歉，模型服务暂时不可用，请稍后重试。
```

此时 `result.ok` 为 `False`，`result.error` 保留原始失败原因；CLI 会打印 `兜底答案: ...` 并以退出码 1 结束。

### 注册自定义工具

工具就是一个“名称 + 描述 + 可调用对象”。描述会原样进入提示词，模型依赖它选择工具，因此要写清楚适用场景：

```python
from datetime import datetime

from superagent_playground import ReActAgent, OpenAILLM, ToolRegistry


def current_time(_: str) -> str:
    return datetime.now().isoformat(timespec="seconds")


tools = ToolRegistry()
tools.register("Now", "返回当前的本地时间。当用户询问当前时间时使用。", current_time)

agent = ReActAgent(OpenAILLM(), tools, max_steps=5)
result = agent.run("现在几点？")
print(result.answer)
```

自定义工具应返回字符串（或可转换为字符串的值）；抛出的异常会被捕获并作为 Observation 反馈给模型。

### 直接使用 Plan-and-Solve 的 API

```python
from superagent_playground import OpenAILLM, PlanAndSolveAgent

agent = PlanAndSolveAgent(OpenAILLM(), max_steps=8)
result = agent.run("一个水果店周一卖出 15 个苹果……")
print(result.plan)          # 规划出的步骤列表
for step in result.steps:   # 每一步的描述和结果
    print(step.index, step.description, step.result)
print(result.answer)        # 最后一步的结果
```

## 测试

```bash
pytest
pytest --cov=superagent_playground
```

测试不访问网络：模型客户端和 Search 的解析逻辑都通过注入的假对象验证。

## 已知局限

- ReAct 的稳定性依赖模型遵循 `Thought/Action` 格式；解析失败时会把格式要求反馈给模型重试，但仍可能耗尽 `max_steps`。
- Plan-and-Solve 的计划是静态的：执行阶段不会调用工具，也不会根据中间结果重新规划；计划质量完全取决于规划器。
- 计划解析失败时最多重试 `plan_retries` 次；仍失败会返回 `plan_error`，而不是猜测执行。
- 计划步骤超过 `max_steps` 时直接终止并返回 `max_steps`；如需执行更长的计划，请调大上限。
- 每次 `run` 都会重置状态，目前没有跨轮对话记忆或持久化。
- `Search` 使用 SerpApi，需要外部 Key 和网络。
- 仅支持同步调用；并发、流式 UI、Reflection 等范式留作后续扩展。

## 参考

- [Hello-Agents 第四章：智能体经典范式构建](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA.md)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/abs/2305.04091)
