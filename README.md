# superagent-playground

一个用于学习和实验经典智能体（Agent）范式的 Python playground。当前实现了第一个范式：**ReAct（Reasoning + Acting）**，参考 [Hello-Agents 第四章：智能体经典范式构建](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA.md)。

项目刻意保持“手写、可读、少抽象”：提示词、输出解析、工具注册和执行、循环控制都在一个很小的代码库里，方便观察 ReAct 的每一步，而不是直接套用现成框架。

> 这是教学与实验用途的沙箱项目，不是生产代码。

## 特性

- 完整的 ReAct 循环：`Thought -> Action -> Observation -> ... -> Finish`
- 兼容 OpenAI Chat Completions 接口，可接官方服务或任意兼容网关 / 本地模型
- 内置两个工具：`Search`（SerpApi 网页搜索）与 `Calculator`（安全的算术计算器）
- 可注册自定义工具，工具描述会自动进入提示词
- `max_steps` 安全阀、工具异常兜底、无法解析 Action 时自动把错误反馈给模型重试
- 结构化返回 `ReActResult` / `Step`，方便测试和二次开发
- 无需 API Key 的离线演示；单元测试覆盖解析、工具、循环、配置与 CLI

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

演示使用一个脚本化的假模型，真实地走完 ReAct 循环（工具调用、Observation 回填、Finish）：

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

### 连接真实模型

```bash
superagent "华为最新的手机是哪一款？它的主要卖点是什么？"
```

也可以以模块方式运行：

```bash
python -m superagent_playground "华为最新的手机是哪一款？它的主要卖点是什么？"
```

常用参数：

```bash
superagent --max-steps 8 "..."   # 放宽循环步数
superagent --stream "..."        # 流式打印模型原始输出
superagent --quiet "..."         # 只输出最终答案
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

### 代码结构

```text
src/superagent_playground/
├── agent.py      # ReActAgent：提示词 -> LLM -> 解析 -> 工具 -> 历史
├── prompt.py     # ReAct 提示词模板
├── parser.py     # 解析 Thought/Action/Finish，容错 Markdown 粗体与代码块
├── tools.py      # 工具注册表、Search、Calculator
├── llm.py        # OpenAI 兼容客户端与配置
├── demo.py       # 离线脚本化模型演示
└── cli.py        # 命令行入口
tests/            # pytest 单元测试
examples/         # 可运行示例
```

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

## 测试

```bash
pytest
pytest --cov=superagent_playground
```

测试不访问网络：模型客户端和 Search 的解析逻辑都通过注入的假对象验证。

## 已知局限

- ReAct 的稳定性依赖模型遵循 `Thought/Action` 格式；解析失败时会把格式要求反馈给模型重试，但仍可能耗尽 `max_steps`。
- 每次 `run` 都会重置历史，目前没有跨轮对话记忆或持久化。
- `Search` 使用 SerpApi，需要外部 Key 和网络。
- 仅支持同步调用；并发、流式 UI、Plan-and-Solve / Reflection 等范式留作后续扩展。

## 参考

- [Hello-Agents 第四章：智能体经典范式构建](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA.md)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
