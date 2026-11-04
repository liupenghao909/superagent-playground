"""Command line entry point for the classic agent paradigms."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .agent import ReActAgent, ReActResult
from .demo import run_offline_demo, run_plan_and_solve_demo
from .llm import LLMError, OpenAILLM
from .plan_and_solve import PlanAndSolveAgent, PlanAndSolveResult
from .tools import build_default_registry

EXAMPLES = """示例:
  superagent "华为最新的手机是哪一款？它的主要卖点是什么？"
  superagent --paradigm plan-and-solve "一个水果店周一卖出 15 个苹果……"
  superagent --demo            # 无需 API Key 的 ReAct 离线演示
  superagent --demo --paradigm plan-and-solve   # Plan-and-Solve 离线演示
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="superagent",
        description="运行一个使用经典范式的智能体（ReAct 或 Plan-and-Solve）。",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", nargs="*", help="要解决的问题。")
    parser.add_argument(
        "--paradigm",
        choices=("react", "plan-and-solve"),
        default="react",
        help="智能体范式，默认 react。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="ReAct 循环或 Plan-and-Solve 计划执行的最大步数（默认分别为 5 和 8）。",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="流式打印模型原始输出。",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行无需 API Key 的离线演示（默认 ReAct）。",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只输出最终答案。",
    )
    return parser


def _print_result(result: ReActResult | PlanAndSolveResult) -> None:
    if result.reason == "llm_fallback":
        print(f"兜底答案: {result.answer}")
        return
    if result.ok:
        print(f"最终答案: {result.answer}")
        return
    if result.reason == "plan_error":
        print(f"未能生成有效计划: {result.error}", file=sys.stderr)
    elif result.reason == "llm_error":
        print(f"LLM 调用失败: {result.error}", file=sys.stderr)
    elif result.reason == "max_steps":
        detail = result.error or (
            f"已达到最大步数 {len(result.steps)}，未能得到最终答案。"
        )
        print(detail, file=sys.stderr)
    else:
        print("未能得到最终答案。", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    max_steps = args.max_steps
    if max_steps is None:
        max_steps = 8 if args.paradigm == "plan-and-solve" else 5
    if max_steps < 1:
        parser.error("--max-steps 必须大于 0。")

    if args.demo:
        if args.paradigm == "plan-and-solve":
            result = run_plan_and_solve_demo(verbose=not args.quiet)
        else:
            result = run_offline_demo(verbose=not args.quiet)
        _print_result(result)
        return 0 if result.ok else 1

    question = " ".join(args.question).strip()
    if not question:
        parser.error(
            '请提供要解决的问题，例如: superagent "华为最新的手机是哪一款？"'
        )

    try:
        llm = OpenAILLM(
            stream=args.stream,
            on_token=(
                (lambda text: print(text, end="", flush=True))
                if args.stream
                else None
            ),
        )
    except (LLMError, ValueError) as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    agent: ReActAgent | PlanAndSolveAgent
    if args.paradigm == "plan-and-solve":
        agent = PlanAndSolveAgent(
            llm,
            max_steps=max_steps,
            verbose=not args.quiet and not args.stream,
        )
    else:
        agent = ReActAgent(
            llm,
            build_default_registry(),
            max_steps=max_steps,
            verbose=not args.quiet and not args.stream,
        )
    result = agent.run(question)
    if args.stream:
        print()
    _print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
