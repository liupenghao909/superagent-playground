"""Run the ReAct loop with a scripted model; no API key required.

    python examples/offline_demo.py
"""

from superagent_playground.demo import run_offline_demo


def main() -> int:
    result = run_offline_demo()
    print()
    if not result.ok:
        print(f"演示失败: {result.reason}")
        return 1
    print(f"最终答案: {result.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
