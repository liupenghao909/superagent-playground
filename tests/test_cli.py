import pytest

from superagent_playground import cli
from superagent_playground.tools import ToolRegistry


class StubLLM:
    def __init__(self, *args, **kwargs):
        pass

    def think(self, messages, *, temperature=None):
        return "Thought: 直接回答。\nAction: Finish[42]"


def test_demo_runs_without_api_key(capsys):
    exit_code = cli.main(["--demo", "--quiet"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "最终答案" in captured.out
    assert "60" in captured.out


def test_demo_verbose_prints_steps(capsys):
    exit_code = cli.main(["--demo"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "--- 第 1/3 步 ---" in captured.out
    assert "Action: Calculator" in captured.out


def test_question_is_required(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])

    assert excinfo.value.code == 2
    assert "请提供要解决的问题" in capsys.readouterr().err


def test_max_steps_must_be_positive():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--max-steps", "0", "--demo"])

    assert excinfo.value.code == 2


def test_missing_configuration_returns_exit_code_2(monkeypatch, capsys):
    monkeypatch.setattr("superagent_playground.llm.load_dotenv", lambda: None)
    for name in ("LLM_API_KEY", "LLM_MODEL_ID", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    exit_code = cli.main(["问题"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "配置错误" in captured.err


def test_cli_uses_injected_model(monkeypatch, capsys):
    monkeypatch.setattr(cli, "OpenAILLM", StubLLM)
    exit_code = cli.main(["问题", "--quiet"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "最终答案: 42" in captured.out


def test_cli_prints_fallback_answer(monkeypatch, capsys):
    class FailingLLM:
        def __init__(self, *args, **kwargs):
            pass

        def think(self, messages, *, temperature=None):
            raise RuntimeError("network down")

    monkeypatch.setattr(cli, "OpenAILLM", FailingLLM)
    exit_code = cli.main(["问题", "--quiet"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "兜底答案" in captured.out


def test_cli_reports_max_steps(monkeypatch, capsys):
    class LoopingLLM:
        def __init__(self, *args, **kwargs):
            pass

        def think(self, messages, *, temperature=None):
            return "Thought: 继续。\nAction: Search[x]"

    registry = ToolRegistry()
    registry.register("Search", "搜索工具。", lambda query: "结果")
    monkeypatch.setattr(cli, "OpenAILLM", LoopingLLM)
    monkeypatch.setattr(cli, "build_default_registry", lambda: registry)

    exit_code = cli.main(["问题", "--quiet", "--max-steps", "2"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "已达到最大步数 2" in captured.err


def test_cli_stream_mode_prints_result(monkeypatch, capsys):
    monkeypatch.setattr(cli, "OpenAILLM", StubLLM)

    exit_code = cli.main(["问题", "--stream"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "最终答案: 42" in captured.out
