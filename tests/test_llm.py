from types import SimpleNamespace

import pytest

from superagent_playground.llm import (
    DEFAULT_BASE_URL,
    LLMConfig,
    LLMError,
    OpenAILLM,
)


class FakeCompletions:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FakeClient:
    def __init__(self, result):
        self.completions = FakeCompletions(result)
        self.chat = SimpleNamespace(completions=self.completions)


def text_response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def stream_chunks(*texts):
    chunks = []
    for text in texts:
        chunks.append(
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
            )
        )
    chunks.append(SimpleNamespace(choices=[]))
    return iter(chunks)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setattr("superagent_playground.llm.load_dotenv", lambda: None)
    for name in (
        "LLM_API_KEY",
        "LLM_MODEL_ID",
        "LLM_BASE_URL",
        "LLM_TIMEOUT",
        "LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(name, raising=False)


def clear_env(monkeypatch):
    for name in (
        "LLM_API_KEY",
        "LLM_MODEL_ID",
        "LLM_BASE_URL",
        "LLM_TIMEOUT",
        "LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_config_reads_environment(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL_ID", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_TIMEOUT", "10")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")

    config = LLMConfig.from_env()

    assert config.model == "test-model"
    assert config.api_key == "test-key"
    assert config.base_url == "http://localhost:8000/v1"
    assert config.timeout == 10
    assert config.temperature == 0.5


def test_config_defaults_base_url(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL_ID", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    assert LLMConfig.from_env().base_url == DEFAULT_BASE_URL


def test_config_requires_model_and_key(monkeypatch):
    clear_env(monkeypatch)

    with pytest.raises(LLMError) as excinfo:
        LLMConfig.from_env()

    assert "LLM_MODEL_ID" in str(excinfo.value)
    assert "LLM_API_KEY" in str(excinfo.value)


def test_config_rejects_invalid_timeout(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL_ID", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_TIMEOUT", "not-a-number")

    with pytest.raises(LLMError, match="LLM_TIMEOUT"):
        LLMConfig.from_env()


def test_openai_llm_returns_completion():
    client = FakeClient(text_response("hello"))
    llm = OpenAILLM(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
        client=client,
    )

    text = llm.think([{"role": "user", "content": "hi"}])

    assert text == "hello"
    call = client.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["temperature"] == 0
    assert call["stream"] is False


def test_openai_llm_streams_tokens():
    client = FakeClient(stream_chunks("Thought: ", "ok", "\nAction: Finish[ok]"))
    tokens = []
    llm = OpenAILLM(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
        client=client,
        stream=True,
        on_token=tokens.append,
    )

    text = llm.think([{"role": "user", "content": "hi"}])

    assert text == "Thought: ok\nAction: Finish[ok]"
    assert tokens == ["Thought: ", "ok", "\nAction: Finish[ok]"]
    assert client.completions.calls[0]["stream"] is True


def test_openai_llm_rejects_empty_completion():
    client = FakeClient(SimpleNamespace(choices=[]))
    llm = OpenAILLM(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
        client=client,
    )

    with pytest.raises(LLMError, match="空响应"):
        llm.think([{"role": "user", "content": "hi"}])


def test_openai_llm_rejects_empty_messages():
    llm = OpenAILLM(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
        client=FakeClient(text_response("x")),
    )

    with pytest.raises(LLMError):
        llm.think([])


def test_openai_llm_wraps_api_errors():
    class BrokenCompletions:
        def create(self, **kwargs):
            raise RuntimeError("502 bad gateway")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=BrokenCompletions())
    )
    llm = OpenAILLM(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
        client=client,
    )

    with pytest.raises(LLMError, match="502 bad gateway"):
        llm.think([{"role": "user", "content": "hi"}])


class ScriptedCompletions:
    """Raise or return each queued outcome, one per ``create`` call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def scripted_llm(outcomes, **kwargs):
    completions = ScriptedCompletions(outcomes)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = OpenAILLM(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
        client=client,
        **kwargs,
    )
    return llm, completions


def test_openai_llm_retries_a_failed_call_once():
    llm, completions = scripted_llm(
        [RuntimeError("502 bad gateway"), text_response("hello")]
    )

    assert llm.think([{"role": "user", "content": "hi"}]) == "hello"
    assert completions.calls == 2


def test_openai_llm_raises_after_retry_is_exhausted():
    llm, completions = scripted_llm(
        [RuntimeError("down"), RuntimeError("still down")]
    )

    with pytest.raises(LLMError, match="已重试 1 次"):
        llm.think([{"role": "user", "content": "hi"}])

    assert completions.calls == 2


def test_openai_llm_retries_an_empty_completion():
    llm, completions = scripted_llm([SimpleNamespace(choices=[]), text_response("ok")])

    assert llm.think([{"role": "user", "content": "hi"}]) == "ok"
    assert completions.calls == 2


def test_openai_llm_can_disable_retries():
    llm, completions = scripted_llm([RuntimeError("down")], retries=0)

    with pytest.raises(LLMError, match="调用 LLM 失败: down"):
        llm.think([{"role": "user", "content": "hi"}])

    assert completions.calls == 1


def test_openai_llm_rejects_negative_retries():
    with pytest.raises(LLMError, match="retries"):
        OpenAILLM(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:8000/v1",
            client=FakeClient(text_response("x")),
            retries=-1,
        )


def test_openai_llm_retries_stream_before_any_token():
    llm, completions = scripted_llm(
        [RuntimeError("down"), stream_chunks("hello")],
        stream=True,
    )

    assert llm.think([{"role": "user", "content": "hi"}]) == "hello"
    assert completions.calls == 2


def test_openai_llm_does_not_retry_after_partial_stream():
    class PartialStreamCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1

            def chunks():
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="部分"))]
                )
                raise RuntimeError("stream broke")

            return chunks()

    completions = PartialStreamCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    tokens = []
    llm = OpenAILLM(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
        client=client,
        stream=True,
        on_token=tokens.append,
    )

    with pytest.raises(LLMError, match="stream broke") as excinfo:
        llm.think([{"role": "user", "content": "hi"}])

    assert "已重试" not in str(excinfo.value)
    assert tokens == ["部分"]
    assert completions.calls == 1
