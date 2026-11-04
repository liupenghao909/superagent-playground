"""Language model clients used by the agent loops."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 60
DEFAULT_TEMPERATURE = 0.0
DEFAULT_RETRIES = 1


class LLMError(RuntimeError):
    """Raised when the model cannot be configured or called."""


@runtime_checkable
class LLM(Protocol):
    """The minimal interface the agent loops need from a language model."""

    def think(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
    ) -> str:
        """Return the model completion for ``messages``."""


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LLMError(f"{name} 必须是整数，当前值为 {raw!r}。") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMError(f"{name} 必须是数字，当前值为 {raw!r}。") from exc


@dataclass(frozen=True)
class LLMConfig:
    """Connection settings for an OpenAI-compatible chat completions API."""

    model: str
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: int = DEFAULT_TIMEOUT
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self) -> None:
        if not self.model:
            raise LLMError("模型 ID 不能为空。")
        if not self.api_key:
            raise LLMError("API Key 不能为空。")
        if self.timeout <= 0:
            raise LLMError("LLM_TIMEOUT 必须大于 0。")
        if not 0.0 <= self.temperature <= 2.0:
            raise LLMError("temperature 必须在 0 到 2 之间。")

    @classmethod
    def resolve(
        cls,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        temperature: float | None = None,
    ) -> "LLMConfig":
        """Build a config from explicit values, falling back to ``.env``/environment."""

        load_dotenv()
        resolved_model = model or _env("LLM_MODEL_ID")
        resolved_api_key = api_key or _env("LLM_API_KEY")
        missing = [
            name
            for name, value in (
                ("LLM_MODEL_ID", resolved_model),
                ("LLM_API_KEY", resolved_api_key),
            )
            if not value
        ]
        if missing:
            raise LLMError(
                "缺少必要的环境变量: "
                + ", ".join(missing)
                + "。请参考 .env.example 配置 .env 文件。"
            )
        return cls(
            model=resolved_model,
            api_key=resolved_api_key,
            base_url=base_url or _env("LLM_BASE_URL") or DEFAULT_BASE_URL,
            timeout=timeout if timeout is not None else _env_int("LLM_TIMEOUT", DEFAULT_TIMEOUT),
            temperature=(
                temperature
                if temperature is not None
                else _env_float("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
            ),
        )

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Build a config from ``.env``/environment variables only."""

        return cls.resolve()


class OpenAILLM:
    """Calls any OpenAI-compatible chat completions endpoint.

    ``client`` can be supplied to inject a pre-built (or fake) OpenAI client,
    which keeps the agent easy to test without network access.

    ``retries`` is the number of extra attempts after one failed call; the
    default of 1 means a transient failure is retried exactly once.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        temperature: float | None = None,
        *,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        client: object | None = None,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        if retries < 0:
            raise LLMError("retries 不能为负数。")
        self.config = LLMConfig.resolve(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
        )
        self.stream = stream
        self.retries = retries
        self._on_token = on_token
        self._client = client

    @property
    def client(self) -> object:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise LLMError(
                    "未安装 openai 包，请执行 `pip install -e .` 安装项目依赖。"
                ) from exc
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                # Keep a single retry policy: ours, not the SDK's default of 2.
                max_retries=0,
            )
        return self._client

    def think(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
    ) -> str:
        """Call the model, retrying a failed call ``retries`` times.

        The agent loops turn the final :class:`LLMError` into a fallback
        answer, so callers only see an exception when no answer can be made.
        """

        if not messages:
            raise LLMError("messages 不能为空。")
        resolved_temperature = (
            self.config.temperature if temperature is None else temperature
        )
        attempts = self.retries + 1
        last_error: LLMError | None = None
        for attempt in range(1, attempts + 1):
            emitted = False

            def on_token(text: str) -> None:
                nonlocal emitted
                emitted = True
                if self._on_token is not None:
                    self._on_token(text)

            try:
                if self.stream:
                    return self._think_stream(messages, resolved_temperature, on_token)
                return self._think_once(messages, resolved_temperature)
            except LLMError as exc:
                last_error = exc
                # Retrying mid-stream would print the same tokens twice.
                if attempt >= attempts or emitted:
                    break
        if attempts == 1:
            raise LLMError(f"调用 LLM 失败: {last_error}") from last_error
        raise LLMError(
            f"调用 LLM 失败（已重试 {self.retries} 次）: {last_error}"
        ) from last_error

    def _create(self, messages: Sequence[Mapping[str, str]], temperature: float, *, stream: bool):
        try:
            return self.client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.config.model,
                messages=list(messages),
                temperature=temperature,
                stream=stream,
            )
        except Exception as exc:
            raise LLMError(str(exc)) from exc

    def _think_once(self, messages: Sequence[Mapping[str, str]], temperature: float) -> str:
        response = self._create(messages, temperature, stream=False)
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("LLM 返回了空响应。")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if not content:
            raise LLMError("LLM 返回了空内容。")
        return content

    def _think_stream(
        self,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        chunks: list[str] = []
        try:
            for chunk in self._create(messages, temperature, stream=True):
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if not content:
                    continue
                chunks.append(content)
                if on_token is not None:
                    on_token(content)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(str(exc)) from exc
        text = "".join(chunks)
        if not text:
            raise LLMError("LLM 返回了空内容。")
        return text
