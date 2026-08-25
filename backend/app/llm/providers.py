"""Localight에서 이식한 LLM 프로바이더 계층 — OpenAI Responses/Chat + 로컬 Ollama.

httpx만 사용하는 자립 모듈(app.* 임포트 없음). 테스트는 transport 주입으로
httpx.MockTransport를 꽂는다 (tests/test_llm_providers.py가 계약을 고정).

이식 시 지켜야 하는 계약:
  * OpenAI 두 경로(/responses, /chat/completions) 모두 store:false — 프라이버시 약속.
  * Chat Completions는 system 프롬프트에 role 'developer' + 최상위 reasoning_effort
    (중첩 reasoning 객체 금지); Responses는 thinking이고 effort가 'default'가 아닐 때만
    중첩 reasoning {effort, summary}.
  * Ollama think 매핑: thinking=False → False, effort∈{low,medium,high} → effort, 그 외 True.
  * :cloud/remote_host 모델은 models() 필터링 + generate() 재검증으로 이중 차단.
  * OpenAI model은 LLM_OPENAI_*_MODELS 허용목록(+기본 모델)에 없으면 요청 전에 거절.
  * 원시 chain-of-thought는 절대 노출하지 않는다 — Responses의 summary_text만 표면화.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx


ProviderId = Literal["openai-responses", "openai-chat", "ollama"]
ReasoningEffort = Literal[
    "default", "none", "minimal", "low", "medium", "high", "xhigh", "max"
]
ReasoningSummary = Literal["none", "auto", "concise", "detailed"]


class LlmError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    size: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    content: str
    model: str
    provider: ProviderId
    reasoning_effort: ReasoningEffort
    thinking_requested: bool
    reasoning_summary: str | None
    usage: dict[str, Any]

    @property
    def remote(self) -> bool:
        return self.provider.startswith("openai-")


def _api_error(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:600]
    except (ValueError, TypeError):
        pass
    return response.text[:600] or f"HTTP {response.status_code}"


class OpenAIClient:
    """Direct client for the OpenAI Responses and Chat Completions APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        responses_models: tuple[str, ...],
        chat_models: tuple[str, ...],
        default_responses_model: str,
        default_chat_model: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.responses_models = responses_models
        self.chat_models = chat_models
        self.default_responses_model = default_responses_model
        self.default_chat_model = default_chat_model
        self.transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def default_model(self, provider: ProviderId) -> str:
        if provider == "openai-responses":
            return self.default_responses_model
        return self.default_chat_model

    def allowed_models(self, provider: ProviderId) -> tuple[str, ...]:
        """LLM_OPENAI_*_MODELS 허용목록 + 해당 공급자의 기본 모델.

        기본 모델이 목록에 없게 설정된 배포에서도 기본 요청은 통과해야 한다."""
        listed = (
            self.responses_models if provider == "openai-responses" else self.chat_models
        )
        return tuple(dict.fromkeys((*listed, self.default_model(provider))))

    async def _post(
        self, path: str, payload: dict[str, Any], timeout: float = 600
    ) -> dict[str, Any]:
        if not self.configured:
            raise LlmError(
                "LLM_OPENAI_API_KEY is not configured. Add it to the host environment or Docker .env file."
            )
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self.transport,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            ) as client:
                response = await client.post(f"{self.base_url}{path}", json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LlmError(f"OpenAI API rejected the request: {_api_error(exc.response)}") from exc
        except httpx.HTTPError as exc:
            raise LlmError(f"OpenAI API request failed: {exc}") from exc
        return response.json()

    @staticmethod
    def _responses_text(data: dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()
        parts: list[str] = []
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(str(content["text"]))
        return "\n".join(parts).strip()

    @staticmethod
    def _responses_summary(data: dict[str, Any]) -> str | None:
        summaries: list[str] = []
        for item in data.get("output", []):
            if item.get("type") != "reasoning":
                continue
            for summary in item.get("summary", []):
                if summary.get("type") == "summary_text" and summary.get("text"):
                    summaries.append(str(summary["text"]))
        return "\n".join(summaries).strip() or None

    async def generate(
        self,
        *,
        provider: ProviderId,
        model: str | None,
        system: str,
        prompt: str,
        reasoning_effort: ReasoningEffort,
        reasoning_summary: ReasoningSummary,
        thinking: bool,
    ) -> GenerationResult:
        if provider not in {"openai-responses", "openai-chat"}:
            raise LlmError(f"Unsupported OpenAI provider: {provider}")
        selected_model = model or self.default_model(provider)
        # 요청 model을 그대로 업스트림에 넘기면 LLM_OPENAI_*_MODELS 허용목록이
        # 실효가 없다 — Ollama의 온디바이스 재검증과 같은 자리에서 차단한다.
        allowed = self.allowed_models(provider)
        if selected_model not in allowed:
            raise LlmError(
                f"'{selected_model}' is not an allowed {provider} model. "
                f"Allowed: {', '.join(allowed)}."
            )

        if provider == "openai-responses":
            payload: dict[str, Any] = {
                "model": selected_model,
                "instructions": system,
                "input": prompt,
                "store": False,
            }
            reasoning: dict[str, str] = {}
            if thinking and reasoning_effort != "default":
                reasoning["effort"] = reasoning_effort
            if thinking and reasoning_summary != "none":
                reasoning["summary"] = reasoning_summary
            if reasoning:
                payload["reasoning"] = reasoning
            data = await self._post("/responses", payload)
            content = self._responses_text(data)
            summary = self._responses_summary(data)
        else:
            payload = {
                "model": selected_model,
                "messages": [
                    {"role": "developer", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "store": False,
            }
            if thinking and reasoning_effort != "default":
                payload["reasoning_effort"] = reasoning_effort
            data = await self._post("/chat/completions", payload)
            message_content = data.get("choices", [{}])[0].get("message", {}).get(
                "content", ""
            )
            if isinstance(message_content, list):
                content = "\n".join(
                    str(part.get("text", ""))
                    for part in message_content
                    if isinstance(part, dict) and part.get("text")
                ).strip()
            else:
                content = str(message_content).strip()
            summary = None

        if not content:
            raise LlmError(f"{provider} returned an empty response")
        return GenerationResult(
            content=content,
            model=str(data.get("model") or selected_model),
            provider=provider,
            reasoning_effort=reasoning_effort,
            thinking_requested=thinking,
            reasoning_summary=summary,
            usage=data.get("usage") or {},
        )


class OllamaClient:
    """Client for an explicitly local Ollama server."""

    def __init__(
        self,
        base_url: str,
        default_model: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.transport = transport

    async def models(self) -> list[ModelInfo]:
        try:
            async with httpx.AsyncClient(timeout=2.5, transport=self.transport) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
        except (httpx.HTTPError, OSError):
            return []

        local_models = []
        for item in response.json().get("models", []):
            name = item.get("name")
            if not name or item.get("remote_host") or name.endswith(":cloud"):
                continue
            local_models.append(ModelInfo(name=name, size=item.get("size")))
        return local_models

    async def generate(
        self,
        *,
        model: str | None,
        system: str,
        prompt: str,
        reasoning_effort: ReasoningEffort,
        reasoning_summary: ReasoningSummary,
        thinking: bool,
    ) -> GenerationResult:
        del reasoning_summary  # Ollama returns raw thinking, not a safe summary.
        selected_model = model or self.default_model
        local_model_names = {candidate.name for candidate in await self.models()}
        if selected_model not in local_model_names:
            raise LlmError(
                f"'{selected_model}' is not an on-device Ollama model. "
                "Cloud models are blocked; install a local model with `ollama pull qwen3:8b`."
            )
        think_value: bool | str = False
        if thinking:
            think_value = (
                reasoning_effort
                if reasoning_effort in {"low", "medium", "high"}
                else True
            )
        payload = {
            "model": selected_model,
            "stream": False,
            "think": think_value,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.1},
        }
        try:
            async with httpx.AsyncClient(timeout=600, transport=self.transport) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LlmError("Ollama is not running. Start it with `ollama serve`.") from exc
        except httpx.HTTPStatusError as exc:
            raise LlmError(
                f"Ollama rejected model '{selected_model}': {_api_error(exc.response)}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LlmError(f"Local Ollama request failed: {exc}") from exc

        data = response.json()
        content = str(data.get("message", {}).get("content", "")).strip()
        if not content:
            raise LlmError("Ollama returned an empty response")
        usage = {
            key: data[key]
            for key in ("prompt_eval_count", "eval_count", "total_duration")
            if key in data
        }
        return GenerationResult(
            content=content,
            model=selected_model,
            provider="ollama",
            reasoning_effort=reasoning_effort,
            thinking_requested=thinking,
            # Never expose Ollama's raw message.thinking chain of thought.
            reasoning_summary=None,
            usage=usage,
        )


class LlmRouter:
    def __init__(
        self,
        *,
        openai: OpenAIClient,
        ollama: OllamaClient,
        default_provider: ProviderId,
        default_reasoning_effort: ReasoningEffort,
    ) -> None:
        self.openai = openai
        self.ollama = ollama
        self.default_provider = default_provider
        self.default_reasoning_effort = default_reasoning_effort

    def default_model(self, provider: ProviderId) -> str:
        if provider.startswith("openai-"):
            return self.openai.default_model(provider)
        return self.ollama.default_model

    async def providers(self) -> dict[str, Any]:
        local_models = await self.ollama.models()
        return {
            "default_provider": self.default_provider,
            "default_reasoning_effort": self.default_reasoning_effort,
            "providers": [
                {
                    "id": "openai-responses",
                    "label": "OpenAI Responses",
                    "available": self.openai.configured,
                    "remote": True,
                    "supports_reasoning_summary": True,
                    "models": list(self.openai.responses_models),
                    "default_model": self.openai.default_responses_model,
                },
                {
                    "id": "openai-chat",
                    "label": "OpenAI Chat Completions",
                    "available": self.openai.configured,
                    "remote": True,
                    "supports_reasoning_summary": False,
                    "models": list(self.openai.chat_models),
                    "default_model": self.openai.default_chat_model,
                },
                {
                    "id": "ollama",
                    "label": "Ollama Local",
                    "available": bool(local_models),
                    "remote": False,
                    "supports_reasoning_summary": False,
                    "models": [model.name for model in local_models],
                    "default_model": self.ollama.default_model,
                },
            ],
        }

    async def _generate(
        self,
        *,
        provider: ProviderId | None,
        model: str | None,
        system: str,
        prompt: str,
        reasoning_effort: ReasoningEffort,
        reasoning_summary: ReasoningSummary,
        thinking: bool,
    ) -> GenerationResult:
        selected_provider = provider or self.default_provider
        if selected_provider.startswith("openai-"):
            return await self.openai.generate(
                provider=selected_provider,
                model=model,
                system=system,
                prompt=prompt,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                thinking=thinking,
            )
        if selected_provider == "ollama":
            return await self.ollama.generate(
                model=model,
                system=system,
                prompt=prompt,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                thinking=thinking,
            )
        raise LlmError(f"Unknown LLM provider: {selected_provider}")

    async def translate(
        self,
        *,
        text: str,
        target_language: str,
        provider: ProviderId | None,
        model: str | None,
        reasoning_effort: ReasoningEffort,
        reasoning_summary: ReasoningSummary,
        thinking: bool,
    ) -> GenerationResult:
        system = (
            "You are a meticulous academic paper translator. Translate only the "
            "provided page into the requested language. Preserve section headings, "
            "paragraph order, citations, equations, list structure, figure/table labels, "
            "and technical terms. Do not summarize, omit, or add commentary. Remove tokens "
            "such as <|12|>. Return only readable translated text with blank lines."
        )
        prompt = f"Target language: {target_language}\n\nPAGE TEXT:\n{text}"
        return await self._generate(
            provider=provider,
            model=model,
            system=system,
            prompt=prompt,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            thinking=thinking,
        )

    async def ask(
        self,
        *,
        question: str,
        context: str,
        provider: ProviderId | None,
        model: str | None,
        reasoning_effort: ReasoningEffort,
        reasoning_summary: ReasoningSummary,
        thinking: bool,
    ) -> GenerationResult:
        system = (
            "You are a research reading assistant. Answer using only the supplied paper "
            "context. If the context is insufficient, say so. Be concise, cite the page "
            "number when present, and never claim access to the rest of the paper."
        )
        prompt = f"PAPER CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        return await self._generate(
            provider=provider,
            model=model,
            system=system,
            prompt=prompt,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            thinking=thinking,
        )
