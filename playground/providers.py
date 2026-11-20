"""
playground/providers.py
Member 3: AI Security & Evaluation Engineer
-------------------------------------------
Extensible LLM Provider Adapter Architecture for the AI Playground.
Supports Google Gemini natively while providing a clean BaseLLMProvider
interface so OpenAI, Anthropic, or local Ollama models can be added later.
Never fabricates API completions when credentials or network are unavailable.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ProviderCompletion:
    """Structured result from a real LLM provider API call."""
    provider: str
    model: str
    output_text: str
    latency_ms: float
    word_count: int
    char_count: int
    success: bool
    error: str = ""


class BaseLLMProvider(ABC):
    """Abstract provider contract for multi-LLM A/B testing."""

    name: str = "base"
    display_name: str = "Base Provider"

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if required API keys / endpoints are configured."""

    @abstractmethod
    def available_models(self) -> List[str]:
        """Return supported model identifiers for this provider."""

    @abstractmethod
    def generate(self, prompt: str, model: str | None = None, temperature: float = 0.4) -> ProviderCompletion:
        """Execute a live completion request against the provider."""


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider implementation."""

    name = "gemini"
    display_name = "Google Gemini"

    def is_configured(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY", "").strip())

    def available_models(self) -> List[str]:
        default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        models = [default_model, "gemini-2.5-flash", "gemini-2.5-pro"]
        return list(dict.fromkeys(models))

    def generate(self, prompt: str, model: str | None = None, temperature: float = 0.4) -> ProviderCompletion:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        selected_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        if not api_key:
            return ProviderCompletion(
                provider=self.display_name,
                model=selected_model,
                output_text="",
                latency_ms=0.0,
                word_count=0,
                char_count=0,
                success=False,
                error="GEMINI_API_KEY is missing in .env. Live API execution requires a valid key.",
            )

        start = time.perf_counter()
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=selected_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 1)
            text = (response.text or "").strip()

            return ProviderCompletion(
                provider=self.display_name,
                model=selected_model,
                output_text=text,
                latency_ms=elapsed_ms,
                word_count=len(text.split()),
                char_count=len(text),
                success=True,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 1)
            logger.exception("Gemini completion failed")
            return ProviderCompletion(
                provider=self.display_name,
                model=selected_model,
                output_text="",
                latency_ms=elapsed_ms,
                word_count=0,
                char_count=0,
                success=False,
                error=str(exc),
            )


PROVIDER_REGISTRY: Dict[str, BaseLLMProvider] = {
    "gemini": GeminiProvider(),
}


def get_provider(name: str = "gemini") -> BaseLLMProvider:
    key = (name or "gemini").strip().lower()
    if key not in PROVIDER_REGISTRY:
        raise ValueError(f"Unsupported LLM provider '{name}'. Available: {list(PROVIDER_REGISTRY.keys())}")
    return PROVIDER_REGISTRY[key]


def list_providers() -> List[Dict[str, object]]:
    return [
        {
            "id": key,
            "name": prov.display_name,
            "configured": prov.is_configured(),
            "models": prov.available_models(),
        }
        for key, prov in PROVIDER_REGISTRY.items()
    ]