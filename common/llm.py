"""LLM factory for Gemini, OpenRouter, or a local Ollama server."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

    if provider == "ollama":
        api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        default_model = "qwen3:8b"
        default_base_url = "http://localhost:11434/v1"
        default_timeout = "300"
        max_retries = 0
        extra_body = {"think": False}
    elif provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        default_model = "openai/gpt-4o-mini"
        default_base_url = "https://openrouter.ai/api/v1"
        default_timeout = "60"
        max_retries = 1
        extra_body = None
    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        default_model = "gemini-2.5-flash"
        default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        default_timeout = "60"
        max_retries = 1
        extra_body = None
    else:
        raise RuntimeError("LLM_PROVIDER must be one of: gemini, openrouter, ollama")

    if not api_key:
        raise RuntimeError(f"API key for LLM_PROVIDER={provider!r} is not set in .env")

    llm_kwargs = {
        "model": os.environ.get("LLM_MODEL", default_model),
        "base_url": os.environ.get("LLM_BASE_URL", default_base_url),
        "api_key": api_key,
        "temperature": temperature,
        "timeout": float(os.environ.get("LLM_TIMEOUT_SECONDS", default_timeout)),
        "max_retries": max_retries,
    }
    if extra_body:
        llm_kwargs["extra_body"] = extra_body

    return ChatOpenAI(**llm_kwargs)
