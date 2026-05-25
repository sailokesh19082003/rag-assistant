import os
import logging
from typing import Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TIMEOUT_SECONDS = 30


def _call_anthropic(prompt: str, system: str) -> Tuple[str, Optional[int]]:
    """Call Claude via Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = response.content[0].text
    tokens_used = response.usage.input_tokens + response.usage.output_tokens
    logger.info(f"Anthropic tokens used: {tokens_used}")
    return reply, tokens_used


def _call_openai(prompt: str, system: str) -> Tuple[str, Optional[int]]:
    """Call OpenAI GPT API."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    reply = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else None
    logger.info(f"OpenAI tokens used: {tokens_used}")
    return reply, tokens_used


def generate_response(prompt: str, system_prompt: str) -> Tuple[str, Optional[int]]:
    """
    Generate a response from the configured LLM.
    Returns (reply_text, tokens_used).
    Handles API errors gracefully.
    """
    provider = LLM_PROVIDER.lower()
    logger.info(f"Calling LLM provider: {provider}")

    try:
        if provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            return _call_anthropic(prompt, system_prompt)

        elif provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment")
            return _call_openai(prompt, system_prompt)

        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}. Use 'anthropic' or 'openai'.")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise

    except httpx.TimeoutException:
        logger.error("LLM API request timed out")
        raise TimeoutError("The AI service timed out. Please try again.")

    except Exception as e:
        error_msg = str(e).lower()
        if "invalid" in error_msg and "api" in error_msg:
            raise PermissionError("Invalid API key. Please check your configuration.")
        if "rate limit" in error_msg or "429" in error_msg:
            raise RuntimeError("Rate limit exceeded. Please wait a moment and try again.")
        logger.exception(f"Unexpected LLM error: {e}")
        raise RuntimeError(f"AI service error: {str(e)}")
