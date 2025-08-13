from collections.abc import Iterable, AsyncIterator
from typing import Any
import os
import time
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncAzureOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from openai.types.chat import ChatCompletionMessageParam

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    safe_unicode_decode,
    logger,
)

import numpy as np


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APIConnectionError)
    ),
)
async def azure_openai_complete_if_cache(
    model,
    prompt,
    system_prompt: str | None = None,
    history_messages: Iterable[ChatCompletionMessageParam] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    token_tracker: Any | None = None,
    **kwargs,
):
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or model or os.getenv("LLM_MODEL")
    base_url = (
        base_url or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BINDING_HOST")
    )
    api_key = (
        api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    )
    api_version = (
        api_version
        or os.getenv("AZURE_OPENAI_API_VERSION")
        or os.getenv("OPENAI_API_VERSION")
    )

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=base_url,
        azure_deployment=deployment,
        api_key=api_key,
        api_version=api_version,
    )
    kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})

    # Track API call time
    api_start_time = time.time()

    if "response_format" in kwargs:
        response = await openai_async_client.beta.chat.completions.parse(
            model=model, messages=messages, **kwargs
        )
    else:
        response = await openai_async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    if hasattr(response, "__aiter__"):

        async def inner():
            final_chunk_usage = None
            async for chunk in response:
                if hasattr(chunk, "usage") and chunk.usage:
                    final_chunk_usage = chunk.usage
                if len(chunk.choices) == 0:
                    continue
                content = chunk.choices[0].delta.content
                if content is None:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                yield content

            # After streaming complete, log token usage
            if token_tracker and final_chunk_usage:
                token_counts = {
                    "prompt_tokens": getattr(final_chunk_usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(
                        final_chunk_usage, "completion_tokens", 0
                    ),
                    "total_tokens": getattr(final_chunk_usage, "total_tokens", 0),
                }
                elapsed = max(time.time() - api_start_time, 1e-6)
                if hasattr(token_tracker, "add_usage") and callable(
                    token_tracker.add_usage
                ):
                    token_tracker.add_usage(
                        token_counts=token_counts,
                        call_time=elapsed,
                        model_name=model,
                        operation_type="streaming_completion",
                    )
                else:
                    token_tracker.add_usage(token_counts)
                logger.info(
                    f"Azure OpenAI Streaming Usage - Model: {model}, Time: {elapsed:.2f}s, "
                    f"Prompt: {token_counts['prompt_tokens']}, Completion: {token_counts['completion_tokens']}, "
                    f"Total: {token_counts['total_tokens']}, "
                    f"Speed: {token_counts['total_tokens']/elapsed:.1f} tokens/s"
                )

            await openai_async_client.close()

        return inner()
    else:
        try:
            content = response.choices[0].message.content
            if r"\u" in content:
                content = safe_unicode_decode(content.encode("utf-8"))
            if token_tracker and hasattr(response, "usage"):
                token_counts = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(
                        response.usage, "completion_tokens", 0
                    ),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
                elapsed = max(time.time() - api_start_time, 1e-6)
                if hasattr(token_tracker, "add_usage") and callable(
                    token_tracker.add_usage
                ):
                    token_tracker.add_usage(
                        token_counts=token_counts,
                        call_time=elapsed,
                        model_name=model,
                        operation_type="general_completion",
                    )
                else:
                    token_tracker.add_usage(token_counts)
                logger.info(
                    f"Azure OpenAI Token Usage - Model: {model}, Time: {elapsed:.2f}s, "
                    f"Prompt: {token_counts['prompt_tokens']}, Completion: {token_counts['completion_tokens']}, "
                    f"Total: {token_counts['total_tokens']}, "
                    f"Speed: {token_counts['total_tokens']/elapsed:.1f} tokens/s"
                )
            return content
        finally:
            await openai_async_client.close()


async def azure_openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str | AsyncIterator[str]:
    kwargs.pop("keyword_extraction", None)
    result = await azure_openai_complete_if_cache(
        os.getenv("LLM_MODEL", "gpt-4o-mini"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    return result


@wrap_embedding_func_with_attrs(embedding_dim=1536)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def azure_openai_embed(
    texts: list[str],
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
) -> np.ndarray:
    deployment = (
        os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        or model
        or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    base_url = (
        base_url
        or os.getenv("AZURE_EMBEDDING_ENDPOINT")
        or os.getenv("EMBEDDING_BINDING_HOST")
    )
    api_key = (
        api_key
        or os.getenv("AZURE_EMBEDDING_API_KEY")
        or os.getenv("EMBEDDING_BINDING_API_KEY")
    )
    api_version = (
        api_version
        or os.getenv("AZURE_EMBEDDING_API_VERSION")
        or os.getenv("OPENAI_API_VERSION")
    )

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=base_url,
        azure_deployment=deployment,
        api_key=api_key,
        api_version=api_version,
    )

    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
