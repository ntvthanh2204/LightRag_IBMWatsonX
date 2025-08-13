from ..utils import verbose_debug, VERBOSE_DEBUG
import sys
import os
import logging
import json
import time
from typing import Any, Union, List, Dict, Optional

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("httpx"):
    pm.install("httpx")

import httpx
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
    TokenTracker,
)
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.api import __api_version__

import numpy as np
from dotenv import load_dotenv

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env", override=False)


class WatsonXError(Exception):
    """Custom exception class for WatsonX API errors"""
    pass


class WatsonXLLM:
    def __init__(
        self,
        api_key: str,
        project_id: str,
        model_id: str = "openai/gpt-oss-120b",
        base_url: str = "https://us-south.ml.cloud.ibm.com/ml/v1",
        iam_url: str = "https://iam.cloud.ibm.com/identity/token",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        token_tracker: Optional[TokenTracker] = None,
        **kwargs
    ):
        self.api_key = api_key
        self.project_id = project_id
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.iam_url = iam_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.access_token = None
        self.token_expires_at = 0
        self.token_tracker = token_tracker
        
        # Create HTTP client
        self.client = httpx.AsyncClient(timeout=120)
        
        if verbose_debug:
            logger.info(f"WatsonX LLM initialized with model: {model_id}")
            if self.token_tracker:
                logger.info("TokenTracker enabled for WatsonX API monitoring")

    async def _get_access_token(self) -> str:
        """Get access token from IBM IAM"""
        # Check if current token is still valid (with 5 minute buffer)
        if self.access_token and time.time() < (self.token_expires_at - 300):
            return self.access_token
            
        if verbose_debug:
            logger.info("Getting new access token from IBM IAM")
            
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.api_key
        }
        
        try:
            response = await self.client.post(self.iam_url, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            # Set expiration time (tokens usually expire in 1 hour)
            self.token_expires_at = time.time() + token_data.get("expires_in", 3600)
            
            if verbose_debug:
                logger.info("Successfully obtained access token")
                
            return self.access_token
            
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            raise WatsonXError(f"Authentication failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, WatsonXError)),
    )
    async def acompletion(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Generate completion using WatsonX API with detailed timing tracking"""
        import time
        
        # Start timing
        start_time = time.time()
        
        token = await self._get_access_token()
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        # Format messages for WatsonX API
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append({
                    "role": "system",
                    "content": msg["content"]
                })
            elif msg["role"] == "user":
                if isinstance(msg["content"], str):
                    formatted_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                else:
                    formatted_messages.append({
                        "role": "user", 
                        "content": msg["content"]
                    })
            else:
                formatted_messages.append(msg)
        
        payload = {
            "project_id": self.project_id,
            "model_id": self.model_id,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "top_p": kwargs.get("top_p", 1)
        }
        
        url = f"{self.base_url}/text/chat?version=2023-05-29"
        
        if verbose_debug:
            logger.info(f"Making request to WatsonX: {url}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Time the actual API call
            api_start_time = time.time()
            response = await self.client.post(url, headers=headers, json=payload)
            api_end_time = time.time()
            api_call_time = api_end_time - api_start_time
            
            response.raise_for_status()
            
            result = response.json()
            
            # End total timing
            end_time = time.time()
            total_time = end_time - start_time
            
            if verbose_debug:
                logger.debug(f"WatsonX response: {json.dumps(result, indent=2)}")
                logger.info(f"WatsonX API timing - Total: {total_time:.2f}s, API Call: {api_call_time:.2f}s")
            
            # Track token usage with timing if tracker is available
            if self.token_tracker:
                if "usage" in result:
                    usage = result["usage"]
                    token_counts = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                else:
                    # Fallback estimation if usage not provided
                    prompt_text = " ".join([msg.get("content", "") for msg in formatted_messages if isinstance(msg.get("content"), str)])
                    estimated_prompt_tokens = len(prompt_text.split()) * 1.3  # Rough estimation
                    token_counts = {
                        "prompt_tokens": int(estimated_prompt_tokens),
                        "completion_tokens": 0,  # Will be updated when we get the response
                        "total_tokens": int(estimated_prompt_tokens)
                    }
                
                # Extract completion text first to estimate completion tokens
                completion_text = ""
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    if isinstance(content, list) and len(content) > 0:
                        completion_text = content[0].get("text", "")
                    elif isinstance(content, str):
                        completion_text = content
                
                # Update completion tokens if not provided
                if "usage" not in result and completion_text:
                    estimated_completion_tokens = len(completion_text.split()) * 1.3
                    token_counts["completion_tokens"] = int(estimated_completion_tokens)
                    token_counts["total_tokens"] = token_counts["prompt_tokens"] + token_counts["completion_tokens"]
                
                # Determine operation type from kwargs or messages
                operation_type = kwargs.get("operation_type")
                if not operation_type:
                    # Try to infer from message content
                    if any("extract" in str(msg).lower() for msg in formatted_messages):
                        operation_type = "entity_extraction"
                    elif any("query" in str(msg).lower() or "question" in str(msg).lower() for msg in formatted_messages):
                        operation_type = "query_response"
                    elif any("relationship" in str(msg).lower() for msg in formatted_messages):
                        operation_type = "relationship_extraction"
                    else:
                        operation_type = "general_completion"
                
                # Add usage with detailed timing and metadata
                self.token_tracker.add_usage(
                    token_counts=token_counts,
                    call_time=total_time,
                    model_name=self.model_id,
                    operation_type=operation_type
                )
                
                if verbose_debug:
                    logger.info(
                        f"WatsonX Token Usage - Model: {self.model_id}, "
                        f"Operation: {operation_type}, "
                        f"Time: {total_time:.2f}s, "
                        f"Prompt: {token_counts['prompt_tokens']}, "
                        f"Completion: {token_counts['completion_tokens']}, "
                        f"Total: {token_counts['total_tokens']}, "
                        f"Speed: {token_counts['total_tokens']/total_time:.1f} tokens/s"
                    )
            
            # Extract the completion text
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", "")
                elif isinstance(content, str):
                    return content
            
            raise WatsonXError("Unexpected response format from WatsonX API")
            
        except httpx.HTTPStatusError as e:
            logger.error(f"WatsonX API error: {e.response.status_code} - {e.response.text}")
            raise WatsonXError(f"API request failed: {e}")
        except Exception as e:
            logger.error(f"Error calling WatsonX API: {e}")
            raise WatsonXError(f"Request failed: {e}")

    async def aclose(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Factory function for creating WatsonX LLM
def watsonx_llm_complete_if_cache(
    prompt: str,
    system_prompt: str = None,
    history_messages: List = None,
    **kwargs
) -> str:
    """Wrapper function for compatibility with LightRAG"""
    # This will be called by LightRAG's async wrapper
    pass


async def watsonx_llm_acomplete(
    prompt: str,
    system_prompt: str = None,
    history_messages: List = None,
    **kwargs
) -> str:
    """Async completion function for WatsonX"""
    api_key = kwargs.get("api_key") or os.getenv("WATSONX_API_KEY")
    project_id = kwargs.get("project_id") or os.getenv("WATSONX_PROJECT_ID") 
    model_id = kwargs.get("model_id") or os.getenv("WATSONX_MODEL_ID", "openai/gpt-oss-120b")
    base_url = kwargs.get("base_url") or os.getenv("WATSONX_BASE_URL", "https://us-south.ml.cloud.ibm.com/ml/v1")
    token_tracker = kwargs.get("token_tracker")
    
    if not api_key or not project_id:
        raise ValueError("WatsonX API key and project ID are required")
    
    # Remove conflicting parameters from kwargs for WatsonXLLM constructor
    llm_init_keys = ['api_key', 'project_id', 'model_id', 'base_url', 'token_tracker', 'max_tokens', 'temperature', 'iam_url']
    llm_kwargs = {k: v for k, v in kwargs.items() if k not in llm_init_keys}

    llm = WatsonXLLM(
        api_key=api_key,
        project_id=project_id,
        model_id=model_id,
        base_url=base_url,
        token_tracker=token_tracker,
        max_tokens=kwargs.get('max_tokens', 2000),
        temperature=kwargs.get('temperature', 0.7),
        iam_url=kwargs.get('iam_url', 'https://iam.cloud.ibm.com/identity/token'),
        **llm_kwargs
    )

    try:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            messages.extend(history_messages)

        messages.append({"role": "user", "content": prompt})

        # Remove init keys from kwargs before passing to acompletion
        acompletion_kwargs = {k: v for k, v in kwargs.items() if k not in llm_init_keys}
        return await llm.acompletion(messages, **acompletion_kwargs)

    finally:
        await llm.aclose()


# Embedding function placeholder (WatsonX might have embedding endpoints)
@wrap_embedding_func_with_attrs(embedding_dim=None, max_token_size=8192)
async def watsonx_embedding(texts: list[str], **kwargs) -> np.ndarray:
    """WatsonX embedding function - placeholder implementation"""
    logger.warning("WatsonX embedding not implemented. Consider using OpenAI or other embedding providers.")
    # Return dummy embeddings for now
    return np.random.rand(len(texts), 1536).astype(np.float32)