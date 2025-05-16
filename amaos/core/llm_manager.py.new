"""LLM Manager for AMAOS.

This module provides a manager for handling LLM requests with fallback support across
multiple providers including OpenAI, Anthropic, Google, and Ollama.
"""

import asyncio
import enum
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator


class ModelStatus(enum.Enum):
    """Status of a model response."""

    SUCCESS = "success"  # Model returned a valid response
    TIMEOUT = "timeout"  # Request timed out
    ERROR = "error"  # Model returned an error
    UNAVAILABLE = "unavailable"  # Model is not available (e.g., API key not configured)


@dataclass
class ModelResult:
    """Result from an LLM model."""

    content: str
    status: ModelStatus
    model_name: str
    tokens: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    cost: float = 0.0
    latency: float = 0.0
    raw_response: Any = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Ensure tokens dictionary has the required keys."""
        # Set default values for tokens if not provided
        if "prompt" not in self.tokens:
            self.tokens["prompt"] = 0
        if "completion" not in self.tokens:
            self.tokens["completion"] = 0
        if "total" not in self.tokens:
            self.tokens["total"] = self.tokens["prompt"] + self.tokens["completion"]


class ModelProvider(str, enum.Enum):
    """Supported model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    MOCK = "mock"  # For testing


class ModelConfig(BaseModel):
    """Configuration for a language model."""

    name: str
    provider: ModelProvider
    api_key_env: Optional[str] = None
    priority: int = 50  # Higher priority = preferred
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 30.0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    local: bool = False  # Whether this is a local model

    @validator("priority")
    def validate_priority(cls, v: int) -> int:
        """Validate priority is in range."""
        if v < 0 or v > 100:
            raise ValueError("Priority must be between 0 and 100")
        return v


class LLMManagerConfig(BaseModel):
    """Configuration for the LLM Manager."""

    models: List[ModelConfig] = Field(default_factory=list)
    default_system_prompt: str = "You are a helpful AI assistant."
    default_max_tokens: int = 1024
    default_temperature: float = 0.7
    local_only: bool = False
    max_attempts: int = 3  # Maximum number of fallback attempts
    max_retries: int = 2  # Maximum number of retries per model
    retry_delay: float = 1.0  # Delay between retries in seconds


class LLMManager:
    """Manager for handling LLM requests with fallback support."""

    def __init__(self, config: Optional[LLMManagerConfig] = None) -> None:
        """Initialize the LLM Manager.
        
        Args:
            config: Configuration for the LLM Manager.
        """
        self.config = config or LLMManagerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Add default models if none provided
        if not self.config.models:
            self._add_default_models()
            
        # Sort models by priority
        self.models = sorted(
            self.config.models, 
            key=lambda m: m.priority, 
            reverse=True
        )
        
        # Initialize stats
        self.stats: Dict[str, Any] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "fallbacks": 0,
            "retries": 0,
            "by_model": {},
            "by_provider": {},
        }

    def _add_default_models(self) -> None:
        """Add default models to the configuration."""
        self.config.models.extend([
            ModelConfig(
                name="gpt-4o",
                provider=ModelProvider.OPENAI,
                api_key_env="OPENAI_API_KEY",
                priority=90,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.03,
            ),
            ModelConfig(
                name="claude-3-opus",
                provider=ModelProvider.ANTHROPIC,
                api_key_env="ANTHROPIC_API_KEY",
                priority=85,
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.075,
            ),
            ModelConfig(
                name="gemini-1.5-pro",
                provider=ModelProvider.GOOGLE,
                api_key_env="GOOGLE_API_KEY",
                priority=80,
                cost_per_1k_input=0.0025,
                cost_per_1k_output=0.0025,
            ),
            ModelConfig(
                name="llama3",
                provider=ModelProvider.OLLAMA,
                priority=70,
                local=True,
            ),
        ])

    def _get_api_key(self, model: ModelConfig) -> Optional[str]:
        """Get the API key for a model.
        
        Args:
            model: Model configuration.
            
        Returns:
            API key if found, None otherwise.
        """
        if not model.api_key_env:
            return None
            
        return os.environ.get(model.api_key_env)

    def _filter_available_models(self) -> List[ModelConfig]:
        """Filter models based on availability and local_only setting.
        
        Returns:
            List of available models.
        """
        available_models = []
        
        for model in self.models:
            # Skip cloud models if local_only is enabled
            if self.config.local_only and not model.local:
                continue
                
            # Skip models that require API keys if not configured
            if model.api_key_env and not self._get_api_key(model):
                self.logger.debug(
                    f"Skipping {model.name}: API key not found in environment variable {model.api_key_env}"
                )
                continue
                
            available_models.append(model)
            
        return available_models

    def _update_stats(self, model: ModelConfig, status: ModelStatus, latency: float) -> None:
        """Update usage statistics.
        
        Args:
            model: Model used.
            status: Status of the response.
            latency: Latency of the request.
        """
        # Update total stats
        self.stats["total_calls"] += 1
        
        if status == ModelStatus.SUCCESS:
            self.stats["successful_calls"] += 1
        else:
            self.stats["failed_calls"] += 1
            
        # Update model-specific stats
        if model.name not in self.stats["by_model"]:
            self.stats["by_model"][model.name] = {
                "calls": 0,
                "successful": 0,
                "failed": 0,
                "avg_latency": 0.0,
                "total_latency": 0.0,
            }
            
        model_stats = self.stats["by_model"][model.name]
        model_stats["calls"] += 1
        
        if status == ModelStatus.SUCCESS:
            model_stats["successful"] += 1
        else:
            model_stats["failed"] += 1
            
        model_stats["total_latency"] += latency
        model_stats["avg_latency"] = model_stats["total_latency"] / model_stats["calls"]
        
        # Update provider-specific stats
        if model.provider not in self.stats["by_provider"]:
            self.stats["by_provider"][model.provider] = {
                "calls": 0,
                "successful": 0,
                "failed": 0,
            }
            
        provider_stats = self.stats["by_provider"][model.provider]
        provider_stats["calls"] += 1
        
        if status == ModelStatus.SUCCESS:
            provider_stats["successful"] += 1
        else:
            provider_stats["failed"] += 1

    async def complete(
        self,
        prompt: str,
        force_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ModelResult:
        """Complete a prompt using an LLM with fallback support.
        
        Args:
            prompt: The prompt to complete.
            force_model: Force the use of a specific model.
            system_prompt: System prompt to use.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature for generation.
            
        Returns:
            Result from the LLM.
        """
        # Use default values if not provided
        system_prompt = system_prompt or self.config.default_system_prompt
        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = temperature or self.config.default_temperature
        
        # Get available models
        available_models = self._filter_available_models()
        
        if not available_models:
            self.logger.warning("No available models found. Using mock response.")
            return await self._call_mock(
                "mock", prompt, system_prompt, max_tokens, temperature
            )
            
        # If force_model is specified, filter to only that model
        if force_model:
            available_models = [m for m in available_models if m.name == force_model]
            
            if not available_models:
                self.logger.warning(f"Forced model {force_model} not available. Using mock response.")
                return await self._call_mock(
                    force_model, prompt, system_prompt, max_tokens, temperature
                )
        
        # Try models in order with fallback
        attempts = 0
        tried_models: List[str] = []
        
        for model in available_models:
            # Check if we've exceeded the maximum attempts
            if attempts >= self.config.max_attempts:
                self.logger.warning(f"Exceeded maximum attempts ({self.config.max_attempts}). Using mock response.")
                return await self._call_mock(
                    "mock_fallback", prompt, system_prompt, max_tokens, temperature
                )
                
            # Skip models we've already tried
            if model.name in tried_models:
                continue
                
            tried_models.append(model.name)
            attempts += 1
            
            # Try the model with retries
            for retry in range(self.config.max_retries + 1):
                try:
                    start_time = time.time()
                    
                    # Call the appropriate provider
                    result = await self._call_provider(
                        model, prompt, system_prompt, max_tokens, temperature
                    )
                    
                    # Calculate latency
                    latency = time.time() - start_time
                    result.latency = latency
                    
                    # Update stats
                    self._update_stats(model, result.status, latency)
                    
                    # If successful, return the result
                    if result.status == ModelStatus.SUCCESS:
                        return result
                        
                    # If this is not the last retry, log and retry
                    if retry < self.config.max_retries:
                        self.logger.warning(
                            f"Model {model.name} failed with status {result.status}. "
                            f"Retrying ({retry + 1}/{self.config.max_retries})..."
                        )
                        self.stats["retries"] += 1
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                        
                    # If we've exhausted retries, log and try the next model
                    self.logger.warning(
                        f"Model {model.name} failed after {self.config.max_retries} retries. "
                        f"Trying next model."
                    )
                    self.stats["fallbacks"] += 1
                    break
                    
                except Exception as e:
                    # Log the error
                    self.logger.error(f"Error calling model {model.name}: {e}")
                    
                    # If this is not the last retry, retry
                    if retry < self.config.max_retries:
                        self.logger.warning(
                            f"Retrying model {model.name} ({retry + 1}/{self.config.max_retries})..."
                        )
                        self.stats["retries"] += 1
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                        
                    # If we've exhausted retries, try the next model
                    self.logger.warning(
                        f"Model {model.name} failed after {self.config.max_retries} retries. "
                        f"Trying next model."
                    )
                    self.stats["fallbacks"] += 1
                    break
        
        # If we've tried all models and none worked, use mock response
        self.logger.warning("All models failed. Using mock response.")
        return await self._call_mock(
            "mock_fallback", prompt, system_prompt, max_tokens, temperature
        )

    async def _call_provider(
        self,
        model: ModelConfig,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> ModelResult:
        """Call the appropriate provider based on the model configuration.
        
        Args:
            model: Model configuration.
            prompt: The prompt to complete.
            system_prompt: System prompt to use.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature for generation.
            
        Returns:
            Result from the LLM.
        """
        if model.provider == ModelProvider.OPENAI:
            return await self._call_openai(
                model.name, prompt, system_prompt, max_tokens, temperature, model.timeout
            )
        elif model.provider == ModelProvider.ANTHROPIC:
            return await self._call_anthropic(
                model.name, prompt, system_prompt, max_tokens, temperature, model.timeout
            )
        elif model.provider == ModelProvider.GOOGLE:
            return await self._call_google(
                model.name, prompt, system_prompt, max_tokens, temperature, model.timeout
            )
        elif model.provider == ModelProvider.OLLAMA:
            return await self._call_local(
                model.name, prompt, system_prompt, max_tokens, temperature, model.timeout
            )
        else:
            return await self._call_mock(
                model.name, prompt, system_prompt, max_tokens, temperature
            )

    async def _call_openai(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> ModelResult:
        """Call OpenAI API.
        
        Args:
            model_name: Name of the model to use.
            prompt: The prompt to complete.
            system_prompt: System prompt to use.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature for generation.
            timeout: Timeout for the request.
            
        Returns:
            Result from the LLM.
        """
        # Check if API key is configured
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.logger.warning("OpenAI API key not configured. Using mock response.")
            return ModelResult(
                content="",
                status=ModelStatus.UNAVAILABLE,
                model_name=model_name,
                error_message="OpenAI API key not configured",
            )
            
        # In a real implementation, we would call the OpenAI API here
        # For now, just return a mock response
        self.logger.info(f"Calling OpenAI model: {model_name}")
        
        # Simulate some processing time
        await asyncio.sleep(0.5)
        
        # Simulate a successful response
        return ModelResult(
            content=f"This is a mock response from OpenAI model {model_name}. You asked: {prompt[:50]}...",
            status=ModelStatus.SUCCESS,
            model_name=model_name,
            tokens={
                "prompt": len(prompt) // 4,
                "completion": 50,
                "total": len(prompt) // 4 + 50,
            },
            cost=0.0001 * (len(prompt) // 4 + 50),
            latency=0.5,
            raw_response={"model": model_name, "mock": True},
        )

    async def _call_anthropic(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> ModelResult:
        """Call Anthropic API.
        
        Args:
            model_name: Name of the model to use.
            prompt: The prompt to complete.
            system_prompt: System prompt to use.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature for generation.
            timeout: Timeout for the request.
            
        Returns:
            Result from the LLM.
        """
        # Check if API key is configured
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self.logger.warning("Anthropic API key not configured. Using mock response.")
            return ModelResult(
                content="",
                status=ModelStatus.UNAVAILABLE,
                model_name=model_name,
                error_message="Anthropic API key not configured",
            )
            
        # In a real implementation, we would call the Anthropic API here
        # For now, just return a mock response
        self.logger.info(f"Calling Anthropic model: {model_name}")
        
        # Simulate some processing time
        await asyncio.sleep(0.7)
        
        # Simulate a successful response
        return ModelResult(
            content=f"This is a mock response from Anthropic model {model_name}. You asked: {prompt[:50]}...",
            status=ModelStatus.SUCCESS,
            model_name=model_name,
            tokens={
                "prompt": len(prompt) // 4,
                "completion": 60,
                "total": len(prompt) // 4 + 60,
            },
            cost=0.0002 * (len(prompt) // 4 + 60),
            latency=0.7,
            raw_response={"model": model_name, "mock": True},
        )

    async def _call_google(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> ModelResult:
        """Call Google API.
        
        Args:
            model_name: Name of the model to use.
            prompt: The prompt to complete.
            system_prompt: System prompt to use.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature for generation.
            timeout: Timeout for the request.
            
        Returns:
            Result from the LLM.
        """
        # Check if API key is configured
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            self.logger.warning("Google API key not configured. Using mock response.")
            return ModelResult(
                content="",
                status=ModelStatus.UNAVAILABLE,
                model_name=model_name,
                error_message="Google API key not configured",
            )
            
        # In a real implementation, we would call the Google API here
        # For now, just return a mock response
        self.logger.info(f"Calling Google model: {model_name}")
        
        # Simulate some processing time
        await asyncio.sleep(0.6)
        
        # Simulate a successful response
        return ModelResult(
            content=f"This is a mock response from Google model {model_name}. You asked: {prompt[:50]}...",
            status=ModelStatus.SUCCESS,
            model_name=model_name,
            tokens={
                "prompt": len(prompt) // 4,
                "completion": 55,
                "total": len(prompt) // 4 + 55,
            },
            cost=0.00015 * (len(prompt) // 4 + 55),
            latency=0.6,
            raw_response={"model": model_name, "mock": True},
        )

    async def _call_local(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> ModelResult:
        """Call local model (e.g., Ollama).
        
        Args:
            model_name: Name of the model to use.
            prompt: The prompt to complete.
            system_prompt: System prompt to use.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature for generation.
            timeout: Timeout for the request.
            
        Returns:
            Result from the LLM.
        """
        # In a real implementation, we would call the local model here
        # For now, just return a mock response
        self.logger.info(f"Calling local model: {model_name}")
        
        # Simulate some processing time
        await asyncio.sleep(1.0)
        
        # Simulate a successful response
        return ModelResult(
            content=f"This is a mock response from local model {model_name}. You asked: {prompt[:50]}...",
            status=ModelStatus.SUCCESS,
            model_name=model_name,
            tokens={
                "prompt": len(prompt) // 4,
                "completion": 45,
                "total": len(prompt) // 4 + 45,
            },
            cost=0.0,  # Local models don't have API costs
            latency=1.0,
            raw_response={"model": model_name, "mock": True},
        )

    async def _call_mock(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> ModelResult:
        """Generate a mock response.
        
        Args:
            model_name: Name of the model to use.
            prompt: The prompt to complete.
            system_prompt: System prompt to use.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature for generation.
            
        Returns:
            Mock result.
        """
        self.logger.info(f"Generating mock response for model: {model_name}")
        
        # Simulate some processing time
        await asyncio.sleep(0.3)
        
        # Generate a mock response
        mock_responses = [
            f"This is a mock response from {model_name}. Your prompt was: {prompt[:50]}...",
            f"I'm a simulated AI response from {model_name}. You asked about: {prompt[:50]}...",
            f"Mock AI {model_name} here. In response to '{prompt[:50]}...', I would say...",
        ]
        
        response = random.choice(mock_responses)
        
        return ModelResult(
            content=response,
            status=ModelStatus.SUCCESS,
            model_name=model_name,
            tokens={
                "prompt": len(prompt) // 4,
                "completion": len(response) // 4,
                "total": len(prompt) // 4 + len(response) // 4,
            },
            cost=0.0,
            latency=0.3,
            raw_response={"model": model_name, "mock": True},
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics.
        
        Returns:
            Dictionary of statistics.
        """
        return self.stats.copy()
