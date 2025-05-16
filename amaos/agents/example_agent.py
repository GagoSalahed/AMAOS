"""Example agent implementation for AMAOS.

This module provides an example agent implementation that extends the base agent.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, cast

from amaos.agents.base_agent import AgentCapability, AgentConfig, AgentContext, AgentResult, BaseAgent
from amaos.core.llm_manager import LLMManager, LLMManagerConfig


class ExampleAgentConfig(AgentConfig):
    """Configuration for the example agent."""

    prompt_template: str = "You are an AI assistant. Answer the following question: {question}"
    llm_config: Dict[str, Any] = {}


class ExampleAgent(BaseAgent):
    """Example agent implementation.
    
    This agent demonstrates how to create a simple agent that uses the LLM Manager
    to process requests.
    """

    def __init__(self, config: ExampleAgentConfig) -> None:
        """Initialize the example agent.
        
        Args:
            config: Configuration for the agent.
        """
        super().__init__(config)
        self.prompt_template = config.prompt_template
        
        # Initialize LLM Manager
        llm_config = LLMManagerConfig(**config.llm_config)
        self.llm_manager = LLMManager(llm_config)

    async def initialize(self) -> None:
        """Initialize the agent."""
        await super().initialize()
        
        # Register additional capabilities
        self.register_capability(
            AgentCapability(
                name="answer_question",
                description="Answer a question using the LLM",
                parameters={
                    "question": "The question to answer",
                },
            )
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute a task.
        
        Args:
            context: Context for the task execution.
            
        Returns:
            Result of the task execution.
        """
        self.logger.info(f"Executing task for example agent: {self.name} ({self.id})")
        
        # Check if the task has a question
        if "question" not in context.inputs:
            return AgentResult(
                agent_id=self.id,
                task_id=context.task_id,
                success=False,
                error="No question provided in inputs",
            )
            
        question = context.inputs["question"]
        
        try:
            # Format the prompt
            prompt = self.prompt_template.format(question=question)
            
            # Get the answer from the LLM
            llm_response = await self.llm_manager.complete(prompt)
            
            # Return the result
            return AgentResult(
                agent_id=self.id,
                task_id=context.task_id,
                success=True,
                outputs={
                    "answer": llm_response.content,
                    "model_used": llm_response.model_name,
                },
                metadata={
                    "tokens": llm_response.tokens,
                    "cost": llm_response.cost,
                    "latency": llm_response.latency,
                },
            )
            
        except Exception as e:
            self.logger.error(f"Error executing task: {e}")
            
            return AgentResult(
                agent_id=self.id,
                task_id=context.task_id,
                success=False,
                error=str(e),
                metadata={"error_type": type(e).__name__},
            )

    async def answer_question(self, question: str) -> str:
        """Answer a question using the LLM.
        
        This is a convenience method that wraps the execute method.
        
        Args:
            question: The question to answer.
            
        Returns:
            The answer to the question.
        """
        context = AgentContext(
            agent_id=self.id,
            task_id=None,
            inputs={"question": question},
        )
        
        result = await self.execute(context)
        
        if not result.success:
            raise Exception(f"Failed to answer question: {result.error}")
            
        return cast(str, result.outputs.get("answer", ""))
