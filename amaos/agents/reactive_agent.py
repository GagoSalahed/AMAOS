"""
ReactiveAgent for AMAOS - Agent implementing the ReAct (Reasoning + Acting) pattern.

This module provides a ReAct pattern agent that can reason through multiple steps
to solve complex tasks using a think-act-observe loop.
"""

import asyncio
import logging
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Awaitable, cast, Tuple, Protocol
from pydantic import BaseModel, Field

from amaos.agents.base_agent import BaseAgent, AgentConfig, AgentContext, AgentResult
from amaos.utils.context_logger import ContextAwareLogger
from amaos.core.node import NodeCapability
from amaos.core.node_protocol import NodeTask, NodeResult
from amaos.core.orchestrator import Orchestrator

class ReactStep(BaseModel):
    """Single step in a React agent's thought process."""
    
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class Tool(BaseModel):
    """Tool that can be used by the ReactiveAgent."""
    
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    function: Optional[Callable] = None


class ReactiveAgentConfig(AgentConfig):
    """Configuration for a ReactiveAgent."""
    
    name: str = "reactive_agent"
    max_iterations: int = 5
    prompt_template: str = (
        "You are an AI assistant solving a task step-by-step.\n"
        "You have access to the following tools:\n"
        "{tools_description}\n\n"
        "Task: {task}\n\n"
        "Think through this step-by-step. For each step, follow this format:\n"
        "Thought: Your reasoning about what to do next.\n"
        "Action: The tool to use (one of: {tool_names})\n"
        "Action Input: The input to the tool as a JSON object\n\n"
        "After I provide an observation, continue with the next step.\n"
        "Once you have enough information, use the 'finish' action to provide the final answer.\n\n"
        "{history}\n"
    )
    reflection_template: str = (
        "Review what you've learned through the previous steps:\n\n"
        "{thoughts}\n\n"
        "Based on all the above information, provide your final answer to the task: {task}"
    )
    llm_node_id: str = "llm"  # ID of the LLM node to use
    parse_regex: str = r"Action:\s*(.+?)\nAction\s*Input:\s*(.+)"


class ReactiveAgent(BaseAgent):
    """Agent implementing the ReAct pattern (Reasoning + Acting).
    
    This agent follows a think-act-observe loop to perform complex
    reasoning about tasks before producing a final answer.
    """
    
    def __init__(self, config: ReactiveAgentConfig):
        """Initialize the reactive agent.
        
        Args:
            config: Configuration for the agent
        """
        super().__init__(config)
        self.config: ReactiveAgentConfig = config  # type: ignore
        self.logger: ContextAwareLogger = ContextAwareLogger(f"ReactiveAgent:{config.name}")  # type: ignore
        self.orchestrator = Orchestrator()
        self.tools: Dict[str, Dict[str, Any]] = {}
        
        # Register built-in tools
        self._register_builtin_tools()
        
    def _register_builtin_tools(self) -> None:
        """Register built-in tools for the agent."""
        # Finish tool is always available
        self.register_tool(
            name="finish",
            description="Use this to provide your final answer",
            parameters={
                "answer": "Your final answer to the task"
            },
            function=self._finish_tool
        )
        
        # Simple mock calculator tool
        self.register_tool(
            name="calculator",
            description="Calculate a mathematical expression",
            parameters={
                "expression": "The mathematical expression to calculate (e.g., '2 + 2')"
            },
            function=self._calculator_tool
        )
        
        # Simple search tool
        self.register_tool(
            name="search",
            description="Search for information",
            parameters={
                "query": "The search query"
            },
            function=self._search_tool
        )
        
    async def initialize(self) -> None:
        """Initialize the agent."""
        await super().initialize()
        self.logger.info(f"Initializing ReactiveAgent: {self.name} ({self.id})")
        
        # Register additional capabilities
        self.register_capability(
            NodeCapability(
                name="react",
                description="Execute a task using the ReAct pattern",
                parameters={
                    "task": "Description of the task to execute",
                    "tools": "List of tools to use",
                    "max_iterations": "Maximum number of iterations"
                }
            )
        )
        
    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], 
                     function: Callable) -> None:
        """Register a tool for the agent to use.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Tool parameters
            function: Function to call when the tool is used
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function
        }
        self.logger.info(f"Registered tool: {name}")
        
    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent with react pattern.
        
        Args:
            context: Context containing the task and its inputs
            
        Returns:
            Result of the agent execution
        """
        # Extract task from context
        task = context.inputs.get("task", "")
        if not task:
            return AgentResult(
                agent_id=self.id,
                task_id=context.task_id,
                success=False,
                error="No task provided",
                outputs={}
            )
            
        # Get trace ID for context-aware logging
        trace_id = context.metadata.get("trace_id")
        if not trace_id:
            trace_id = str(uuid.uuid4())
            context.metadata["trace_id"] = trace_id
            
        logger = self.logger.with_trace(trace_id)
        logger.info(f"Executing ReactiveAgent task: {task}")
        
        # Track steps
        steps: List[ReactStep] = []
        
        # Get custom tools if provided
        custom_tools = context.inputs.get("tools", [])
        if custom_tools:
            # TODO: Implement loading custom tools
            logger.info(f"Custom tools provided: {len(custom_tools)}")
            
        # Get max iterations from context or config
        max_iterations = context.inputs.get("max_iterations", self.config.max_iterations)
        
        # Execute react loop
        for i in range(max_iterations):
            logger.info(f"Starting iteration {i+1}/{max_iterations}")
            
            # Build prompt
            prompt = self._build_prompt(task, steps)
            
            # Call LLM to get next step
            try:
                llm_result = await self._call_llm(prompt)
                
                # Parse response to extract thought, action, and action_input
                action_name, action_input = self._parse_llm_response(llm_result)
                
                # Create step record
                step = ReactStep(
                    thought="",
                    action=action_name,
                    action_input=action_input
                )
                
                logger.info(f"Action: {action_name}")
                
                # Execute action if it exists
                if action_name in self.tools:
                    try:
                        tool = self.tools[action_name]
                        # Convert the string action_input to dict if it's a string
                        input_data = action_input if isinstance(action_input, dict) else {"input": action_input}
                        result = await self._execute_tool(tool, input_data)
                        step.observation = str(result)
                        logger.info(f"Observation: {step.observation[:50]}...")
                    except Exception as e:
                        step.observation = f"Error: {str(e)}"
                        logger.error(f"Tool execution error: {e}")
                else:
                    step.observation = f"Error: Unknown action '{action_name}'. Please use one of: {', '.join(self.tools.keys())}"
                    logger.warning(f"Unknown action: {action_name}")
                    
                steps.append(step)
                
                # Check if the action was to finish
                if action_name == "finish":
                    logger.info("Finish action called, ending iterations")
                    break
                    
            except Exception as e:
                logger.error(f"Error in react loop: {e}")
                return AgentResult(
                    agent_id=self.id,
                    task_id=context.task_id,
                    success=False,
                    error=f"Error in react loop: {str(e)}",
                    outputs={}
                )
                
        # Generate final answer with reflection
        logger.info("Generating final answer with reflection")
        
        # Handle different types of action_input
        final_answer = ""
        # Check if the last step is a finish action
        # Note: mypy thinks this is unreachable in certain flows, but it's not
        if steps and steps[-1].action == "finish":  # type: ignore
            if isinstance(steps[-1].action_input, dict):
                final_answer = steps[-1].action_input.get("answer", "")
            elif isinstance(steps[-1].action_input, str):
                final_answer = steps[-1].action_input
                
        if not final_answer:
            # Generate reflection and final answer
            reflection_prompt = self.config.reflection_template.format(
                thoughts="\n".join(f"Step {i+1}:\nThought: {s.thought}\nAction: {s.action}\nAction Input: {json.dumps(s.action_input)}\nObservation: {s.observation}" 
                                 for i, s in enumerate(steps)),
                task=task
            )
            
            try:
                final_answer = await self._call_llm(reflection_prompt)
            except Exception as e:
                logger.error(f"Error generating reflection: {e}")
                final_answer = "Failed to generate reflection."
        
        # Create final result
        return AgentResult(
            agent_id=self.id,
            task_id=context.task_id,
            success=True,
            outputs={
                "answer": final_answer,
                "steps": [s.model_dump() for s in steps]
            },
            metadata={
                "iterations": len(steps),
                "trace_id": trace_id
            }
        )
        
    def _build_prompt(self, task: str, steps: List[ReactStep]) -> str:
        """Build prompt for LLM based on task and previous steps.
        
        Args:
            task: Task description
            steps: Previous steps
            
        Returns:
            Formatted prompt
        """
        # Build tool descriptions
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}" 
            for tool in self.tools.values()
        ])
        
        # Build history
        history = ""
        for i, step in enumerate(steps):
            history += f"Step {i+1}:\n"
            history += f"Thought: {step.thought}\n"
            history += f"Action: {step.action}\n"
            history += f"Action Input: {json.dumps(step.action_input, indent=2)}\n"
            
            if step.observation:
                history += f"Observation: {step.observation}\n\n"
                
        # Add prompt for next step
        if steps:
            history += "Step " + str(len(steps) + 1) + ":\n"
        
        # Format prompt
        prompt = self.config.prompt_template.format(
            tools_description=tools_description,
            tool_names=", ".join(self.tools.keys()),
            task=task,
            history=history
        )
        
        return prompt
        
    def _parse_llm_response(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Parse LLM response to extract action and input.
        
        Args:
            text: Raw LLM response text
            
        Returns:
            Tuple of (action_name, action_input)
        """                # Parse action and action input from the LLM response
        match = re.search(self.config.parse_regex, text, re.DOTALL)
        if not match:
            return ("invalid_format", {"text": text})
        
        action_name = match.group(1).strip()
        action_input_str = match.group(2).strip()
        
        # Try to parse action input as JSON, fall back to text if that fails
        try:
            action_input = json.loads(action_input_str)
            if not isinstance(action_input, dict):
                action_input = {"value": action_input}
        except Exception:
            # If JSON parsing fails, use the string directly
            action_input = {"text": action_input_str}
        
        return (action_name, action_input)
        
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM node to generate text.
        
        Args:
            prompt: Prompt to send to LLM
            
        Returns:
            Generated text
            
        Raises:
            Exception: If LLM call fails
        """
        try:
            # Create a task for the LLM node
            from amaos.core.node_protocol import NodeTask
            task = NodeTask(
                task_type=self.config.llm_node_id,
                payload={
                    "prompt": prompt
                },
                metadata={
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "trace_id": self.logger.get_trace_id()
                }
            )
            
            from amaos.core.orchestrator import Orchestrator
            orchestrator = Orchestrator()
            
            # Check if the orchestrator has the process method, if not we'll need to handle it
            # Add type ignore comment to suppress the mypy error
            if hasattr(orchestrator, 'process'):
                result = await orchestrator.process(task)  # type: ignore
            else:
                # Fall back to handle method if process is not available
                result = await orchestrator.handle(task)  # type: ignore
            
            if not result.success:
                raise Exception(f"LLM call failed: {result.error}")
                
            # Extract text from result
            result_text: str
            if isinstance(result.result, dict):
                result_text = result.result.get("text", "")
            elif isinstance(result.result, str):
                result_text = result.result
            else:
                # Convert to string with explicit annotation
                result_text = str(result.result)
                
            return result_text
                
        except ImportError:
            # For testing or when orchestrator is not available
            self.logger.warning("Orchestrator not available, using mock LLM response")
            return (
                "Thought: I need to find information about this task.\n"
                "Action: search\n"
                "Action Input: {\"query\": \"information about the task\"}"
            )
            
    async def _execute_tool(self, tool: Dict[str, Any], action_input: Dict[str, Any]) -> Any:
        """Execute a tool with the given input.
        
        Args:
            tool: Tool to execute
            action_input: Input for the tool
            
        Returns:
            Tool result
            
        Raises:
            Exception: If tool execution fails
        """
        function = tool.get("function")
        if not function:
            raise Exception(f"Tool {tool.get('name', 'unknown')} has no function")
            
        # Execute tool function
        result = function(**action_input)
        
        # Handle async functions
        if asyncio.iscoroutine(result):
            result = await result
            
        return result
        
    async def _finish_tool(self, answer: str) -> str:
        """Finish tool that returns the final answer.
        
        Args:
            answer: Final answer
            
        Returns:
            Success message
        """
        return f"Task completed with answer: {answer}"
        
    async def _calculator_tool(self, expression: str) -> str:
        """Calculator tool that evaluates mathematical expressions.
        
        Args:
            expression: Mathematical expression
            
        Returns:
            Calculated result
            
        Raises:
            Exception: If expression cannot be calculated
        """
        try:
            # Basic sanitization to prevent code execution
            if any(keyword in expression for keyword in ['__', 'import', 'eval', 'exec', 'compile']):
                raise ValueError("Potentially unsafe expression")
                
            # Use safer eval with limited scope
            result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "max": max, "min": min})
            return f"Result: {result}"
            
        except Exception as e:
            return f"Calculation error: {str(e)}"
            
    async def _search_tool(self, query: str) -> str:
        """Mock search tool that simulates information retrieval.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        # Mock search database
        search_db = {
            "weather": "The weather is currently sunny with a high of 75°F.",
            "time": f"The current time is {time.strftime('%H:%M:%S')}.",
            "date": f"Today's date is {time.strftime('%Y-%m-%d')}.",
            "amaos": "AMAOS (AI Modular Asynchronous Orchestration System) is a framework for building AI systems with modular components and asynchronous processing.",
            "population": "The world population is approximately 8 billion people.",
            "quantum computing": "Quantum computing is a type of computation that harnesses quantum mechanical phenomena.",
            "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines."
        }
        
        # Check for exact matches
        if query.lower() in search_db:
            return search_db[query.lower()]
            
        # Check for partial matches
        matches = []
        for key, value in search_db.items():
            if query.lower() in key or key in query.lower():
                matches.append(f"- {key.capitalize()}: {value}")
                
        if matches:
            return "Found the following results:\n" + "\n".join(matches)
            
        return f"No results found for query: {query}"
