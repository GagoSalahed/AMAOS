import pytest
from amaos.nodes.control_node import ControlNode, ChainedControlNode
from amaos.nodes.llm_node import LLMNode
from amaos.nodes.tool_node import ToolNode
from amaos.core.node_protocol import NodeTask, NodeResult
from amaos.core.node_protocol import Node
from typing import cast, Dict, Any, Union

@pytest.mark.asyncio
async def make_llm_node() -> LLMNode:
    """Create and return an LLMNode instance for testing."""
    node = LLMNode()
    return node

@pytest.mark.asyncio
async def make_tool_node() -> ToolNode:
    """Create and return a ToolNode with an echo tool registered."""
    node = ToolNode()
    # Register a simple echo tool
    from amaos.models.tool import ToolResult, ToolTask
    class EchoTool:
        async def run(self, task: ToolTask) -> ToolResult:
            # task.input is a dict, so echo task.input.get('input') if present, else the whole dict
            output = task.input.get('input') if isinstance(task.input, dict) and 'input' in task.input else task.input
            return ToolResult(success=True, result={"output": output}, error=None)
    node.register_tool("echo", EchoTool())
    return node

@pytest.mark.asyncio
async def test_control_node_routing() -> None:
    llm_node = await make_llm_node()
    tool_node = await make_tool_node()
    control = ControlNode({"llm": cast(Node, llm_node), "tool": cast(Node, tool_node)})
    await control.initialize()

    # Basic LLM routing
    result = await control.handle(NodeTask(task_type="llm", payload={"prompt": "Hello"}))
    assert result.success is True
    assert "Hello" in str(result.result)

    # Basic tool routing
    result2 = await control.handle(NodeTask(task_type="tool", payload={"tool": "echo", "input": {"input": "World"}}))
    assert result2.success is True
    assert isinstance(result2.result, dict) and result2.result.get("output") == "World"

    # Unknown node type
    result3 = await control.handle(NodeTask(task_type="unknown", payload={}))
    assert not result3.success
    assert result3.error == "NODE_NOT_FOUND"

@pytest.mark.asyncio
async def test_chained_control_node_pipeline() -> None:
    llm_node = await make_llm_node()
    tool_node = await make_tool_node()
    control = ChainedControlNode({"llm": cast(Node, llm_node), "tool": cast(Node, tool_node)})
    await control.initialize()

    pipeline = NodeTask(
        task_type="pipeline",
        payload={
            "steps": [
                {"type": "llm", "payload": {"prompt": "Chain 1"}},
                {"type": "tool", "payload": {"tool": "echo", "input": {}}}
            ]
        }
    )
    result = await control.handle(pipeline)
    assert result.success is True
    # Should echo the LLM output
    assert isinstance(result.result, dict) and "Chain 1" in str(result.result.get("output", ""))

@pytest.mark.asyncio
async def test_control_node_metadata_routing() -> None:
    llm_node = await make_llm_node()
    tool_node = await make_tool_node()
    control = ChainedControlNode({"llm": cast(Node, llm_node), "tool": cast(Node, tool_node)})
    await control.initialize()

    # Metadata for max_tokens and role
    result = await control.handle(NodeTask(
        task_type="llm",
        payload={"prompt": "Meta"},
        metadata={"max_tokens": 5, "role": "assistant"}
    ))
    assert result.success is True
    # The payload should have max_tokens and role injected (if node supports)
    assert "Meta" in str(result.result)

@pytest.mark.asyncio
async def test_control_node_retry_fallback() -> None:
    class FailingNode(Node):
        def __init__(self) -> None:
            self.calls = 0
        async def initialize(self) -> None:
            pass
        async def handle(self, task: NodeTask) -> NodeResult:
            self.calls += 1
            return NodeResult(success=False, result="fail", error="ERR")
        def get_stats(self) -> Dict[str, Any]:
            return {"calls": self.calls}
    failing = FailingNode()
    control = ControlNode({"fail": failing}, max_retries=2)
    await control.initialize()
    result = await control.handle(NodeTask(task_type="fail", payload={}))
    assert not result.success
    assert result.error == "RETRY_EXCEEDED"
    assert failing.calls == 3  # 1 initial + 2 retries
