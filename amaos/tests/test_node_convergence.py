import pytest
from amaos.nodes.llm_node import LLMNode
from amaos.nodes.tool_node import ToolNode
from amaos.nodes.memory_node import MemoryNode
from amaos.core.node_protocol import Node, NodeTask, NodeResult
from amaos.models.tool import ToolResult, ToolTask
from typing import Dict, Any

@pytest.mark.asyncio
async def test_node_handle_unified_protocol() -> None:
    llm_node = LLMNode()
    await llm_node.initialize()

    task = NodeTask(task_type="llm", payload={"prompt": "Hello AI"})
    result = await llm_node.handle(task)

    assert result.success is True
    assert isinstance(result.result, dict) and "Hello" in result.result.get("content", "")

    tool_node = ToolNode()
    await tool_node.initialize()
    # Register a dummy tool
    class DummyTool:
        async def run(self, task: ToolTask) -> ToolResult:
            # Access input directly from ToolTask
            input_data = task.input
            output = input_data.get('input') if isinstance(input_data, dict) and 'input' in input_data else input_data
            return ToolResult(success=True, result={"output": "tool ok"}, error=None)
    tool_node.register_tool("dummy", DummyTool())
    tool_task = NodeTask(task_type="tool", payload={"tool": "dummy", "input": {}})
    tool_result = await tool_node.handle(tool_task)
    assert tool_result.success is True
    assert isinstance(tool_result.result, dict) and tool_result.result.get("output") == "tool ok"

    memory_node = MockMemoryNode()
    await memory_node.initialize()
    mem_set = NodeTask(task_type="memory", payload={"action": "set", "key": "foo", "value": "bar"})
    mem_set_result = await memory_node.handle(mem_set)
    assert mem_set_result.success is True
    mem_get = NodeTask(task_type="memory", payload={"action": "get", "key": "foo"})
    mem_get_result = await memory_node.handle(mem_get)
    assert mem_get_result.success is True
    assert isinstance(mem_get_result.result, dict) and mem_get_result.result.get("value") == "bar"

class MockMemoryNode(Node):
    def __init__(self) -> None:
        self.memory: Dict[str, Any] = {}
    async def initialize(self) -> None:
        pass
    async def handle(self, task: NodeTask) -> NodeResult:
        if task.payload.get("action") == "set":
            self.memory[task.payload["key"]] = task.payload["value"]
            # Return a dictionary for the result
            return NodeResult(success=True, result={"status": "set"}, error=None)
        elif task.payload.get("action") == "get":
            # Return a dictionary for the result, including the value
            return NodeResult(success=True, result={"value": self.memory.get(task.payload["key"])}, error=None)
        # Return a dictionary for the error case as well
        return NodeResult(success=False, result={"error": "Unknown action"}, error="Unknown action")
    def get_stats(self) -> dict:
        return {"count": len(self.memory)}
