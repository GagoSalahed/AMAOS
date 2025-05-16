import pytest
from typing import cast
from typing import cast
from amaos.nodes.crew_manager import CrewManager
from amaos.core.node_protocol import Node, NodeTask
from amaos.nodes.llm_node import LLMNode
from amaos.nodes.tool_node import ToolNode
from amaos.models.tool import ToolResult, ToolTask
from amaos.core.node_protocol import NodeTask

def test_roster_registration_and_metadata() -> None:
    crew = CrewManager()
    llm = LLMNode()
    tool = ToolNode()
    # Register 'echo' tool for completeness
    class EchoTool:
        async def run(self, task: ToolTask) -> ToolResult:
            # Access input directly from ToolTask
            output = task.input.get('input') if isinstance(task.input, dict) and 'input' in task.input else task.input
            return ToolResult(success=True, result={"output": output}, error=None)
    tool.register_tool("echo", EchoTool())
    crew.register_agent("llm1", cast(Node, llm), role="writer", goal="draft text")
    crew.register_agent("tool1", tool, role="calculator", goal="compute")
    roster = crew.get_roster()
    assert "llm1" in roster and "tool1" in roster
    assert roster["llm1"]["role"] == "writer"
    assert roster["tool1"]["goal"] == "compute"
    crew.assign_role("llm1", "editor")
    crew.assign_goal("tool1", "summarize")
    assert crew.get_roster()["llm1"]["role"] == "editor"
    assert crew.get_roster()["tool1"]["goal"] == "summarize"
    crew.deregister_agent("llm1")
    assert "llm1" not in crew.get_roster()

@pytest.mark.asyncio
async def test_crew_manager_routing_and_stats() -> None:
    crew = CrewManager()
    llm = LLMNode()
    tool = ToolNode()
    # Register 'echo' tool
    class EchoTool:
        async def run(self, task: ToolTask) -> ToolResult:
            # Access input directly from ToolTask
            output = task.input.get('input') if isinstance(task.input, dict) and 'input' in task.input else task.input
            return ToolResult(success=True, result={"output": output}, error=None)
    tool.register_tool("echo", EchoTool())
    crew.register_agent("llm", cast(Node, llm))
    crew.register_agent("tool", tool)
    await crew.initialize()
    # Direct routing by task_type
    result = await crew.handle(NodeTask(task_type="llm", payload={"prompt": "Hello"}))
    assert result.success is True
    result2 = await crew.handle(NodeTask(task_type="tool", payload={"tool": "echo", "input": {"input": "42"}}))
    assert result2.success is True
    stats = crew.get_stats()
    assert "llm" in stats and "tool" in stats

@pytest.mark.asyncio
async def test_crew_manager_with_control_node() -> None:
    from amaos.nodes.control_node import ControlNode
    llm = LLMNode()
    tool = ToolNode()
    # Register 'echo' tool
    class EchoTool:
        async def run(self, task: ToolTask) -> ToolResult:
            # Access input directly from ToolTask
            output = task.input.get('input') if isinstance(task.input, dict) and 'input' in task.input else task.input
            return ToolResult(success=True, result={"output": output}, error=None)
    tool.register_tool("echo", EchoTool())
    control = ControlNode({"llm": cast(Node, llm), "tool": cast(Node, tool)})
    crew = CrewManager()
    crew.register_agent("llm", cast(Node, llm))
    crew.register_agent("tool", tool)
    crew.set_control_node(control)
    await crew.initialize()
    result = await crew.handle(NodeTask(task_type="llm", payload={"prompt": "Crew control!"}))
    assert result.success is True
    result2 = await crew.handle(NodeTask(task_type="tool", payload={"tool": "echo", "input": {"input": "Crew"}}))
    assert result2.success is True
