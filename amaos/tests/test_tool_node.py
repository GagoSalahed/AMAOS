import pytest
import pytest_asyncio
from amaos.nodes.tool_node import ToolNode
from amaos.plugins.base_plugin import BasePlugin
from amaos.models.tool import ToolTask, ToolResult
from amaos.core.plugin_interfaces import PluginMetadata

def make_metadata(name: str) -> PluginMetadata:
    return PluginMetadata(
        name=name,
        version="0.1",
        description=f"Test tool {name}",
        author="test"
    )

class SuccessfulTool(BasePlugin):
    async def run(self, task: ToolTask) -> ToolResult:
        return ToolResult(success=True, result={"output": "it worked"})

class FailingTool(BasePlugin):
    async def run(self, task: ToolTask) -> ToolResult:
        raise RuntimeError("Tool crashed")

class FlakyTool(BasePlugin):
    def __init__(self, metadata: PluginMetadata) -> None:
        super().__init__(metadata)
        self.counter = 0

    async def run(self, task: ToolTask) -> ToolResult:
        self.counter += 1
        if self.counter < 2:
            raise RuntimeError("fail once")
        return ToolResult(success=True, result={"output": "retry success"})

@pytest_asyncio.fixture
async def tool_node() -> ToolNode:
    node = ToolNode()
    await node.initialize()
    return node

@pytest.mark.asyncio
async def test_tool_execution_success(tool_node: ToolNode) -> None:
    tool_node.register_tool("echo", SuccessfulTool(metadata=make_metadata("echo")))
    task = ToolTask(tool="echo", input={"msg": "hi"})
    result = await tool_node.execute(task)
    assert result.success is True
    assert result.result["output"] == "it worked"

@pytest.mark.asyncio
async def test_tool_execution_failure(tool_node: ToolNode) -> None:
    tool_node.register_tool("fail", FailingTool(metadata=make_metadata("fail")))
    task = ToolTask(tool="fail", input={})
    result = await tool_node.execute(task)
    assert result.success is False
    assert "error" in result.result

@pytest.mark.asyncio
async def test_tool_not_found(tool_node: ToolNode) -> None:
    task = ToolTask(tool="unknown", input={})
    result = await tool_node.execute(task)
    assert result.success is False
    assert "not found" in result.result.get("error", "").lower()

@pytest.mark.asyncio
async def test_tool_retry_logic(tool_node: ToolNode) -> None:
    flaky = FlakyTool(metadata=make_metadata("flaky"))
    tool_node.register_tool("flaky", flaky)
    task = ToolTask(tool="flaky", input={})
    result = await tool_node.execute(task)
    assert result.success is True
    assert result.result["output"] == "retry success"

@pytest.mark.asyncio
async def test_tool_metrics_tracking(tool_node: ToolNode) -> None:
    tool_node.register_tool("echo", SuccessfulTool(metadata=make_metadata("echo")))
    await tool_node.execute(ToolTask(tool="echo", input={}))
    await tool_node.execute(ToolTask(tool="echo", input={}))
    stats = tool_node.get_stats()
    assert stats["echo"]["calls"] == 2
    assert stats["echo"]["successes"] == 2

def test_roster_registration_and_metadata() -> None:
    pass
