import pytest
import asyncio
from amaos.nodes.fallback_node import FallbackNode, FallbackNodeConfig
from amaos.nodes.llm_node import LLMNode
from amaos.nodes.tool_node import ToolNode
from amaos.core.node_protocol import NodeTask, NodeResult

from amaos.core.node_protocol import Node

class AlwaysFailNode(Node):
    async def initialize(self) -> None:
        pass
    async def handle(self, task: NodeTask) -> NodeResult:
        return NodeResult(success=False, result="fail", error="FAIL")
    def get_stats(self) -> dict:
        return {"failures": 1}

class AlwaysSucceedNode(Node):
    async def initialize(self) -> None:
        pass
    async def handle(self, task: NodeTask) -> NodeResult:
        return NodeResult(success=True, result="success", error=None)
    def get_stats(self) -> dict:
        return {"success": 1}

@pytest.mark.asyncio
async def test_fallback_node_primary_success() -> None:
    node = AlwaysSucceedNode()
    fb = FallbackNode([node])
    await fb.initialize()
    result = await fb.handle(NodeTask(task_type="llm", payload={"prompt": "hi"}))
    assert result.success is True
    assert result.result == "success"

@pytest.mark.asyncio
async def test_fallback_node_primary_fails_alternate_succeeds() -> None:
    fail_node = AlwaysFailNode()
    succeed_node = AlwaysSucceedNode()
    fb = FallbackNode([fail_node, succeed_node], FallbackNodeConfig(max_retries=2, max_alternate_attempts=2))
    await fb.initialize()
    result = await fb.handle(NodeTask(task_type="llm", payload={"prompt": "hi"}))
    assert result.success is True
    assert result.result == "success"

@pytest.mark.asyncio
async def test_fallback_node_all_fail() -> None:
    fail_node1 = AlwaysFailNode()
    fail_node2 = AlwaysFailNode()
    fb = FallbackNode([fail_node1, fail_node2], FallbackNodeConfig(max_retries=2, max_alternate_attempts=2))
    await fb.initialize()
    result = await fb.handle(NodeTask(task_type="llm", payload={"prompt": "hi"}))
    assert result.success is False
    assert result.error == "FALLBACK_EXHAUSTED"

@pytest.mark.asyncio
async def test_fallback_node_stats() -> None:
    fail_node = AlwaysFailNode()
    succeed_node = AlwaysSucceedNode()
    fb = FallbackNode([fail_node, succeed_node], FallbackNodeConfig(max_retries=2, max_alternate_attempts=2))
    await fb.initialize()
    await fb.handle(NodeTask(task_type="llm", payload={"prompt": "hi"}))
    stats = fb.get_stats()
    assert stats["0"]["attempts"] >= 1
    assert stats["1"]["attempts"] >= 1
