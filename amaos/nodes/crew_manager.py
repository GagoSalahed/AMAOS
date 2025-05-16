"""
CrewManager/AgentManager for AMAOS: Maintains a roster of specialized agents (nodes), supports dynamic registration, role/goal assignment, and ControlNode-based routing.
"""
from amaos.core.node_protocol import Node, NodeTask, NodeResult
from typing import Dict, Any, Optional
import asyncio

class CrewManager(Node):
    def __init__(self) -> None:
        self.agents: Dict[str, Node] = {}  # node_id -> Node
        self.metadata: Dict[str, Dict[str, Any]] = {}  # node_id -> metadata (role, goal, status, etc.)
        self.control_node: Optional[Node] = None

    def register_agent(self, node_id: str, node: Node, role: Optional[str] = None, goal: Optional[str] = None) -> None:
        self.agents[node_id] = node
        self.metadata[node_id] = {"role": role, "goal": goal, "status": "idle"}

    def deregister_agent(self, node_id: str) -> None:
        self.agents.pop(node_id, None)
        self.metadata.pop(node_id, None)

    def set_control_node(self, control_node: Node) -> None:
        self.control_node = control_node

    def assign_role(self, node_id: str, role: str) -> None:
        if node_id in self.metadata:
            self.metadata[node_id]["role"] = role

    def assign_goal(self, node_id: str, goal: str) -> None:
        if node_id in self.metadata:
            self.metadata[node_id]["goal"] = goal

    def get_roster(self) -> Dict[str, Dict[str, Any]]:
        return {nid: dict(meta) for nid, meta in self.metadata.items()}

    async def initialize(self) -> None:
        for agent in self.agents.values():
            await agent.initialize()
        if self.control_node:
            await self.control_node.initialize()

    async def handle(self, task: NodeTask) -> NodeResult:
        # Route via control_node if set, else direct to agent by task_type
        if self.control_node:
            return await self.control_node.handle(task)
        node = self.agents.get(task.task_type)
        if node:
            return await node.handle(task)
        return NodeResult(success=False, result=f"Unknown agent: {task.task_type}", error="AGENT_NOT_FOUND")

    def get_stats(self) -> Dict[str, Any]:
        return {nid: agent.get_stats() for nid, agent in self.agents.items()}
