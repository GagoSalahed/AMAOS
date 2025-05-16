from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel

class NodeTask(BaseModel):
    task_type: str
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class NodeResult(BaseModel):
    success: bool
    result: Union[Dict[str, Any], str]
    latency: Optional[float] = None
    source: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # Added for compatibility and type safety

class Node(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        ...

    @abstractmethod
    async def handle(self, task: NodeTask) -> NodeResult:
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        ...
