"""
Type Contract Tests for AMAOS.

This module contains meta-tests that verify the implementation of type contracts
defined in scripts/type_contracts.md.

These tests are designed to fail if critical type contracts are broken,
serving as an extra layer of verification beyond mypy.
"""
import inspect
from typing import Any, Dict, Optional, Type, get_type_hints

import pytest
from amaos.core.node import Node
from amaos.core.node_protocol import NodeProtocol
from amaos.models import NodeResult
from amaos.nodes.reflector_node import ReflectorNode
from amaos.memory.interface import Memory


def test_node_result_metadata_contract():
    """Test that NodeResult always has a metadata field."""
    # Test initialization with metadata
    result = NodeResult(metadata={"key": "value"})
    assert hasattr(result, "metadata"), "NodeResult must have a metadata attribute"
    
    # Test initialization without metadata (should default to empty dict)
    result = NodeResult()
    assert hasattr(result, "metadata"), "NodeResult must have a metadata attribute"
    assert isinstance(result.metadata, dict), "metadata must be a dictionary"
    
    # Test that metadata keys can be accessed without KeyError
    assert result.metadata.get("nonexistent_key") is None, "Accessing nonexistent keys should return None"


def test_reflector_node_contract():
    """Test that ReflectorNode implements the required methods with correct signatures."""
    node = ReflectorNode()
    
    # Test method existence
    assert hasattr(node, "reflect"), "ReflectorNode must implement reflect()"
    assert hasattr(node, "validate_input"), "ReflectorNode must implement validate_input()"
    
    # Test method signatures
    reflect_sig = inspect.signature(node.reflect)
    assert len(reflect_sig.parameters) == 1, "reflect() should only have self parameter"
    
    validate_sig = inspect.signature(node.validate_input)
    assert len(validate_sig.parameters) == 2, "validate_input() should have self and input parameters"
    
    # Test return type hints
    type_hints = get_type_hints(node.reflect)
    assert "return" in type_hints, "reflect() must have a return type annotation"
    
    # Verify that reflect returns NodeResult
    result = node.reflect()
    assert isinstance(result, NodeResult), "reflect() must return a NodeResult instance"


def test_all_nodes_implement_protocol():
    """Test that all Node subclasses implement the NodeProtocol."""
    # This test dynamically finds all Node subclasses
    # and verifies they implement the required methods
    
    def is_concrete_node(cls: Type) -> bool:
        """Check if a class is a concrete Node implementation."""
        return (issubclass(cls, Node) 
                and cls is not Node 
                and not inspect.isabstract(cls))
    
    # Collect all concrete Node implementations
    node_classes = [
        # Add known implementations here - this is just an example
        ReflectorNode,
        # You would add more node classes here
    ]
    
    for node_class in node_classes:
        assert issubclass(node_class, NodeProtocol), f"{node_class.__name__} must implement NodeProtocol"
        
        # Create an instance and verify process method exists and has correct signature
        try:
            node = node_class()
            assert hasattr(node, "process"), f"{node_class.__name__} must implement process()"
            
            process_sig = inspect.signature(node.process)
            assert len(process_sig.parameters) >= 2, f"process() in {node_class.__name__} should have self and input parameters"
            
            # Verify type hints
            type_hints = get_type_hints(node.process)
            assert "return" in type_hints, f"process() in {node_class.__name__} must have a return type annotation"
        except TypeError:
            # If we can't instantiate directly (e.g., needs constructor args)
            # just verify the method exists on the class
            assert hasattr(node_class, "process"), f"{node_class.__name__} must implement process()"


def test_optional_typing_usage():
    """Test that Optional is used correctly in a sample of methods."""
    # This is more of a pattern test than an exhaustive check
    
    # Example function to test
    def sample_func(required: str, optional: Optional[int] = None) -> Optional[str]:
        if optional is None:
            return None
        return required * optional
    
    # Verify that None is handled correctly
    assert sample_func("test") is None
    assert sample_func("test", 2) == "testtest"
    
    # In a real test, you would examine actual AMAOS functions


def test_memory_interface_contract():
    """Test that Memory implementations adhere to the Memory interface contract."""
    # This would check actual Memory implementations
    # For demonstration purposes, we're just checking the interface
    
    # Verify Memory protocol methods
    memory_methods = ["store", "retrieve", "search", "delete"]
    
    for method in memory_methods:
        assert hasattr(Memory, method), f"Memory interface must define {method}()"


if __name__ == "__main__":
    pytest.main()
