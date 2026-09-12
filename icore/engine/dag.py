"""
icore.engine.dag - Directed Acyclic Graph for task dependency scheduling.

The DAG is the heart of the workflow engine. It describes:
    - Which tasks (nodes) make up a workflow
    - What order they should run in (edges = dependencies)
    - How data flows between tasks (input builders)
    - Which branches to take (conditional edges)
    - Whether a node delegates to a sub-workflow

Design:
    - DAGNode holds task metadata and an optional input_builder callable.
    - DAGEdge optionally holds a condition callable for conditional branching.
    - DAG provides topological sort, cycle detection, and validation.
    - The graph is built declaratively in BaseWorkflow.define().

Example::

    dag = DAG()
    dag.add_node("chunk", task_name="text_chunker")
    dag.add_node("summarize", task_name="summarizer",
                 input_builder=lambda params, upstream:
                     SummaryInput(chunks=upstream["chunk"].data["chunks"]))
    dag.add_node("merge", task_name="merger")

    dag.add_edge("chunk", "summarize")
    dag.add_edge("summarize", "merge")

    # Conditional branching
    dag.add_edge("classify", "route_a",
                 condition=lambda out: out.data.get("label") == "A")
    dag.add_edge("classify", "route_b",
                 condition=lambda out: out.data.get("label") == "B")

    assert dag.validate()  # no cycles, all edges reference existing nodes
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from icore.core.models import BaseTaskInput, BaseTaskOutput

logger = logging.getLogger(__name__)

# Type alias: a function that builds a task's input from workflow params
# and the outputs of upstream (predecessor) nodes.
#
#   params:            The original workflow input parameters (dict).
#   upstream_outputs:  Mapping of predecessor node_id -> BaseTaskOutput.
#
# Returns: A BaseTaskInput (or subclass) instance to pass to the task.
InputBuilder = Callable[
    [dict[str, Any], dict[str, BaseTaskOutput]], BaseTaskInput
]

# Type alias: a condition predicate for conditional edges.
# Receives the source node's output and returns True if this edge
# should be traversed (i.e., the target node should be executed).
EdgeCondition = Callable[[BaseTaskOutput], bool]


class DAGNode(BaseModel):
    """
    A node in the workflow DAG representing a single task execution.

    Attributes:
        node_id:        Unique identifier for this node within the DAG.
        task_name:      Registered name of the task to execute (TaskRegistry lookup).
        workflow_name:  If is_subworkflow is True, the registered workflow name
                        to invoke instead of a task.
        is_subworkflow: Whether this node delegates to a sub-workflow.
        input_builder:  Optional callable to construct the task's input from
                        workflow params and upstream outputs. If None, the
                        executor merges upstream data + params automatically.
        model_id:       Optional model override for this specific node.
                        None means use the workflow-level model_id.
        retries:        Number of retry attempts on failure (default 0).
        timeout:        Optional per-node timeout in seconds.
        metadata:       Free-form metadata for extensibility.
        is_agent:       Whether this node is an Agent node (v0.6).
        agent_config:   AgentConfig instance when is_agent=True (v0.6).
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    node_id: str = Field(description="Unique node identifier within the DAG")
    task_name: str = Field(default="", description="Registered task name")
    workflow_name: str = Field(
        default="",
        description="Registered workflow name (used when is_subworkflow=True)",
    )
    is_subworkflow: bool = Field(
        default=False,
        description="Whether this node invokes a sub-workflow",
    )
    input_builder: Optional[InputBuilder] = Field(
        default=None,
        description="Callable to build task input from params + upstream outputs",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional model override for this node",
    )
    retries: int = Field(default=0, ge=0, description="Retry attempts on failure")
    timeout: Optional[float] = Field(
        default=None,
        description="Per-node timeout in seconds (None = no timeout)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form node metadata",
    )
    # v0.6: Agent 节点显式字段（AGENTS.md §10.11）。
    # is_agent / agent_config 必须作为直接关键字参数传递，不能塞进 metadata dict。
    # agent_config 实际类型为 AgentConfig | None，用 Any 避免 import 循环
    # （icore.engine.agent 依赖 icore.engine.dag，反向 import 会循环）。
    is_agent: bool = Field(
        default=False,
        description="Whether this node is an Agent node (v0.6)",
    )
    agent_config: Any = Field(
        default=None,
        description="AgentConfig instance when is_agent=True (v0.6)",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate node configuration after initialization."""
        if self.is_subworkflow:
            if not self.workflow_name:
                raise ValueError(
                    f"Node '{self.node_id}' is_subworkflow=True but "
                    f"workflow_name is not set"
                )
        else:
            if not self.task_name:
                raise ValueError(
                    f"Node '{self.node_id}' has no task_name "
                    f"(set task_name or use is_subworkflow=True)"
                )


class DAGEdge(BaseModel):
    """
    A directed edge in the workflow DAG representing a dependency.

    Edges define execution order: the target node can only start after
    the source node completes. An optional condition predicate enables
    conditional branching -- the target is only executed if the
    condition evaluates to True against the source's output.

    Attributes:
        source:               Source node ID (predecessor).
        target:               Target node ID (successor).
        condition:            Optional predicate for conditional branching.
                              None means unconditional (always traverse).
        condition_description: Human-readable description of the condition
                              (for documentation and debugging).
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    source: str = Field(description="Source (predecessor) node ID")
    target: str = Field(description="Target (successor) node ID")
    condition: Optional[EdgeCondition] = Field(
        default=None,
        description="Optional condition predicate for conditional branching",
    )
    condition_description: str = Field(
        default="",
        description="Human-readable description of the condition",
    )


class DAGValidationError(Exception):
    """Raised when the DAG fails validation (cycles, missing nodes, etc.)."""


class DAG:
    """
    Directed Acyclic Graph for workflow task scheduling.

    The DAG is built declaratively by BaseWorkflow.define(). The
    WorkflowExecutor uses it to determine execution order, parallelize
    independent tasks, and pass data between tasks.

    Key operations:
        - add_node / remove_node: Manage task nodes
        - add_edge / remove_edge: Manage dependencies
        - topological_sort:       Get execution order (Kahn's algorithm)
        - detect_cycle:            Check for cycles (invalid DAG)
        - get_start_nodes:         Nodes with no predecessors (entry points)
        - get_terminal_nodes:      Nodes with no successors (exit points)
        - validate:                Full structural validation
    """

    def __init__(self) -> None:
        """Initialize an empty DAG."""
        self._nodes: dict[str, DAGNode] = {}
        self._edges: list[DAGEdge] = []
        # Adjacency: node_id -> list of successor node_ids
        self._successors: dict[str, list[str]] = {}
        # Reverse adjacency: node_id -> list of predecessor node_ids
        self._predecessors: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        task_name: str = "",
        workflow_name: str = "",
        is_subworkflow: bool = False,
        input_builder: InputBuilder | None = None,
        model_id: str | None = None,
        retries: int = 0,
        timeout: float | None = None,
        is_agent: bool = False,
        agent_config: Any = None,
        **metadata: Any,
    ) -> DAGNode:
        """
        Add a node to the DAG.

        For a regular task node, provide ``task_name``.
        For a sub-workflow node, set ``is_subworkflow=True`` and provide
        ``workflow_name``.

        Args:
            node_id:        Unique identifier for this node.
            task_name:      Registered task name (for regular task nodes).
            workflow_name:  Registered workflow name (for sub-workflow nodes).
            is_subworkflow: Whether this node invokes a sub-workflow.
            input_builder:  Callable to build task input from params + upstream.
            model_id:       Optional model override for this node.
            retries:        Retry attempts on failure.
            timeout:        Per-node timeout in seconds.
            is_agent:       Whether this node is an Agent node (v0.6).
            agent_config:   AgentConfig instance when is_agent=True (v0.6).
            **metadata:     Additional metadata stored on the node.

        Returns:
            The created DAGNode instance.

        Raises:
            ValueError: If node_id already exists, or required fields missing.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists in the DAG")

        # Backward compat: extract is_agent / agent_config from the metadata
        # dict if the caller passed them via metadata={"is_agent": True, ...}
        # instead of as explicit keyword arguments (AGENTS.md §10.11).
        # Explicit keyword arguments take precedence.
        if "is_agent" in metadata and not is_agent:
            is_agent = bool(metadata["is_agent"])
        if "agent_config" in metadata and agent_config is None:
            agent_config = metadata["agent_config"]

        # Mirror is_agent / agent_config into the metadata dict so that
        # existing code reading node.metadata["is_agent"] still works
        # (AGENTS.md §10.11 backward compatibility).
        if is_agent:
            metadata["is_agent"] = is_agent
        if agent_config is not None:
            metadata["agent_config"] = agent_config

        node = DAGNode(
            node_id=node_id,
            task_name=task_name,
            workflow_name=workflow_name,
            is_subworkflow=is_subworkflow,
            input_builder=input_builder,
            model_id=model_id,
            retries=retries,
            timeout=timeout,
            is_agent=is_agent,
            agent_config=agent_config,
            metadata=metadata,
        )
        self._nodes[node_id] = node
        self._successors[node_id] = []
        self._predecessors[node_id] = []
        return node

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all its edges from the DAG.

        Args:
            node_id: The node to remove.

        Raises:
            KeyError: If the node does not exist.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' does not exist in the DAG")

        # Remove edges involving this node
        self._edges = [
            e for e in self._edges if e.source != node_id and e.target != node_id
        ]

        # Update adjacency lists
        for succ in self._successors.get(node_id, []):
            self._predecessors[succ] = [
                p for p in self._predecessors.get(succ, []) if p != node_id
            ]
        for pred in self._predecessors.get(node_id, []):
            self._successors[pred] = [
                s for s in self._successors.get(pred, []) if s != node_id
            ]

        del self._nodes[node_id]
        self._successors.pop(node_id, None)
        self._predecessors.pop(node_id, None)

    def get_node(self, node_id: str) -> DAGNode:
        """Get a node by ID. Raises KeyError if not found."""
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' does not exist in the DAG")
        return self._nodes[node_id]

    @property
    def nodes(self) -> dict[str, DAGNode]:
        """All nodes in the DAG (node_id -> DAGNode)."""
        return dict(self._nodes)

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        condition: EdgeCondition | None = None,
        condition_description: str = "",
    ) -> DAGEdge:
        """
        Add a directed edge (dependency) from source to target.

        The target node can only execute after the source node completes.
        If a condition is provided, the target only executes when the
        condition evaluates to True against the source's output.

        Conditional skip semantics (v0.6.x, ICORE-ISSUE-002 —— 支持
        菱形分支汇合）::

                       ┌── cond A ──→ node_a ──┐
            node_start ┤                        ├──→ join ──→ end
                       └── cond B ──→ node_b ──┘

        - A node runs when at least one incoming edge is "active"
          (predecessor completed AND edge unconditional / condition True).
        - A condition-skipped sibling does **not** force-skip a fan-in
          join — the join runs with the executed branch's output only.
        - A node is skipped only when NO incoming edge is active
          (AND-join truncation; linear chains keep the legacy cascade).
        - A FAILED predecessor still force-skips its downstream cone.

        Args:
            source:               Source (predecessor) node ID.
            target:               Target (successor) node ID.
            condition:            Optional predicate for conditional branching.
            condition_description: Human-readable description of the condition.

        Returns:
            The created DAGEdge instance.

        Raises:
            KeyError:  If source or target node does not exist.
            ValueError: If source == target (self-loop).
        """
        if source not in self._nodes:
            raise KeyError(f"Source node '{source}' does not exist")
        if target not in self._nodes:
            raise KeyError(f"Target node '{target}' does not exist")
        if source == target:
            raise ValueError(f"Self-loop not allowed: {source} -> {target}")

        edge = DAGEdge(
            source=source,
            target=target,
            condition=condition,
            condition_description=condition_description,
        )
        self._edges.append(edge)
        self._successors[source].append(target)
        self._predecessors[target].append(source)
        return edge

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from source to target."""
        self._edges = [
            e
            for e in self._edges
            if not (e.source == source and e.target == target)
        ]
        if target in self._successors.get(source, []):
            self._successors[source].remove(target)
        if source in self._predecessors.get(target, []):
            self._predecessors[target].remove(source)

    @property
    def edges(self) -> list[DAGEdge]:
        """All edges in the DAG."""
        return list(self._edges)

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def get_predecessors(self, node_id: str) -> list[str]:
        """Get the predecessor node IDs of a node (dependencies)."""
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' does not exist")
        return list(self._predecessors.get(node_id, []))

    def get_dependencies(self, node_id: str) -> list[str]:
        """
        Alias for get_predecessors().

        Returns the list of node IDs that this node depends on
        (must complete before this node can execute).
        """
        return self.get_predecessors(node_id)

    def get_successors(self, node_id: str) -> list[str]:
        """Get the successor node IDs of a node (dependents)."""
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' does not exist")
        return list(self._successors.get(node_id, []))

    def get_dependents(self, node_id: str) -> list[str]:
        """
        Alias for get_successors().

        Returns the list of node IDs that depend on this node
        (can only execute after this node completes).
        """
        return self.get_successors(node_id)

    def get_edges_from(self, node_id: str) -> list[DAGEdge]:
        """Get all outgoing edges from a node."""
        return [e for e in self._edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> list[DAGEdge]:
        """Get all incoming edges to a node."""
        return [e for e in self._edges if e.target == node_id]

    def get_start_nodes(self) -> list[str]:
        """Get nodes with no predecessors (entry points of the DAG)."""
        return [
            node_id
            for node_id in self._nodes
            if not self._predecessors.get(node_id)
        ]

    def get_terminal_nodes(self) -> list[str]:
        """Get nodes with no successors (exit points of the DAG)."""
        return [
            node_id
            for node_id in self._nodes
            if not self._successors.get(node_id)
        ]

    def in_degree(self, node_id: str) -> int:
        """Number of incoming edges (predecessors count)."""
        return len(self._predecessors.get(node_id, []))

    def out_degree(self, node_id: str) -> int:
        """Number of outgoing edges (successors count)."""
        return len(self._successors.get(node_id, []))

    # ------------------------------------------------------------------
    # Topological sort & cycle detection
    # ------------------------------------------------------------------

    def detect_cycle(self) -> bool:
        """
        Detect whether the DAG contains a cycle.

        Uses Kahn's algorithm: repeatedly remove nodes with in-degree 0.
        If not all nodes are removed, a cycle exists.

        Returns:
            True if a cycle is detected, False otherwise.
        """
        in_degrees = {nid: len(self._predecessors.get(nid, [])) for nid in self._nodes}
        queue = [nid for nid, deg in in_degrees.items() if deg == 0]
        processed = 0

        while queue:
            node = queue.pop(0)
            processed += 1
            for succ in self._successors.get(node, []):
                in_degrees[succ] -= 1
                if in_degrees[succ] == 0:
                    queue.append(succ)

        return processed != len(self._nodes)

    def topological_sort(self) -> list[str]:
        """
        Return a topological ordering of node IDs.

        Uses Kahn's algorithm. Nodes at the same "level" (same wave of
        in-degree reduction) are returned in insertion order.

        Returns:
            List of node IDs in topological order.

        Raises:
            DAGValidationError: If the graph contains a cycle.
        """
        if self.detect_cycle():
            raise DAGValidationError(
                "Cannot topologically sort: the DAG contains a cycle"
            )

        in_degrees = {
            nid: len(self._predecessors.get(nid, [])) for nid in self._nodes
        }
        queue = [nid for nid in self._nodes if in_degrees[nid] == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for succ in self._successors.get(node, []):
                in_degrees[succ] -= 1
                if in_degrees[succ] == 0:
                    queue.append(succ)

        return result

    def get_execution_waves(self) -> list[list[str]]:
        """
        Group nodes into execution waves for parallel execution.

        Each wave contains nodes that can run concurrently (all their
        predecessors are in earlier waves). Wave 0 contains start nodes.

        Returns:
            List of waves, each wave is a list of node IDs.

        Raises:
            DAGValidationError: If the graph contains a cycle.
        """
        if self.detect_cycle():
            raise DAGValidationError(
                "Cannot compute execution waves: the DAG contains a cycle"
            )

        in_degrees = {
            nid: len(self._predecessors.get(nid, [])) for nid in self._nodes
        }
        waves: list[list[str]] = []
        current_wave = [nid for nid in self._nodes if in_degrees[nid] == 0]

        while current_wave:
            waves.append(current_wave)
            next_wave: list[str] = []
            for node in current_wave:
                for succ in self._successors.get(node, []):
                    in_degrees[succ] -= 1
                    if in_degrees[succ] == 0:
                        next_wave.append(succ)
            current_wave = next_wave

        return waves

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """
        Validate the DAG structure.

        Checks:
            1. No cycles (it is a DAG, not a general graph).
            2. All edge endpoints reference existing nodes.
            3. At least one node exists.
            4. All nodes are reachable from at least one start node.

        Returns:
            True if the DAG is valid.

        Raises:
            DAGValidationError: If validation fails (with a descriptive message).
        """
        if not self._nodes:
            raise DAGValidationError("DAG has no nodes")

        # Check for cycles
        if self.detect_cycle():
            # Find the cycle for a helpful error message
            cycle = self._find_cycle()
            cycle_str = " -> ".join(cycle) if cycle else "(unknown)"
            raise DAGValidationError(f"DAG contains a cycle: {cycle_str}")

        # All edges should reference existing nodes (enforced by add_edge,
        # but check anyway in case of direct manipulation)
        for edge in self._edges:
            if edge.source not in self._nodes:
                raise DAGValidationError(
                    f"Edge references unknown source node: {edge.source}"
                )
            if edge.target not in self._nodes:
                raise DAGValidationError(
                    f"Edge references unknown target node: {edge.target}"
                )

        return True

    def _find_cycle(self) -> list[str] | None:
        """
        Find a cycle in the graph using DFS.

        Returns:
            List of node IDs forming a cycle, or None if no cycle.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {nid: WHITE for nid in self._nodes}
        parent: dict[str, str | None] = {nid: None for nid in self._nodes}

        def dfs(node: str) -> list[str] | None:
            color[node] = GRAY
            for succ in self._successors.get(node, []):
                if color[succ] == GRAY:
                    # Found a back edge -> cycle
                    cycle: list[str] = [succ, node]
                    curr = node
                    while curr != succ and curr is not None:
                        curr = parent[curr]
                        if curr is not None:
                            cycle.append(curr)
                    cycle.reverse()
                    return cycle
                if color[succ] == WHITE:
                    parent[succ] = node
                    result = dfs(succ)
                    if result is not None:
                        return result
            color[node] = BLACK
            return None

        for nid in self._nodes:
            if color[nid] == WHITE:
                result = dfs(nid)
                if result is not None:
                    return result
        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Number of nodes in the DAG."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the DAG."""
        return len(self._edges)

    def is_empty(self) -> bool:
        """True if the DAG has no nodes."""
        return len(self._nodes) == 0

    def __repr__(self) -> str:
        """Concise representation for logging."""
        return f"DAG(nodes={self.node_count}, edges={self.edge_count})"
