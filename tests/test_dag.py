"""
Tests for icore.engine.dag - DAG construction, validation, and scheduling.

Covers:
    - Node/edge add & remove
    - Cycle detection (Kahn's algorithm)
    - Topological sort
    - Execution wave computation (parallel scheduling)
    - Conditional edges (condition predicate storage)
    - DAGValidationError on invalid structures
    - Sub-workflow node configuration
"""

from __future__ import annotations

import pytest

from icore.engine.dag import DAG, DAGValidationError, DAGEdge, DAGNode


# ---------------------------------------------------------------------------
# Node management
# ---------------------------------------------------------------------------

class TestDAGNodes:
    def test_add_node_basic(self):
        dag = DAG()
        node = dag.add_node("a", task_name="task_a")
        assert isinstance(node, DAGNode)
        assert node.node_id == "a"
        assert node.task_name == "task_a"
        assert dag.node_count == 1

    def test_add_node_duplicate_raises(self):
        dag = DAG()
        dag.add_node("a", task_name="task_a")
        with pytest.raises(ValueError, match="already exists"):
            dag.add_node("a", task_name="task_b")

    def test_add_node_requires_task_name_or_subworkflow(self):
        dag = DAG()
        with pytest.raises(ValueError, match="no task_name"):
            dag.add_node("empty")

    def test_add_subworkflow_node_requires_workflow_name(self):
        dag = DAG()
        with pytest.raises(ValueError, match="workflow_name is not set"):
            dag.add_node("sub", is_subworkflow=True)

    def test_add_subworkflow_node_success(self):
        dag = DAG()
        node = dag.add_node(
            "sub", is_subworkflow=True, workflow_name="child_wf"
        )
        assert node.is_subworkflow is True
        assert node.workflow_name == "child_wf"

    def test_remove_node(self):
        dag = DAG()
        dag.add_node("a", task_name="t_a")
        dag.add_node("b", task_name="t_b")
        dag.add_edge("a", "b")
        dag.remove_node("a")
        assert "a" not in dag.nodes
        assert dag.node_count == 1
        # Edges involving removed node should be gone
        assert dag.edge_count == 0

    def test_remove_nonexistent_node_raises(self):
        dag = DAG()
        with pytest.raises(KeyError):
            dag.remove_node("ghost")

    def test_get_node(self):
        dag = DAG()
        dag.add_node("a", task_name="t_a")
        node = dag.get_node("a")
        assert node.task_name == "t_a"
        with pytest.raises(KeyError):
            dag.get_node("ghost")


# ---------------------------------------------------------------------------
# Edge management
# ---------------------------------------------------------------------------

class TestDAGEdges:
    def test_add_edge_basic(self):
        dag = DAG()
        dag.add_node("a", task_name="t_a")
        dag.add_node("b", task_name="t_b")
        edge = dag.add_edge("a", "b")
        assert isinstance(edge, DAGEdge)
        assert edge.source == "a"
        assert edge.target == "b"
        assert dag.edge_count == 1

    def test_add_edge_unknown_source_raises(self):
        dag = DAG()
        dag.add_node("b", task_name="t_b")
        with pytest.raises(KeyError, match="Source node"):
            dag.add_edge("ghost", "b")

    def test_add_edge_unknown_target_raises(self):
        dag = DAG()
        dag.add_node("a", task_name="t_a")
        with pytest.raises(KeyError, match="Target node"):
            dag.add_edge("a", "ghost")

    def test_self_loop_raises(self):
        dag = DAG()
        dag.add_node("a", task_name="t_a")
        with pytest.raises(ValueError, match="Self-loop"):
            dag.add_edge("a", "a")

    def test_add_conditional_edge(self):
        dag = DAG()
        dag.add_node("a", task_name="t_a")
        dag.add_node("b", task_name="t_b")
        cond = lambda out: out.data.get("ok") is True  # noqa: E731
        edge = dag.add_edge("a", "b", condition=cond)
        assert edge.condition is cond
        assert edge.condition_description == ""

    def test_remove_edge(self):
        dag = DAG()
        dag.add_node("a", task_name="t_a")
        dag.add_node("b", task_name="t_b")
        dag.add_edge("a", "b")
        dag.remove_edge("a", "b")
        assert dag.edge_count == 0
        # After removal, b has no predecessor
        assert dag.get_predecessors("b") == []


# ---------------------------------------------------------------------------
# Graph queries
# ---------------------------------------------------------------------------

class TestDAGQueries:
    def _build_diamond(self):
        """Build a diamond DAG: a -> b, a -> c, b -> d, c -> d."""
        dag = DAG()
        for nid in ("a", "b", "c", "d"):
            dag.add_node(nid, task_name=f"t_{nid}")
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        dag.add_edge("b", "d")
        dag.add_edge("c", "d")
        return dag

    def test_get_predecessors(self):
        dag = self._build_diamond()
        assert set(dag.get_predecessors("d")) == {"b", "c"}
        assert dag.get_predecessors("a") == []

    def test_get_successors(self):
        dag = self._build_diamond()
        assert set(dag.get_successors("a")) == {"b", "c"}
        assert dag.get_successors("d") == []

    def test_get_start_nodes(self):
        dag = self._build_diamond()
        assert dag.get_start_nodes() == ["a"]

    def test_get_terminal_nodes(self):
        dag = self._build_diamond()
        assert dag.get_terminal_nodes() == ["d"]

    def test_in_out_degree(self):
        dag = self._build_diamond()
        assert dag.in_degree("a") == 0
        assert dag.out_degree("a") == 2
        assert dag.in_degree("d") == 2
        assert dag.out_degree("d") == 0

    def test_get_edges_from_to(self):
        dag = self._build_diamond()
        from_a = dag.get_edges_from("a")
        assert len(from_a) == 2
        to_d = dag.get_edges_to("d")
        assert len(to_d) == 2


# ---------------------------------------------------------------------------
# Topological sort & cycle detection
# ---------------------------------------------------------------------------

class TestDAGTopology:
    def test_topological_sort_linear(self):
        dag = DAG()
        for nid in ("a", "b", "c"):
            dag.add_node(nid, task_name=f"t_{nid}")
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        order = dag.topological_sort()
        assert order == ["a", "b", "c"]

    def test_topological_sort_diamond(self):
        dag = DAG()
        for nid in ("a", "b", "c", "d"):
            dag.add_node(nid, task_name=f"t_{nid}")
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        dag.add_edge("b", "d")
        dag.add_edge("c", "d")
        order = dag.topological_sort()
        # a must come first, d must come last; b and c in between
        assert order[0] == "a"
        assert order[-1] == "d"
        assert set(order[1:3]) == {"b", "c"}

    def test_detect_cycle_no_cycle(self):
        dag = DAG()
        for nid in ("a", "b", "c"):
            dag.add_node(nid, task_name=f"t_{nid}")
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        assert dag.detect_cycle() is False

    def test_detect_cycle_simple(self):
        dag = DAG()
        for nid in ("a", "b", "c"):
            dag.add_node(nid, task_name=f"t_{nid}")
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "a")
        assert dag.detect_cycle() is True

    def test_topological_sort_with_cycle_raises(self):
        dag = DAG()
        for nid in ("a", "b"):
            dag.add_node(nid, task_name=f"t_{nid}")
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(DAGValidationError, match="cycle"):
            dag.topological_sort()

    def test_execution_waves_linear(self):
        dag = DAG()
        for nid in ("a", "b", "c"):
            dag.add_node(nid, task_name=f"t_{nid}")
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        waves = dag.get_execution_waves()
        assert waves == [["a"], ["b"], ["c"]]

    def test_execution_waves_diamond(self):
        dag = DAG()
        for nid in ("a", "b", "c", "d"):
            dag.add_node(nid, task_name=f"t_{nid}")
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        dag.add_edge("b", "d")
        dag.add_edge("c", "d")
        waves = dag.get_execution_waves()
        assert waves[0] == ["a"]
        assert set(waves[1]) == {"b", "c"}
        assert waves[2] == ["d"]

    def test_execution_waves_with_cycle_raises(self):
        dag = DAG()
        for nid in ("a", "b"):
            dag.add_node(nid, task_name=f"t_{nid}")
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(DAGValidationError):
            dag.get_execution_waves()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestDAGValidation:
    def test_empty_dag_raises(self):
        dag = DAG()
        with pytest.raises(DAGValidationError, match="no nodes"):
            dag.validate()

    def test_valid_dag_passes(self):
        dag = DAG()
        dag.add_node("a", task_name="t_a")
        dag.add_node("b", task_name="t_b")
        dag.add_edge("a", "b")
        assert dag.validate() is True

    def test_cyclic_dag_fails_with_path(self):
        dag = DAG()
        for nid in ("a", "b", "c"):
            dag.add_node(nid, task_name=f"t_{nid}")
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "a")
        with pytest.raises(DAGValidationError, match="cycle"):
            dag.validate()

    def test_is_empty(self):
        dag = DAG()
        assert dag.is_empty() is True
        dag.add_node("a", task_name="t_a")
        assert dag.is_empty() is False
