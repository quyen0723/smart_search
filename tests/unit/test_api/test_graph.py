"""Tests for graph API endpoints."""

import pytest
from fastapi.testclient import TestClient

from smart_search.api import create_test_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_test_app()
    return TestClient(app)


class TestStatsEndpoint:
    """Tests for /graph/stats endpoint."""

    def test_get_stats(self, client):
        """Test get graph stats."""
        response = client.get("/api/v1/graph/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_nodes" in data
        assert "total_edges" in data
        assert "node_types" in data
        assert "edge_types" in data


class TestNodeEndpoint:
    """Tests for /graph/node endpoint."""

    def test_get_node(self, client):
        """Test get node by ID."""
        response = client.get("/api/v1/graph/node/test_id")
        assert response.status_code == 404  # Mock has no nodes

    def test_get_node_not_found(self, client):
        """Test get non-existent node."""
        response = client.get("/api/v1/graph/node/nonexistent")
        assert response.status_code == 404


class TestNodesEndpoint:
    """Tests for /graph/nodes endpoint."""

    def test_list_nodes(self, client):
        """Test list nodes."""
        response = client.get("/api/v1/graph/nodes")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_nodes_with_filter(self, client):
        """Test list nodes with filter."""
        response = client.get(
            "/api/v1/graph/nodes",
            params={"code_type": "function", "limit": 10},
        )
        assert response.status_code == 200

    def test_list_nodes_by_file(self, client):
        """Test list nodes by file."""
        response = client.get(
            "/api/v1/graph/nodes",
            params={"file_path": "/test.py"},
        )
        assert response.status_code == 200


class TestEdgesEndpoint:
    """Tests for /graph/edges endpoint."""

    def test_list_edges(self, client):
        """Test list edges."""
        response = client.get("/api/v1/graph/edges")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_edges_by_source(self, client):
        """Test edges by source."""
        response = client.get(
            "/api/v1/graph/edges",
            params={"source_id": "test_id"},
        )
        assert response.status_code == 200

    def test_list_edges_by_target(self, client):
        """Test edges by target."""
        response = client.get(
            "/api/v1/graph/edges",
            params={"target_id": "test_id"},
        )
        assert response.status_code == 200


class TestSubgraphEndpoint:
    """Tests for /graph/subgraph endpoint."""

    def test_get_subgraph(self, client):
        """Test get subgraph."""
        response = client.get("/api/v1/graph/subgraph/test_id")
        assert response.status_code == 404  # Mock has no nodes

    def test_get_subgraph_with_options(self, client):
        """Test subgraph with options."""
        response = client.get(
            "/api/v1/graph/subgraph/test_id",
            params={"depth": 3, "direction": "out", "max_nodes": 100},
        )
        assert response.status_code == 404


class TestPathEndpoint:
    """Tests for /graph/path endpoint."""

    def test_find_path(self, client):
        """Test find path."""
        response = client.get(
            "/api/v1/graph/path",
            params={"source": "id1", "target": "id2"},
        )
        # Mock has no nodes
        assert response.status_code == 404

    def test_find_path_with_depth(self, client):
        """Test find path with depth."""
        response = client.get(
            "/api/v1/graph/path",
            params={"source": "id1", "target": "id2", "max_depth": 3},
        )
        assert response.status_code == 404


class TestCommunitiesEndpoint:
    """Tests for /graph/communities endpoint."""

    def test_list_communities(self, client):
        """Test list communities."""
        response = client.get("/api/v1/graph/communities")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_communities_with_limit(self, client):
        """Test communities with limit."""
        response = client.get(
            "/api/v1/graph/communities",
            params={"limit": 5},
        )
        assert response.status_code == 200


class TestCommunityEndpoint:
    """Tests for /graph/community endpoint."""

    def test_get_community(self, client):
        """Test get community."""
        response = client.get("/api/v1/graph/community/comm_1")
        assert response.status_code == 404  # Mock has no communities


class TestFilesEndpoint:
    """Tests for /graph/files endpoint."""

    def test_list_files(self, client):
        """Test list files."""
        response = client.get("/api/v1/graph/files")
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert "total" in data

    def test_list_files_with_pattern(self, client):
        """Test list files with pattern."""
        response = client.get(
            "/api/v1/graph/files",
            params={"pattern": "*.py"},
        )
        assert response.status_code == 200


class TestFileNodesEndpoint:
    """Tests for /graph/file/.../nodes endpoint."""

    def test_get_file_nodes(self, client):
        """Test get file nodes."""
        response = client.get("/api/v1/graph/file/test.py/nodes")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestGraphModels:
    """Tests for graph models."""

    def test_node_info_model(self):
        """Test NodeInfo model."""
        from smart_search.api.endpoints.graph import NodeInfo

        node = NodeInfo(
            id="test_id",
            name="test_func",
            qualified_name="module.test_func",
            code_type="function",
            file_path="/test.py",
            line_start=1,
            line_end=10,
        )
        assert node.id == "test_id"
        assert node.name == "test_func"

    def test_edge_info_model(self):
        """Test EdgeInfo model."""
        from smart_search.api.endpoints.graph import EdgeInfo

        edge = EdgeInfo(
            source="id1",
            target="id2",
            edge_type="CALLS",
            weight=1.0,
        )
        assert edge.source == "id1"
        assert edge.edge_type == "CALLS"

    def test_community_info_model(self):
        """Test CommunityInfo model."""
        from smart_search.api.endpoints.graph import CommunityInfo

        community = CommunityInfo(
            id="comm_1",
            name="Community 1",
            size=10,
            members=["m1", "m2"],
            key_members=["m1"],
        )
        assert community.size == 10
        assert len(community.members) == 2

    def test_graph_stats_model(self):
        """Test GraphStats model."""
        from smart_search.api.endpoints.graph import GraphStats

        stats = GraphStats(
            total_nodes=100,
            total_edges=200,
            node_types={"function": 50, "class": 50},
            edge_types={"CALLS": 200},
            files_count=10,
            communities_count=5,
            avg_degree=4.0,
        )
        assert stats.total_nodes == 100
        assert stats.avg_degree == 4.0

    def test_path_info_model(self):
        """Test PathInfo model."""
        from smart_search.api.endpoints.graph import PathInfo, EdgeInfo

        path = PathInfo(
            source="id1",
            target="id3",
            path=["id1", "id2", "id3"],
            length=2,
            edges=[
                EdgeInfo(source="id1", target="id2", edge_type="CALLS"),
                EdgeInfo(source="id2", target="id3", edge_type="CALLS"),
            ],
        )
        assert path.length == 2
        assert len(path.edges) == 2

    def test_subgraph_response_model(self):
        """Test SubgraphResponse model."""
        from smart_search.api.endpoints.graph import SubgraphResponse, NodeInfo, EdgeInfo

        subgraph = SubgraphResponse(
            center="center_id",
            nodes=[
                NodeInfo(
                    id="center_id",
                    name="center",
                    qualified_name="center",
                    code_type="function",
                    file_path="/test.py",
                    line_start=1,
                    line_end=10,
                )
            ],
            edges=[],
            total_nodes=1,
            total_edges=0,
        )
        assert subgraph.center == "center_id"
        assert subgraph.total_nodes == 1


class TestGraphDependencies:
    """Tests for graph dependencies."""

    def test_set_get_graph(self):
        """Test graph setter/getter."""
        from smart_search.api.endpoints import graph

        original = graph._graph
        try:
            graph.set_graph(None)
            with pytest.raises(Exception):
                graph.get_graph()

            mock_graph = object()
            graph.set_graph(mock_graph)
            assert graph.get_graph() is mock_graph
        finally:
            graph._graph = original


class TestGraphHelpers:
    """Tests for graph helper functions."""

    def test_node_to_info(self):
        """Test node_to_info helper."""
        from smart_search.api.endpoints.graph import node_to_info
        from dataclasses import dataclass
        from enum import Enum

        class NodeType(Enum):
            FUNCTION = "function"

        @dataclass
        class MockNode:
            id: str = "test"
            name: str = "test_func"
            qualified_name: str = "module.test_func"
            type: NodeType = NodeType.FUNCTION
            file_path: str = "/test.py"
            line_start: int = 1
            line_end: int = 10

        node = MockNode()
        info = node_to_info(node)
        assert info.id == "test"
        assert info.code_type == "function"
