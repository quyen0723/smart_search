#!/usr/bin/env python3
"""Load testing script for Smart Search API.

Uses Locust for load testing. Run with:
    locust -f scripts/load_test.py --host=http://localhost:8000

Or headless mode for CI:
    locust -f scripts/load_test.py --host=http://localhost:8000 \
           --headless -u 100 -r 10 --run-time 60s

Requirements:
    pip install locust

Target metrics (from implementation plan):
    - search() p50: 20ms, p99: 50ms
    - find_references() p50: 100ms, p99: 500ms
    - 100 concurrent users stable
"""

import random
from locust import HttpUser, task, between


class SmartSearchUser(HttpUser):
    """Simulates a user performing search operations."""

    wait_time = between(0.5, 2)  # Wait 0.5-2 seconds between tasks

    # Sample queries for testing
    SEARCH_QUERIES = [
        "get_user",
        "calculate",
        "process",
        "validate",
        "parse",
        "render",
        "handle",
        "create",
        "update",
        "delete",
        "find",
        "search",
        "index",
        "query",
        "fetch",
    ]

    SYMBOL_NAMES = [
        "main",
        "init",
        "config",
        "utils",
        "helper",
        "service",
        "controller",
        "model",
        "view",
        "handler",
    ]

    @task(10)
    def search_keyword(self):
        """Test keyword search - most common operation."""
        query = random.choice(self.SEARCH_QUERIES)
        self.client.post(
            "/api/v1/search/search",
            json={
                "query": query,
                "limit": 20,
                "search_type": "keyword",
            },
            name="/api/v1/search/search [POST]",
        )

    @task(5)
    def search_get(self):
        """Test GET search endpoint."""
        query = random.choice(self.SEARCH_QUERIES)
        self.client.get(
            f"/api/v1/search/search?q={query}&limit=10",
            name="/api/v1/search/search [GET]",
        )

    @task(3)
    def find_references(self):
        """Test find_references endpoint."""
        filename = random.choice(self.SYMBOL_NAMES) + ".py"
        self.client.get(
            f"/api/v1/index/index/references?filename={filename}&limit=50",
            name="/api/v1/index/references",
        )

    @task(2)
    def search_symbols(self):
        """Test symbol search."""
        query = random.choice(self.SYMBOL_NAMES)
        self.client.get(
            f"/api/v1/navigate/navigate/symbols?query={query}&limit=20",
            name="/api/v1/navigate/symbols",
        )

    @task(2)
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health", name="/health")

    @task(1)
    def get_stats(self):
        """Test graph stats endpoint."""
        self.client.get("/api/v1/graph/graph/stats", name="/api/v1/graph/stats")

    @task(1)
    def index_stats(self):
        """Test index stats endpoint."""
        self.client.get("/api/v1/index/index/stats", name="/api/v1/index/stats")


class HeavySearchUser(HttpUser):
    """Simulates a power user with more complex queries."""

    wait_time = between(1, 3)
    weight = 1  # Less common than regular users

    @task(5)
    def rag_search(self):
        """Test RAG search - heavier operation."""
        queries = [
            "How does user authentication work?",
            "What functions handle database queries?",
            "Where is error handling implemented?",
            "How is caching done in this project?",
        ]
        self.client.post(
            "/api/v1/search/search/rag",
            json={
                "query": random.choice(queries),
                "mode": "hybrid",
                "include_explanation": True,
            },
            name="/api/v1/search/rag",
        )

    @task(3)
    def similar_code(self):
        """Test similar code search."""
        code_snippets = [
            "def get_user(user_id): return db.query(User).get(user_id)",
            "class UserService: pass",
            "async def fetch_data(url): return await http.get(url)",
        ]
        self.client.post(
            "/api/v1/search/search/similar",
            json={
                "code": random.choice(code_snippets),
                "limit": 10,
            },
            name="/api/v1/search/similar",
        )

    @task(2)
    def analyze_impact(self):
        """Test impact analysis."""
        self.client.post(
            "/api/v1/analyze/analyze/impact",
            json={
                "code_id": "test::main",
                "depth": 3,
                "include_indirect": True,
            },
            name="/api/v1/analyze/impact",
        )


if __name__ == "__main__":
    import subprocess
    import sys

    # Quick test mode - run for 30 seconds with 10 users
    print("Starting quick load test (30s, 10 users)...")
    print("For full test, run: locust -f scripts/load_test.py --host=http://localhost:8000")

    result = subprocess.run([
        sys.executable, "-m", "locust",
        "-f", __file__,
        "--host", "http://localhost:8000",
        "--headless",
        "-u", "10",
        "-r", "2",
        "--run-time", "30s",
    ])
    sys.exit(result.returncode)
