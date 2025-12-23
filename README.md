# Smart Search

Intelligent Source Code Navigation System with Hybrid Search, Graph Analysis, and GraphRAG.

## Features

- **Hybrid Search**: Combined keyword + semantic search using Meilisearch
- **Graph Analysis**: Code dependency graph analysis using RustworkX
- **GraphRAG**: AI-powered code understanding with graph-enhanced retrieval
- **Incremental Indexing**: Smart change detection for efficient updates

## Requirements

- Python 3.11+
- Docker (for Meilisearch)

## Quick Start

```bash
# Clone and setup
git clone <repo>
cd smart_search

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Start Meilisearch
docker-compose up -d

# Run the server
smart-search
```

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src/smart_search --cov-report=html

# Type checking
mypy src/

# Linting
ruff check src/
```

## License

MIT
