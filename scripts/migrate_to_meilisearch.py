#!/usr/bin/env python3
"""Index migration script: Backfill data from Graph to Meilisearch.

This script migrates existing indexed code units from the Graph storage
to Meilisearch for full-text search. Run this after enabling:
  - FF_USE_MEILISEARCH_SEARCH=true
  - FF_USE_HYBRID_SEARCHER=true

Usage:
    python scripts/migrate_to_meilisearch.py [--batch-size=100] [--dry-run]

Example:
    # Dry run to see what would be migrated
    python scripts/migrate_to_meilisearch.py --dry-run

    # Full migration with custom batch size
    python scripts/migrate_to_meilisearch.py --batch-size=500
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smart_search.config import get_settings
from smart_search.graph import CodeGraph, GraphPersistence
from smart_search.search import MeilisearchClient, IndexDocument
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


async def migrate_graph_to_meilisearch(
    graph_path: Path,
    meilisearch_url: str,
    meilisearch_api_key: str | None,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict:
    """Migrate data from Graph to Meilisearch.

    Args:
        graph_path: Path to graph storage directory.
        meilisearch_url: Meilisearch server URL.
        meilisearch_api_key: Optional API key.
        batch_size: Documents per batch.
        dry_run: If True, don't actually write to Meilisearch.

    Returns:
        Migration statistics.
    """
    stats = {
        "nodes_found": 0,
        "documents_created": 0,
        "documents_indexed": 0,
        "errors": 0,
        "skipped": 0,
    }

    # Load graph
    print(f"Loading graph from {graph_path}...")
    persistence = GraphPersistence(graph_path)
    graph_file = graph_path / "graph.json"

    if not graph_file.exists():
        print(f"Graph file not found: {graph_file}")
        return stats

    graph = persistence.load_json(graph_file)
    if graph is None:
        print("Failed to load graph")
        return stats

    # Get all nodes
    try:
        nodes = graph.get_all_nodes()
        stats["nodes_found"] = len(nodes)
        print(f"Found {len(nodes)} nodes in graph")
    except Exception as e:
        print(f"Error getting nodes: {e}")
        return stats

    if not nodes:
        print("No nodes to migrate")
        return stats

    # Convert nodes to IndexDocuments
    documents = []
    for node in nodes:
        try:
            data = node.data if hasattr(node, 'data') else node

            # Get file content if available
            content = ""
            file_path = str(data.file_path) if data.file_path else ""
            if file_path and Path(file_path).exists():
                try:
                    file_content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
                    lines = file_content.split('\n')
                    start = max(0, (data.line_start or 1) - 1)
                    end = min(len(lines), data.line_end or len(lines))
                    content = '\n'.join(lines[start:end])
                except Exception:
                    pass

            doc = IndexDocument(
                id=data.id,
                name=data.name,
                qualified_name=getattr(data, 'qualified_name', data.name),
                code_type=str(getattr(data, 'node_type', 'unknown')),
                file_path=file_path,
                line_start=data.line_start or 0,
                line_end=data.line_end or 0,
                content=content,
                language=str(getattr(data, 'language', 'unknown')),
                signature=getattr(data, 'signature', ''),
                docstring=getattr(data, 'docstring', ''),
                embedding=None,
            )
            documents.append(doc)
            stats["documents_created"] += 1

        except Exception as e:
            logger.warning(f"Error converting node: {e}")
            stats["errors"] += 1

    print(f"Created {stats['documents_created']} documents")

    if dry_run:
        print("\n[DRY RUN] Would index the following:")
        for i, doc in enumerate(documents[:5]):
            print(f"  {i+1}. {doc.name} ({doc.code_type}) - {doc.file_path}")
        if len(documents) > 5:
            print(f"  ... and {len(documents) - 5} more")
        return stats

    # Connect to Meilisearch and index
    print(f"\nConnecting to Meilisearch at {meilisearch_url}...")
    client = MeilisearchClient(
        url=meilisearch_url,
        api_key=meilisearch_api_key,
    )

    try:
        # Initialize index
        print("Initializing index...")
        await client.initialize_index(recreate=False)

        # Index in batches
        print(f"Indexing {len(documents)} documents in batches of {batch_size}...")
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                count = await client.add_documents(batch, batch_size=batch_size)
                stats["documents_indexed"] += count
                progress = min(i + batch_size, len(documents))
                print(f"  Indexed {progress}/{len(documents)} documents")
            except Exception as e:
                logger.error(f"Batch indexing failed: {e}")
                stats["errors"] += len(batch)

        print(f"\nMigration complete: {stats['documents_indexed']} documents indexed")

    finally:
        await client.close()

    return stats


async def main():
    parser = argparse.ArgumentParser(
        description="Migrate index data from Graph to Meilisearch"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Documents per batch (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually indexing"
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help="Path to graph storage (default: from settings)"
    )
    parser.add_argument(
        "--meilisearch-url",
        type=str,
        default=None,
        help="Meilisearch URL (default: from settings)"
    )

    args = parser.parse_args()

    # Get settings
    settings = get_settings()

    graph_path = Path(args.graph_path) if args.graph_path else settings.graph.storage_path
    meilisearch_url = args.meilisearch_url or settings.meilisearch.url
    meilisearch_api_key = settings.meilisearch.master_key

    print("=" * 60)
    print("Smart Search Index Migration: Graph -> Meilisearch")
    print("=" * 60)
    print(f"Graph path: {graph_path}")
    print(f"Meilisearch: {meilisearch_url}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)

    stats = await migrate_graph_to_meilisearch(
        graph_path=graph_path,
        meilisearch_url=meilisearch_url,
        meilisearch_api_key=meilisearch_api_key,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print("Migration Statistics:")
    print(f"  Nodes found:        {stats['nodes_found']}")
    print(f"  Documents created:  {stats['documents_created']}")
    print(f"  Documents indexed:  {stats['documents_indexed']}")
    print(f"  Errors:             {stats['errors']}")
    print(f"  Skipped:            {stats['skipped']}")
    print("=" * 60)

    return 0 if stats['errors'] == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
