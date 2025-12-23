"""Indexing module for incremental code indexing.

Provides file watching, content hashing, and scheduled indexing.
"""

from smart_search.indexing.hasher import (
    ContentHasher,
    FileHash,
    HashComparison,
    HashStore,
    compute_directory_hash,
)
from smart_search.indexing.incremental import (
    DeltaBuilder,
    IncrementalIndexer,
    IncrementalUpdate,
    IndexState,
)
from smart_search.indexing.scheduler import (
    IndexingScheduler,
    IndexTask,
    RateLimiter,
    SchedulerConfig,
    SchedulerStats,
    TaskPriority,
    TaskQueue,
    TaskStatus,
    ThrottledScheduler,
)
from smart_search.indexing.watcher import (
    ChangeCollector,
    ChangeType,
    FileChange,
    FileWatcher,
    WatcherConfig,
)

__all__ = [
    # Hasher
    "ContentHasher",
    "FileHash",
    "HashComparison",
    "HashStore",
    "compute_directory_hash",
    # Incremental
    "DeltaBuilder",
    "IncrementalIndexer",
    "IncrementalUpdate",
    "IndexState",
    # Scheduler
    "IndexingScheduler",
    "IndexTask",
    "RateLimiter",
    "SchedulerConfig",
    "SchedulerStats",
    "TaskPriority",
    "TaskQueue",
    "TaskStatus",
    "ThrottledScheduler",
    # Watcher
    "ChangeCollector",
    "ChangeType",
    "FileChange",
    "FileWatcher",
    "WatcherConfig",
]
