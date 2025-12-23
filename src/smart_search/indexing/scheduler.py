"""Indexing scheduler for background operations.

Schedules and manages indexing tasks with priority queues.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class TaskPriority(int, Enum):
    """Priority levels for indexing tasks."""

    CRITICAL = 0  # Immediate processing
    HIGH = 1  # User-triggered changes
    NORMAL = 2  # Regular file changes
    LOW = 3  # Background maintenance
    IDLE = 4  # Only when idle


class TaskStatus(str, Enum):
    """Status of an indexing task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class IndexTask:
    """An indexing task in the queue.

    Attributes:
        priority: Task priority (lower = higher priority).
        created_at: Task creation timestamp.
        task_id: Unique task identifier.
        file_paths: Files to index.
        callback: Optional completion callback.
        status: Current task status.
        error: Error message if failed.
        result: Task result if completed.
    """

    priority: TaskPriority
    created_at: float = field(compare=True)
    task_id: str = field(compare=False)
    file_paths: list[Path] = field(compare=False, default_factory=list)
    callback: Callable[[Any], Coroutine[Any, Any, None]] | None = field(
        compare=False, default=None
    )
    status: TaskStatus = field(compare=False, default=TaskStatus.PENDING)
    error: str | None = field(compare=False, default=None)
    result: Any = field(compare=False, default=None)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "priority": self.priority.name,
            "status": self.status.value,
            "file_paths": [str(p) for p in self.file_paths],
            "created_at": self.created_at,
            "error": self.error,
        }


@dataclass
class SchedulerConfig:
    """Configuration for the indexing scheduler.

    Attributes:
        max_concurrent: Maximum concurrent tasks.
        batch_size: Files per batch.
        batch_delay: Delay between batches (seconds).
        idle_threshold: Seconds of inactivity before idle tasks.
        max_retries: Maximum retry attempts for failed tasks.
    """

    max_concurrent: int = 2
    batch_size: int = 50
    batch_delay: float = 0.1
    idle_threshold: float = 5.0
    max_retries: int = 3


class TaskQueue:
    """Priority queue for indexing tasks."""

    def __init__(self) -> None:
        """Initialize task queue."""
        self._tasks: list[IndexTask] = []
        self._task_map: dict[str, IndexTask] = {}
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition()

    async def put(self, task: IndexTask) -> None:
        """Add a task to the queue.

        Args:
            task: Task to add.
        """
        async with self._lock:
            if task.task_id in self._task_map:
                # Update existing task if still pending
                existing = self._task_map[task.task_id]
                if existing.status == TaskStatus.PENDING:
                    # Merge file paths
                    existing.file_paths.extend(task.file_paths)
                    # Use higher priority
                    if task.priority < existing.priority:
                        existing.priority = task.priority
                    return

            self._tasks.append(task)
            self._tasks.sort()  # Maintain priority order
            self._task_map[task.task_id] = task

        async with self._not_empty:
            self._not_empty.notify()

    async def get(self) -> IndexTask | None:
        """Get the highest priority task.

        Returns:
            Highest priority pending task or None.
        """
        async with self._lock:
            for task in self._tasks:
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.RUNNING
                    return task
            return None

    async def wait_for_task(self, timeout: float | None = None) -> IndexTask | None:
        """Wait for a task to become available.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            Task or None if timeout.
        """
        async with self._not_empty:
            try:
                await asyncio.wait_for(
                    self._not_empty.wait(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return None

        return await self.get()

    def get_task(self, task_id: str) -> IndexTask | None:
        """Get a task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            Task or None if not found.
        """
        return self._task_map.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.

        Args:
            task_id: Task identifier.

        Returns:
            True if cancelled.
        """
        task = self._task_map.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
        return False

    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """Mark a task as completed.

        Args:
            task_id: Task identifier.
            result: Task result.

        Returns:
            True if updated.
        """
        task = self._task_map.get(task_id)
        if task and task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.COMPLETED
            task.result = result
            return True
        return False

    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed.

        Args:
            task_id: Task identifier.
            error: Error message.

        Returns:
            True if updated.
        """
        task = self._task_map.get(task_id)
        if task and task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.FAILED
            task.error = error
            return True
        return False

    @property
    def pending_count(self) -> int:
        """Number of pending tasks."""
        return sum(1 for t in self._tasks if t.status == TaskStatus.PENDING)

    @property
    def running_count(self) -> int:
        """Number of running tasks."""
        return sum(1 for t in self._tasks if t.status == TaskStatus.RUNNING)

    def get_pending_tasks(self) -> list[IndexTask]:
        """Get all pending tasks."""
        return [t for t in self._tasks if t.status == TaskStatus.PENDING]

    def get_completed_tasks(self) -> list[IndexTask]:
        """Get all completed tasks."""
        return [t for t in self._tasks if t.status == TaskStatus.COMPLETED]

    def get_failed_tasks(self) -> list[IndexTask]:
        """Get all failed tasks."""
        return [t for t in self._tasks if t.status == TaskStatus.FAILED]

    def clear_completed(self) -> int:
        """Remove completed tasks from queue.

        Returns:
            Number of tasks removed.
        """
        original_count = len(self._tasks)
        self._tasks = [
            t for t in self._tasks if t.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
        ]
        removed = original_count - len(self._tasks)

        # Update task map
        for task_id in list(self._task_map.keys()):
            if self._task_map[task_id].status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                del self._task_map[task_id]

        return removed


class IndexingScheduler:
    """Schedules and executes indexing tasks."""

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        index_func: Callable[[list[Path]], Coroutine[Any, Any, Any]] | None = None,
    ) -> None:
        """Initialize scheduler.

        Args:
            config: Scheduler configuration.
            index_func: Function to call for indexing files.
        """
        self.config = config or SchedulerConfig()
        self._index_func = index_func
        self._queue = TaskQueue()
        self._running = False
        self._workers: list[asyncio.Task[None]] = []
        self._task_counter = 0
        self._last_activity = time.time()
        self._stats = SchedulerStats()

    def set_index_function(
        self,
        func: Callable[[list[Path]], Coroutine[Any, Any, Any]],
    ) -> None:
        """Set the indexing function.

        Args:
            func: Function to call for indexing.
        """
        self._index_func = func

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        self._task_counter += 1
        return f"task_{self._task_counter}_{int(time.time() * 1000)}"

    async def schedule(
        self,
        file_paths: list[Path],
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Callable[[Any], Coroutine[Any, Any, None]] | None = None,
    ) -> str:
        """Schedule files for indexing.

        Args:
            file_paths: Files to index.
            priority: Task priority.
            callback: Optional completion callback.

        Returns:
            Task ID.
        """
        task = IndexTask(
            task_id=self._generate_task_id(),
            priority=priority,
            created_at=time.time(),
            file_paths=file_paths,
            callback=callback,
        )

        await self._queue.put(task)
        self._stats.tasks_scheduled += 1

        logger.debug(
            "Task scheduled",
            task_id=task.task_id,
            priority=priority.name,
            file_count=len(file_paths),
        )

        return task.task_id

    async def schedule_batch(
        self,
        file_paths: list[Path],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> list[str]:
        """Schedule files in batches.

        Args:
            file_paths: Files to index.
            priority: Task priority.

        Returns:
            List of task IDs.
        """
        task_ids = []
        batch_size = self.config.batch_size

        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i : i + batch_size]
            task_id = await self.schedule(batch, priority)
            task_ids.append(task_id)

        return task_ids

    def get_task_status(self, task_id: str) -> TaskStatus | None:
        """Get status of a task.

        Args:
            task_id: Task identifier.

        Returns:
            Task status or None if not found.
        """
        task = self._queue.get_task(task_id)
        return task.status if task else None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.

        Args:
            task_id: Task identifier.

        Returns:
            True if cancelled.
        """
        if self._queue.cancel_task(task_id):
            self._stats.tasks_cancelled += 1
            return True
        return False

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine for processing tasks.

        Args:
            worker_id: Worker identifier.
        """
        logger.debug("Worker started", worker_id=worker_id)

        while self._running:
            task = await self._queue.wait_for_task(timeout=1.0)

            if task is None:
                continue

            self._last_activity = time.time()

            try:
                logger.debug(
                    "Processing task",
                    worker_id=worker_id,
                    task_id=task.task_id,
                    file_count=len(task.file_paths),
                )

                if self._index_func:
                    result = await self._index_func(task.file_paths)
                else:
                    result = {"files_processed": len(task.file_paths)}

                self._queue.complete_task(task.task_id, result)
                self._stats.tasks_completed += 1
                self._stats.files_indexed += len(task.file_paths)

                if task.callback:
                    try:
                        await task.callback(result)
                    except Exception as e:
                        logger.warning(
                            "Task callback failed",
                            task_id=task.task_id,
                            error=str(e),
                        )

                # Delay between batches
                if self.config.batch_delay > 0:
                    await asyncio.sleep(self.config.batch_delay)

            except Exception as e:
                logger.error(
                    "Task failed",
                    task_id=task.task_id,
                    error=str(e),
                )
                self._queue.fail_task(task.task_id, str(e))
                self._stats.tasks_failed += 1

        logger.debug("Worker stopped", worker_id=worker_id)

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._last_activity = time.time()

        # Start workers
        for i in range(self.config.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        logger.info(
            "Scheduler started",
            workers=self.config.max_concurrent,
        )

    async def stop(self, wait: bool = True) -> None:
        """Stop the scheduler.

        Args:
            wait: Whether to wait for tasks to complete.
        """
        if not self._running:
            return

        self._running = False

        if wait and self._workers:
            # Wait for workers to finish
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        logger.info("Scheduler stopped")

    async def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Wait for all pending tasks to complete.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            True if all tasks completed.
        """
        start = time.time()

        while self._queue.pending_count > 0 or self._queue.running_count > 0:
            if timeout and (time.time() - start) > timeout:
                return False
            await asyncio.sleep(0.1)

        return True

    @property
    def is_running(self) -> bool:
        """Whether scheduler is running."""
        return self._running

    @property
    def is_idle(self) -> bool:
        """Whether scheduler is idle."""
        return (
            self._queue.pending_count == 0
            and self._queue.running_count == 0
            and (time.time() - self._last_activity) > self.config.idle_threshold
        )

    @property
    def pending_count(self) -> int:
        """Number of pending tasks."""
        return self._queue.pending_count

    @property
    def stats(self) -> "SchedulerStats":
        """Get scheduler statistics."""
        return self._stats

    def get_queue_status(self) -> dict[str, Any]:
        """Get queue status summary."""
        return {
            "pending": self._queue.pending_count,
            "running": self._queue.running_count,
            "completed": len(self._queue.get_completed_tasks()),
            "failed": len(self._queue.get_failed_tasks()),
            "is_idle": self.is_idle,
        }

    def clear_completed(self) -> int:
        """Clear completed tasks from queue.

        Returns:
            Number of tasks cleared.
        """
        return self._queue.clear_completed()


@dataclass
class SchedulerStats:
    """Statistics for the scheduler.

    Attributes:
        tasks_scheduled: Total tasks scheduled.
        tasks_completed: Total tasks completed.
        tasks_failed: Total tasks failed.
        tasks_cancelled: Total tasks cancelled.
        files_indexed: Total files indexed.
    """

    tasks_scheduled: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_cancelled: int = 0
    files_indexed: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "tasks_scheduled": self.tasks_scheduled,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_cancelled": self.tasks_cancelled,
            "files_indexed": self.files_indexed,
        }

    @property
    def success_rate(self) -> float:
        """Task success rate (0-1)."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_completed / total


class ThrottledScheduler(IndexingScheduler):
    """Scheduler with rate limiting and throttling."""

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        index_func: Callable[[list[Path]], Coroutine[Any, Any, Any]] | None = None,
        max_rate: float = 10.0,  # Max tasks per second
    ) -> None:
        """Initialize throttled scheduler.

        Args:
            config: Scheduler configuration.
            index_func: Indexing function.
            max_rate: Maximum tasks per second.
        """
        super().__init__(config, index_func)
        self._max_rate = max_rate
        self._rate_limiter = RateLimiter(max_rate)

    async def schedule(
        self,
        file_paths: list[Path],
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Callable[[Any], Coroutine[Any, Any, None]] | None = None,
    ) -> str:
        """Schedule with rate limiting."""
        await self._rate_limiter.acquire()
        return await super().schedule(file_paths, priority, callback)


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: float) -> None:
        """Initialize rate limiter.

        Args:
            rate: Maximum operations per second.
        """
        self._rate = rate
        self._tokens = rate
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            self._last_update = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self._rate
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1
