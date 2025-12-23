"""Tests for indexing scheduler."""

import asyncio
import time
from pathlib import Path

import pytest

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


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_ordering(self) -> None:
        """Test priority ordering."""
        assert TaskPriority.CRITICAL < TaskPriority.HIGH
        assert TaskPriority.HIGH < TaskPriority.NORMAL
        assert TaskPriority.NORMAL < TaskPriority.LOW
        assert TaskPriority.LOW < TaskPriority.IDLE


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_values(self) -> None:
        """Test status values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestIndexTask:
    """Tests for IndexTask."""

    def test_creation(self, tmp_path: Path) -> None:
        """Test task creation."""
        task = IndexTask(
            task_id="task_1",
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            file_paths=[tmp_path / "file.py"],
        )

        assert task.task_id == "task_1"
        assert task.priority == TaskPriority.NORMAL
        assert task.status == TaskStatus.PENDING
        assert len(task.file_paths) == 1

    def test_to_dict(self, tmp_path: Path) -> None:
        """Test conversion to dict."""
        task = IndexTask(
            task_id="task_1",
            priority=TaskPriority.HIGH,
            created_at=1234567890.0,
            file_paths=[tmp_path / "file.py"],
        )

        d = task.to_dict()

        assert d["task_id"] == "task_1"
        assert d["priority"] == "HIGH"
        assert d["status"] == "pending"

    def test_ordering_by_priority(self) -> None:
        """Test tasks are ordered by priority."""
        task_high = IndexTask(
            task_id="high",
            priority=TaskPriority.HIGH,
            created_at=1.0,
        )
        task_low = IndexTask(
            task_id="low",
            priority=TaskPriority.LOW,
            created_at=1.0,
        )

        assert task_high < task_low

    def test_ordering_by_time(self) -> None:
        """Test tasks with same priority are ordered by time."""
        task_early = IndexTask(
            task_id="early",
            priority=TaskPriority.NORMAL,
            created_at=1.0,
        )
        task_late = IndexTask(
            task_id="late",
            priority=TaskPriority.NORMAL,
            created_at=2.0,
        )

        assert task_early < task_late


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = SchedulerConfig()

        assert config.max_concurrent == 2
        assert config.batch_size == 50
        assert config.batch_delay == 0.1
        assert config.idle_threshold == 5.0
        assert config.max_retries == 3

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SchedulerConfig(
            max_concurrent=4,
            batch_size=100,
            batch_delay=0.5,
        )

        assert config.max_concurrent == 4
        assert config.batch_size == 100


class TestTaskQueue:
    """Tests for TaskQueue."""

    @pytest.fixture
    def queue(self) -> TaskQueue:
        """Create task queue."""
        return TaskQueue()

    @pytest.mark.asyncio
    async def test_put_and_get(self, queue: TaskQueue) -> None:
        """Test put and get."""
        task = IndexTask(
            task_id="task_1",
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
        )

        await queue.put(task)
        retrieved = await queue.get()

        assert retrieved is not None
        assert retrieved.task_id == "task_1"
        assert retrieved.status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue: TaskQueue) -> None:
        """Test tasks are retrieved by priority."""
        task_low = IndexTask(task_id="low", priority=TaskPriority.LOW, created_at=time.time())
        task_high = IndexTask(task_id="high", priority=TaskPriority.HIGH, created_at=time.time())
        task_normal = IndexTask(task_id="normal", priority=TaskPriority.NORMAL, created_at=time.time())

        await queue.put(task_low)
        await queue.put(task_high)
        await queue.put(task_normal)

        first = await queue.get()
        second = await queue.get()
        third = await queue.get()

        assert first.task_id == "high"
        assert second.task_id == "normal"
        assert third.task_id == "low"

    @pytest.mark.asyncio
    async def test_get_empty_queue(self, queue: TaskQueue) -> None:
        """Test get returns None for empty queue."""
        result = await queue.get()
        assert result is None

    @pytest.mark.asyncio
    async def test_merge_duplicate_task(self, queue: TaskQueue, tmp_path: Path) -> None:
        """Test merging duplicate task IDs."""
        task1 = IndexTask(
            task_id="same_id",
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            file_paths=[tmp_path / "file1.py"],
        )
        task2 = IndexTask(
            task_id="same_id",
            priority=TaskPriority.HIGH,
            created_at=time.time(),
            file_paths=[tmp_path / "file2.py"],
        )

        await queue.put(task1)
        await queue.put(task2)

        # Should merge into one task with higher priority
        assert queue.pending_count == 1

        retrieved = await queue.get()
        assert retrieved.priority == TaskPriority.HIGH
        assert len(retrieved.file_paths) == 2

    def test_get_task(self, queue: TaskQueue) -> None:
        """Test get_task by ID."""
        task = IndexTask(task_id="task_1", priority=TaskPriority.NORMAL, created_at=time.time())
        queue._tasks.append(task)
        queue._task_map[task.task_id] = task

        retrieved = queue.get_task("task_1")
        assert retrieved is not None
        assert retrieved.task_id == "task_1"

        assert queue.get_task("nonexistent") is None

    def test_cancel_task(self, queue: TaskQueue) -> None:
        """Test cancelling a task."""
        task = IndexTask(task_id="task_1", priority=TaskPriority.NORMAL, created_at=time.time())
        queue._tasks.append(task)
        queue._task_map[task.task_id] = task

        result = queue.cancel_task("task_1")

        assert result is True
        assert task.status == TaskStatus.CANCELLED

    def test_cancel_task_not_pending(self, queue: TaskQueue) -> None:
        """Test cancelling non-pending task fails."""
        task = IndexTask(task_id="task_1", priority=TaskPriority.NORMAL, created_at=time.time())
        task.status = TaskStatus.RUNNING
        queue._tasks.append(task)
        queue._task_map[task.task_id] = task

        result = queue.cancel_task("task_1")

        assert result is False

    def test_complete_task(self, queue: TaskQueue) -> None:
        """Test completing a task."""
        task = IndexTask(task_id="task_1", priority=TaskPriority.NORMAL, created_at=time.time())
        task.status = TaskStatus.RUNNING
        queue._tasks.append(task)
        queue._task_map[task.task_id] = task

        result = queue.complete_task("task_1", result={"count": 5})

        assert result is True
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"count": 5}

    def test_fail_task(self, queue: TaskQueue) -> None:
        """Test failing a task."""
        task = IndexTask(task_id="task_1", priority=TaskPriority.NORMAL, created_at=time.time())
        task.status = TaskStatus.RUNNING
        queue._tasks.append(task)
        queue._task_map[task.task_id] = task

        result = queue.fail_task("task_1", "Error message")

        assert result is True
        assert task.status == TaskStatus.FAILED
        assert task.error == "Error message"

    def test_pending_count(self, queue: TaskQueue) -> None:
        """Test pending_count property."""
        task1 = IndexTask(task_id="task_1", priority=TaskPriority.NORMAL, created_at=time.time())
        task2 = IndexTask(task_id="task_2", priority=TaskPriority.NORMAL, created_at=time.time())
        task2.status = TaskStatus.COMPLETED

        queue._tasks = [task1, task2]

        assert queue.pending_count == 1

    def test_running_count(self, queue: TaskQueue) -> None:
        """Test running_count property."""
        task1 = IndexTask(task_id="task_1", priority=TaskPriority.NORMAL, created_at=time.time())
        task1.status = TaskStatus.RUNNING
        task2 = IndexTask(task_id="task_2", priority=TaskPriority.NORMAL, created_at=time.time())

        queue._tasks = [task1, task2]

        assert queue.running_count == 1

    def test_get_pending_tasks(self, queue: TaskQueue) -> None:
        """Test get_pending_tasks method."""
        pending = IndexTask(task_id="pending", priority=TaskPriority.NORMAL, created_at=time.time())
        running = IndexTask(task_id="running", priority=TaskPriority.NORMAL, created_at=time.time())
        running.status = TaskStatus.RUNNING

        queue._tasks = [pending, running]

        pending_tasks = queue.get_pending_tasks()
        assert len(pending_tasks) == 1
        assert pending_tasks[0].task_id == "pending"

    def test_get_completed_tasks(self, queue: TaskQueue) -> None:
        """Test get_completed_tasks method."""
        completed = IndexTask(task_id="completed", priority=TaskPriority.NORMAL, created_at=time.time())
        completed.status = TaskStatus.COMPLETED
        pending = IndexTask(task_id="pending", priority=TaskPriority.NORMAL, created_at=time.time())

        queue._tasks = [completed, pending]

        completed_tasks = queue.get_completed_tasks()
        assert len(completed_tasks) == 1

    def test_get_failed_tasks(self, queue: TaskQueue) -> None:
        """Test get_failed_tasks method."""
        failed = IndexTask(task_id="failed", priority=TaskPriority.NORMAL, created_at=time.time())
        failed.status = TaskStatus.FAILED

        queue._tasks = [failed]

        failed_tasks = queue.get_failed_tasks()
        assert len(failed_tasks) == 1

    def test_clear_completed(self, queue: TaskQueue) -> None:
        """Test clear_completed method."""
        completed = IndexTask(task_id="completed", priority=TaskPriority.NORMAL, created_at=time.time())
        completed.status = TaskStatus.COMPLETED
        cancelled = IndexTask(task_id="cancelled", priority=TaskPriority.NORMAL, created_at=time.time())
        cancelled.status = TaskStatus.CANCELLED
        pending = IndexTask(task_id="pending", priority=TaskPriority.NORMAL, created_at=time.time())

        queue._tasks = [completed, cancelled, pending]
        queue._task_map = {t.task_id: t for t in queue._tasks}

        removed = queue.clear_completed()

        assert removed == 2
        assert len(queue._tasks) == 1
        assert queue._tasks[0].task_id == "pending"


class TestIndexingScheduler:
    """Tests for IndexingScheduler."""

    @pytest.fixture
    def scheduler(self) -> IndexingScheduler:
        """Create scheduler."""
        config = SchedulerConfig(max_concurrent=2, batch_delay=0.01)
        return IndexingScheduler(config)

    @pytest.mark.asyncio
    async def test_schedule(self, scheduler: IndexingScheduler, tmp_path: Path) -> None:
        """Test scheduling a task."""
        files = [tmp_path / "file.py"]

        task_id = await scheduler.schedule(files)

        assert task_id is not None
        assert scheduler.pending_count == 1

    @pytest.mark.asyncio
    async def test_schedule_with_priority(
        self, scheduler: IndexingScheduler, tmp_path: Path
    ) -> None:
        """Test scheduling with priority."""
        files = [tmp_path / "file.py"]

        task_id = await scheduler.schedule(files, priority=TaskPriority.HIGH)

        status = scheduler.get_task_status(task_id)
        assert status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_schedule_batch(
        self, scheduler: IndexingScheduler, tmp_path: Path
    ) -> None:
        """Test batch scheduling."""
        scheduler.config.batch_size = 10
        files = [tmp_path / f"file{i}.py" for i in range(25)]

        task_ids = await scheduler.schedule_batch(files)

        assert len(task_ids) == 3  # 25 files / 10 batch size = 3 batches

    @pytest.mark.asyncio
    async def test_cancel_task(
        self, scheduler: IndexingScheduler, tmp_path: Path
    ) -> None:
        """Test cancelling a task."""
        task_id = await scheduler.schedule([tmp_path / "file.py"])

        result = scheduler.cancel_task(task_id)

        assert result is True
        assert scheduler.get_task_status(task_id) == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_start_and_stop(self, scheduler: IndexingScheduler) -> None:
        """Test starting and stopping scheduler."""
        await scheduler.start()
        assert scheduler.is_running

        await scheduler.stop()
        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_worker_processes_tasks(
        self, scheduler: IndexingScheduler, tmp_path: Path
    ) -> None:
        """Test worker processes tasks."""
        processed = []

        async def index_func(files: list[Path]) -> dict:
            processed.extend(files)
            return {"count": len(files)}

        scheduler.set_index_function(index_func)

        # Start scheduler first
        await scheduler.start()
        # Give workers time to start
        await asyncio.sleep(0.1)

        file_path = tmp_path / "file.py"
        await scheduler.schedule([file_path])

        # Wait for task to be processed
        for _ in range(50):  # 5 seconds max
            if file_path in processed:
                break
            await asyncio.sleep(0.1)

        await scheduler.stop()

        assert file_path in processed

    @pytest.mark.asyncio
    async def test_callback_invoked(
        self, scheduler: IndexingScheduler, tmp_path: Path
    ) -> None:
        """Test callback is invoked on completion."""
        callback_results = []

        async def callback(result: dict) -> None:
            callback_results.append(result)

        async def index_func(files: list[Path]) -> dict:
            return {"files_processed": len(files)}

        scheduler.set_index_function(index_func)

        await scheduler.start()
        await asyncio.sleep(0.1)

        await scheduler.schedule(
            [tmp_path / "file.py"],
            callback=callback,
        )

        # Wait for task to be processed
        for _ in range(50):  # 5 seconds max
            if callback_results:
                break
            await asyncio.sleep(0.1)

        await scheduler.stop()

        assert len(callback_results) == 1

    @pytest.mark.asyncio
    async def test_task_failure(
        self, scheduler: IndexingScheduler, tmp_path: Path
    ) -> None:
        """Test handling task failure."""

        async def failing_func(files: list[Path]) -> dict:
            raise ValueError("Test error")

        scheduler.set_index_function(failing_func)

        await scheduler.start()
        await asyncio.sleep(0.1)

        task_id = await scheduler.schedule([tmp_path / "file.py"])

        # Wait for task to fail
        for _ in range(50):  # 5 seconds max
            if scheduler.get_task_status(task_id) == TaskStatus.FAILED:
                break
            await asyncio.sleep(0.1)

        await scheduler.stop()

        assert scheduler.get_task_status(task_id) == TaskStatus.FAILED
        assert scheduler.stats.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_wait_for_completion_no_tasks(
        self, scheduler: IndexingScheduler
    ) -> None:
        """Test wait_for_completion with no tasks."""
        await scheduler.start()

        result = await scheduler.wait_for_completion(timeout=0.1)

        await scheduler.stop()

        assert result is True  # No tasks means already complete

    def test_get_queue_status(self, scheduler: IndexingScheduler) -> None:
        """Test get_queue_status method."""
        status = scheduler.get_queue_status()

        assert "pending" in status
        assert "running" in status
        assert "completed" in status
        assert "failed" in status
        assert "is_idle" in status

    @pytest.mark.asyncio
    async def test_clear_completed(
        self, scheduler: IndexingScheduler, tmp_path: Path
    ) -> None:
        """Test clearing completed tasks."""

        async def index_func(files: list[Path]) -> dict:
            return {}

        scheduler.set_index_function(index_func)

        await scheduler.start()
        await asyncio.sleep(0.1)

        await scheduler.schedule([tmp_path / "file.py"])

        # Wait for task to complete
        for _ in range(50):  # 5 seconds max
            if scheduler.stats.tasks_completed > 0:
                break
            await asyncio.sleep(0.1)

        removed = scheduler.clear_completed()
        await scheduler.stop()

        assert removed >= 1


class TestSchedulerStats:
    """Tests for SchedulerStats."""

    def test_default_values(self) -> None:
        """Test default stat values."""
        stats = SchedulerStats()

        assert stats.tasks_scheduled == 0
        assert stats.tasks_completed == 0
        assert stats.tasks_failed == 0
        assert stats.tasks_cancelled == 0
        assert stats.files_indexed == 0

    def test_to_dict(self) -> None:
        """Test conversion to dict."""
        stats = SchedulerStats(
            tasks_scheduled=10,
            tasks_completed=8,
            tasks_failed=2,
            files_indexed=100,
        )

        d = stats.to_dict()

        assert d["tasks_scheduled"] == 10
        assert d["tasks_completed"] == 8
        assert d["tasks_failed"] == 2
        assert d["files_indexed"] == 100

    def test_success_rate(self) -> None:
        """Test success_rate calculation."""
        stats = SchedulerStats(tasks_completed=8, tasks_failed=2)
        assert stats.success_rate == 0.8

    def test_success_rate_no_tasks(self) -> None:
        """Test success_rate with no tasks."""
        stats = SchedulerStats()
        assert stats.success_rate == 1.0


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_under_limit(self) -> None:
        """Test acquire when under limit."""
        limiter = RateLimiter(rate=100.0)

        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be fast

    @pytest.mark.asyncio
    async def test_acquire_rate_limited(self) -> None:
        """Test acquire with rate limiting."""
        limiter = RateLimiter(rate=10.0)

        # Exhaust tokens
        for _ in range(10):
            await limiter.acquire()

        # Next acquire should be delayed
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        assert elapsed >= 0.05  # Should wait ~0.1s


class TestThrottledScheduler:
    """Tests for ThrottledScheduler."""

    @pytest.mark.asyncio
    async def test_throttled_scheduling(self, tmp_path: Path) -> None:
        """Test throttled task scheduling."""
        scheduler = ThrottledScheduler(max_rate=100.0)

        task_id = await scheduler.schedule([tmp_path / "file.py"])

        assert task_id is not None
        assert scheduler.pending_count == 1
