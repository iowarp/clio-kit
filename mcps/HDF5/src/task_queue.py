"""
Task queue implementation for asynchronous processing.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Generic
from datetime import datetime
import heapq
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Task result type
P = TypeVar('P')  # Task priority type

class TaskPriority(Enum):
    """Task priority levels."""
    HIGH = 0
    NORMAL = 1
    LOW = 2
    BATCH = 3

@dataclass
class Task(Generic[T]):
    """Represents an asynchronous task with metadata."""
    id: str
    coroutine: Callable[..., Awaitable[T]]
    args: tuple
    kwargs: dict
    priority: TaskPriority
    group_id: Optional[str] = None
    created_at: datetime = datetime.now()
    timeout: Optional[float] = None
    _future: Optional[asyncio.Future] = None
    
    def __lt__(self, other: Task) -> bool:
        """Compare tasks for priority queue."""
        if not isinstance(other, Task):
            return NotImplemented
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)

class TaskGroup:
    """Groups related tasks for batch processing."""
    def __init__(self, group_id: str):
        self.group_id = group_id
        self.tasks: List[Task] = []
        self.created_at = datetime.now()
        self._lock = asyncio.Lock()
        self._complete = asyncio.Event()
        self._results: Dict[str, Any] = {}
        
    async def add_task(self, task: Task):
        """Add a task to the group."""
        async with self._lock:
            task.group_id = self.group_id
            self.tasks.append(task)
    
    def mark_complete(self, task_id: str, result: Any):
        """Mark a task as complete with its result."""
        self._results[task_id] = result
        if len(self._results) == len(self.tasks):
            self._complete.set()
    
    async def wait(self, timeout: Optional[float] = None):
        """Wait for all tasks in the group to complete."""
        try:
            await asyncio.wait_for(self._complete.wait(), timeout)
            return self._results
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task group {self.group_id} timed out")

class AsyncTaskQueue:
    """Asynchronous task queue with priority and batch support."""
    def __init__(self, max_workers: int = 4, batch_size: int = 100, 
                 max_batch_wait: float = 0.5):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_batch_wait = max_batch_wait
        
        self._queue: List[Task] = []
        self._workers: List[asyncio.Task] = []
        self._groups: Dict[str, TaskGroup] = {}
        self._running = False
        self._lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        
    async def start(self):
        """Start the task queue processing."""
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(), name=f"Task-Worker-{i}")
            for i in range(self.max_workers)
        ]
        self._batch_processor_task = asyncio.create_task(self._batch_processor(), name="Task-BatchProcessor")
    
    async def stop(self):
        """Stop the task queue processing."""
        if not self._running:
            return  # Already stopped
        
        self._running = False
        self._batch_event.set()  # Signal batch processor
        
        # Cancel all workers and batch processor
        for worker in self._workers:
            if not worker.done():
                worker.cancel()
        
        if hasattr(self, '_batch_processor_task') and not self._batch_processor_task.done():
            self._batch_processor_task.cancel()
        
        # Wait for workers to complete
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Wait for batch processor
        if hasattr(self, '_batch_processor_task'):
            await asyncio.gather(self._batch_processor_task, return_exceptions=True)
        
        # Clear references
        self._workers = []
        self._batch_processor_task = None
    
    async def submit(self, coroutine: Callable[..., Awaitable[T]], 
                    *args, priority: TaskPriority = TaskPriority.NORMAL,
                    group_id: Optional[str] = None,
                    timeout: Optional[float] = None,
                    **kwargs) -> asyncio.Future[T]:
        """Submit a task to the queue."""
        task = Task(
            id=f"task_{len(self._queue)}",
            coroutine=coroutine,
            args=args,
            kwargs=kwargs,
            priority=priority,
            group_id=group_id,
            timeout=timeout
        )
        
        task._future = asyncio.Future()
        
        async with self._lock:
            if group_id:
                group = self._groups.get(group_id)
                if not group:
                    group = TaskGroup(group_id)
                    self._groups[group_id] = group
                await group.add_task(task)
                
                if len(group.tasks) >= self.batch_size:
                    self._batch_event.set()
            
            heapq.heappush(self._queue, task)
        
        return task._future
    
    @asynccontextmanager
    async def batch_context(self, group_id: str, timeout: Optional[float] = None):
        """Create a batch context for operations that should be processed together."""
        group = self._groups.setdefault(group_id, TaskGroup(group_id))
        try:
            yield group
        finally:
            if group_id in self._groups:
                try:
                    await group.wait(timeout)
                    del self._groups[group_id]
                except TimeoutError:
                    logger.error(f"Batch operation {group_id} timed out")
                    raise
    
    async def _worker(self):
        """Worker process that executes tasks."""
        while self._running:
            task = None
            try:
                async with self._lock:
                    if self._queue:
                        task = heapq.heappop(self._queue)
                
                if task:
                    try:
                        if task.timeout:
                            result = await asyncio.wait_for(
                                task.coroutine(*task.args, **task.kwargs),
                                timeout=task.timeout
                            )
                        else:
                            result = await task.coroutine(*task.args, **task.kwargs)
                        
                        if not task._future.done():
                            task._future.set_result(result)
                        
                        if task.group_id and task.group_id in self._groups:
                            self._groups[task.group_id].mark_complete(task.id, result)
                            
                    except Exception as e:
                        if not task._future.done():
                            task._future.set_exception(e)
                        logger.error(f"Task {task.id} failed: {str(e)}")
                else:
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
    
    async def _batch_processor(self):
        """Process batched tasks."""
        while self._running:
            try:
                # Wait for batch event or timeout
                try:
                    await asyncio.wait_for(
                        self._batch_event.wait(),
                        timeout=self.max_batch_wait
                    )
                except asyncio.TimeoutError:
                    pass
                
                self._batch_event.clear()
                
                # Process completed batches
                async with self._lock:
                    completed_groups = []
                    for group_id, group in self._groups.items():
                        if (len(group.tasks) >= self.batch_size or
                            (datetime.now() - group.created_at).total_seconds() >= self.max_batch_wait):
                            completed_groups.append(group_id)
                    
                    # Remove completed groups
                    for group_id in completed_groups:
                        if group_id in self._groups:
                            del self._groups[group_id]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {str(e)}")
                await asyncio.sleep(1)  # Avoid tight loop on error

# Export AsyncTaskQueue as TaskQueue for backward compatibility
TaskQueue = AsyncTaskQueue 