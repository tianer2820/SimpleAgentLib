import threading
import heapq
import time
import contextlib
from typing import Optional, List, Any
from .context import ChatContext, TurnOutput
from .skills_tools import Tool

class PriorityWaiter:
    """Helper representing a waiting thread with a specific priority and sequence number."""
    def __init__(self, priority: int, seq: int):
        self.priority = priority
        self.seq = seq
        self.event = threading.Event()

    def __lt__(self, other: "PriorityWaiter") -> bool:
        if self.priority == other.priority:
            return self.seq < other.seq
        return self.priority < other.priority


class PriorityConcurrencyLock:
    """
    Locking mechanism that enforces a maximum concurrency limit
    and wakes up waiting threads in order of priority (lower priority integers first).
    """
    def __init__(self, max_concurrency: int):
        self.max_concurrency = max_concurrency
        self.active_count = 0
        self.lock = threading.Lock()
        self.waiters: List[PriorityWaiter] = []
        self.seq = 0

    def acquire(self, priority: int) -> None:
        """Acquires a concurrency slot. Blocks if max_concurrency is reached."""
        if self.max_concurrency == -1:
            return

        waiter = None
        with self.lock:
            if self.active_count < self.max_concurrency:
                self.active_count += 1
                return
            
            self.seq += 1
            waiter = PriorityWaiter(priority, self.seq)
            heapq.heappush(self.waiters, waiter)

        # Block until notified
        waiter.event.wait()

    def release(self) -> None:
        """Releases a concurrency slot and wakes the highest-priority waiter."""
        if self.max_concurrency == -1:
            return

        with self.lock:
            if self.waiters:
                next_waiter = heapq.heappop(self.waiters)
                next_waiter.event.set()
                # active_count remains self.max_concurrency since the slot is transferred
            else:
                self.active_count -= 1

    @contextlib.contextmanager
    def limit_concurrency(self, priority: int):
        """Context manager to acquire and release the lock safely."""
        self.acquire(priority)
        try:
            yield
        finally:
            self.release()


class ModelBackend:
    """Abstract base class for all LLM backends, implementing retries and concurrency control."""
    def __init__(self, max_concurrency: int = -1, **kwargs):
        self.max_concurrency = max_concurrency
        self.concurrency_lock = PriorityConcurrencyLock(max_concurrency)

    def chat(
        self,
        context: ChatContext,
        max_retry: int = 3,
        retry_delay: float = 1.0,
        priority: int = 0,
        tools: Optional[List[Tool]] = None,
    ) -> TurnOutput:
        """
        Sends the context to the model with concurrency controls and retry backoffs.
        Fills the last conversation turn's output inplace and returns it.
        """
        with self.concurrency_lock.limit_concurrency(priority):
            for attempt in range(max_retry + 1):
                try:
                    turn_output = self._chat_impl(context, tools=tools)
                    if context.conversations:
                        context.conversations[-1].turn_output = turn_output
                    return turn_output
                except Exception as e:
                    if attempt == max_retry:
                        raise e
                    time.sleep(retry_delay)
            # Should not reach here, but just in case
            raise Exception("Max retries exceeded")

    def _chat_impl(self, context: ChatContext, tools: Optional[List[Tool]] = None) -> TurnOutput:
        """Concrete backend execution logic. Must be overridden by subclasses."""
        raise NotImplementedError
