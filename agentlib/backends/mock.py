import time
import threading
from typing import Optional, List
from ..model_backend import ModelBackend
from ..context import ChatContext, TurnOutput
from ..skills_tools import Tool

class MockBackend(ModelBackend):
    """Mock model backend to test queue merging, priority locking, retries, and tool execution loops."""
    
    def __init__(self, responses: List[TurnOutput], delay: float = 0.0, max_concurrency: int = -1, **kwargs):
        super().__init__(max_concurrency=max_concurrency)
        self.responses = responses
        self.delay = delay
        self.call_count = 0
        self.lock = threading.Lock()

    from typing import Callable

    def _chat_impl(
        self,
        context: ChatContext,
        tools: Optional[List[Tool]] = None,
        on_stream_chunk: Optional[Callable[[Optional[str], Optional[str]], None]] = None,
    ) -> TurnOutput:
        """Simulates response extraction and delay in a thread-safe manner."""
        if self.delay > 0:
            time.sleep(self.delay)
            
        with self.lock:
            if not self.responses:
                resp = TurnOutput(text="Mock response placeholder")
            else:
                resp = self.responses[self.call_count % len(self.responses)]
                self.call_count += 1
            
            if on_stream_chunk:
                if resp.thought:
                    on_stream_chunk(None, resp.thought)
                if resp.text:
                    on_stream_chunk(resp.text, None)
                    
            return resp
