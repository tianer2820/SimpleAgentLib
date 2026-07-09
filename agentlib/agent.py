import queue
import threading
import collections
from typing import List, Optional, Any, Callable
from .context import ChatContext, ChatTurn, TurnInput, TurnOutput, ToolCall, ToolResponse
from .skills_tools import Tool
from .model_backend import ModelBackend

class AgentCallbacks:
    """Interface to handle key state-change hooks during agent execution."""
    
    def on_input_received(self, turn_input: TurnInput) -> None:
        """Called immediately when an input is queued via send_input."""
        pass

    def on_input_processing(self, turn_input: TurnInput) -> TurnInput:
        """Called before sending input to backend. Return a modified TurnInput for RAG."""
        return turn_input

    def on_context_updated(self, context: ChatContext, update_type: str, data: Any) -> None:
        """
        Called when context is modified.
        update_type options: 'user_input', 'tool_responses', 'model_response', 'tool_calls'
        """
        pass

    def on_final_output(self, turn_output: TurnOutput) -> None:
        """Called when the tool execution loop finishes and the final response is ready."""
        pass

    def on_error(self, error: Exception) -> None:
        """Called when a background processing thread encounters an error."""
        pass

    def on_stream_chunk(self, text: Optional[str], thought: Optional[str]) -> None:
        """Called when a stream chunk is received from the LLM backend."""
        pass


class AgentQueue:
    """Thread-safe queue with prepend capability to facilitate text merging."""
    def __init__(self):
        self.queue = collections.deque()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def put(self, item: Optional[TurnInput]) -> None:
        """Appends item to the queue."""
        with self.lock:
            self.queue.append(item)
            self.cond.notify()

    def put_first(self, item: Optional[TurnInput]) -> None:
        """Prepends item to the queue."""
        with self.lock:
            self.queue.appendleft(item)
            self.cond.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[TurnInput]:
        """Blocks until an item is retrieved or timeout is met."""
        with self.lock:
            while not self.queue:
                if not self.cond.wait(timeout):
                    return None
            return self.queue.popleft()

    def get_nowait(self) -> TurnInput:
        """Retrieves item without blocking. Raises queue.Empty if queue is empty."""
        with self.lock:
            if self.queue:
                return self.queue.popleft()
            raise queue.Empty()


class Agent:
    """Central agent manager that handles input queues, text merging, callbacks, and tool loops."""
    
    def __init__(
        self,
        sys_prompt: str,
        backend: ModelBackend,
        tools: Optional[List[Tool]] = None,
        callbacks: Optional[AgentCallbacks] = None,
        terminating_tool_calls: Optional[List[str]] = None,
    ):
        self.context = ChatContext(system_prompt=sys_prompt)
        self.backend = backend
        self.tools = tools or []
        self.callbacks = callbacks
        self.terminating_tool_calls = terminating_tool_calls or []
        
        self.queue = AgentQueue()
        self.error = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def send_input(self, turn_input: TurnInput) -> None:
        """Pushes a new TurnInput to the agent processing queue."""
        self._trigger_callback("on_input_received", turn_input)
        self.queue.put(turn_input)

    def stop(self) -> None:
        """Stops the background execution thread."""
        self._stop_event.set()
        self.queue.put(None)  # Wake up the queue thread to exit
        if self._thread.is_alive():
            self._thread.join()

    def _trigger_callback(self, name: str, *args, **kwargs) -> Any:
        """Invokes callback method if registered, with generic safe-guards."""
        if self.callbacks and hasattr(self.callbacks, name):
            method = getattr(self.callbacks, name)
            return method(*args, **kwargs)
        if name == "on_input_processing":
            return args[0]
        return None

    def _is_pure_text(self, turn_input: TurnInput) -> bool:
        """Helper to determine if an input contains only permanent text."""
        return (
            bool(turn_input.perm_text) and
            not turn_input.imgs and
            not turn_input.temp_text and
            not turn_input.tool_responses
        )

    def _find_tool(self, name: str) -> Optional[Tool]:
        """Finds a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def _run_loop(self) -> None:
        """Main processing thread loop."""
        try:
            while not self._stop_event.is_set():
                turn_input = self.queue.get()
                if turn_input is None:
                    break

                # 1. Consolidate consecutive pure text inputs
                if self._is_pure_text(turn_input):
                    while True:
                        try:
                            next_input = self.queue.get_nowait()
                            if next_input is None:
                                break
                            if self._is_pure_text(next_input):
                                turn_input.perm_text.extend(next_input.perm_text)
                            else:
                                self.queue.put_first(next_input)
                                break
                        except queue.Empty:
                            break

                # 2. Trigger on_input_processing callback (RAG)
                turn_input = self._trigger_callback("on_input_processing", turn_input)

                # 3. Add input to context turn
                turn = ChatTurn(turn_input=turn_input)
                self.context.conversations.append(turn)
                self._trigger_callback("on_context_updated", self.context, "user_input", turn_input)

                # 4. Multi-turn tool loops
                while True:
                    # Call model backend (fills in turn_output inplace)
                    def stream_cb(text_chunk: Optional[str], thought_chunk: Optional[str]) -> None:
                        self._trigger_callback("on_stream_chunk", text_chunk, thought_chunk)

                    turn_output = self.backend.chat(
                        context=self.context,
                        tools=self.tools,
                        on_stream_chunk=stream_cb
                    )
                    
                    # Only trigger model_response update if the response contains text/thoughts
                    if turn_output.text or turn_output.thought:
                        self._trigger_callback("on_context_updated", self.context, "model_response", turn_output)

                    if not turn_output.tool_calls:
                        break

                    self._trigger_callback("on_context_updated", self.context, "tool_calls", turn_output.tool_calls)

                    tool_responses = []
                    should_terminate = False

                    for call in turn_output.tool_calls:
                        tool = self._find_tool(call.name)
                        if tool:
                            try:
                                res = tool(**call.args)
                                content = str(res)
                            except Exception as e:
                                content = f"Error executing tool: {e}"
                        else:
                            content = f"Error: Tool '{call.name}' not found."

                        tool_responses.append(ToolResponse(
                            tool_call_id=call.id,
                            name=call.name,
                            content=content
                        ))

                        if call.name in self.terminating_tool_calls:
                            should_terminate = True

                    self._trigger_callback("on_context_updated", self.context, "tool_responses", tool_responses)

                    # Add tool response as a new context turn
                    new_input = TurnInput(tool_responses=tool_responses)
                    new_turn = ChatTurn(turn_input=new_input)
                    self.context.conversations.append(new_turn)

                    if should_terminate:
                        break

                # 5. Output completed turn
                self._trigger_callback("on_final_output", self.context.conversations[-1].turn_output)
        except Exception as e:
            self.error = e
            self._trigger_callback("on_error", e)
