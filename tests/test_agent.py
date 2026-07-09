import time
import unittest
import threading
from agentlib import (
    Agent,
    AgentCallbacks,
    MockBackend,
    Tool,
    TurnInput,
    TurnOutput,
    ToolCall,
    ToolResponse
)
from agentlib.model_backend import PriorityConcurrencyLock

class TestAgentLib(unittest.TestCase):
    
    def test_tool_from_function(self):
        def add(x: int, y: int = 10) -> int:
            """Adds x and y."""
            return x + y
            
        tool = Tool.from_function(add)
        self.assertEqual(tool.name, "add")
        self.assertEqual(tool.description, "Adds x and y.")
        self.assertEqual(tool.parameters["properties"]["x"]["type"], "integer")
        self.assertEqual(tool.parameters["properties"]["y"]["type"], "integer")
        self.assertIn("x", tool.parameters["required"])
        self.assertNotIn("y", tool.parameters.get("required", []))

    def test_priority_concurrency_lock(self):
        lock = PriorityConcurrencyLock(max_concurrency=1)
        acquired_order = []

        # Hold slot 1
        lock.acquire(priority=10)
        
        def worker(priority, name):
            lock.acquire(priority=priority)
            acquired_order.append(name)
            time.sleep(0.01)
            lock.release()

        # Thread B has low priority (5)
        t_b = threading.Thread(target=worker, args=(5, "B"))
        # Thread C has high priority (1)
        t_c = threading.Thread(target=worker, args=(1, "C"))

        t_b.start()
        time.sleep(0.02)  # Let thread B enter the queue
        t_c.start()
        time.sleep(0.02)  # Let thread C enter the queue

        # Release the lock holding slot 1. This should wake the highest priority (C) first.
        lock.release()

        t_b.join()
        t_c.join()

        self.assertEqual(acquired_order, ["C", "B"])

    def test_text_merging(self):
        responses = [TurnOutput(text="Done")]
        backend = MockBackend(responses=responses, delay=0.1)
        agent = Agent(sys_prompt="System", backend=backend)
        
        # Send two pure text inputs in rapid succession
        agent.send_input(TurnInput(perm_text=["First"]))
        agent.send_input(TurnInput(perm_text=["Second"]))
        
        time.sleep(0.2)
        agent.stop()
        
        # Ensure they were combined into a single turn input
        conversations = agent.context.conversations
        self.assertEqual(len(conversations), 1)
        self.assertEqual(conversations[0].turn_input.perm_text, ["First", "Second"])

    def test_tool_calling_loop_and_termination(self):
        def get_weather(location: str) -> str:
            return f"Sunny in {location}"

        def trigger_alert(msg: str) -> str:
            return f"Alert: {msg}"

        weather_tool = Tool.from_function(get_weather)
        alert_tool = Tool.from_function(trigger_alert)

        # Preset outputs representing sequential calls
        responses = [
            # Call 1: model requests weather
            TurnOutput(tool_calls=[ToolCall(id="c1", name="get_weather", args={"location": "Paris"})]),
            # Call 2: model requests alert (terminating tool)
            TurnOutput(tool_calls=[ToolCall(id="c2", name="trigger_alert", args={"msg": "High temperature Alert!"})]),
            # Call 3: would return text, but should be bypassed
            TurnOutput(text="This should not be reached.")
        ]
        backend = MockBackend(responses=responses)
        
        agent = Agent(
            sys_prompt="System",
            backend=backend,
            tools=[weather_tool, alert_tool],
            terminating_tool_calls=["trigger_alert"]
        )
        
        agent.send_input(TurnInput(perm_text=["Start check"]))
        time.sleep(0.2)
        agent.stop()

        # The loop should terminate after executing trigger_alert, leaving only 3 turns:
        # Turn 1: user_input -> returns weather call
        # Turn 2: weather response -> returns alert call
        # Turn 3: alert response (and early break)
        self.assertEqual(len(agent.context.conversations), 3)
        self.assertEqual(backend.call_count, 2)

    def test_callbacks(self):
        responses = [TurnOutput(text="Hello")]
        backend = MockBackend(responses=responses)
        events = []

        class TestCallbacks(AgentCallbacks):
            def on_input_received(self, turn_input):
                events.append("received")
                
            def on_input_processing(self, turn_input):
                events.append("processing")
                turn_input.temp_text.append("RAG Context Added")
                return turn_input
                
            def on_context_updated(self, context, update_type, data):
                events.append(f"updated_{update_type}")
                
            def on_final_output(self, turn_output):
                events.append("final")

        agent = Agent(
            sys_prompt="System Prompt",
            backend=backend,
            callbacks=TestCallbacks()
        )
        
        agent.send_input(TurnInput(perm_text=["Hello"]))
        time.sleep(0.1)
        agent.stop()

        expected_sequence = [
            "received",
            "processing",
            "updated_user_input",
            "updated_model_response",
            "final"
        ]
        self.assertEqual(events, expected_sequence)
        # Verify RAG context mutation
        self.assertEqual(agent.context.conversations[0].turn_input.temp_text, ["RAG Context Added"])

    def test_streaming_callback(self):
        responses = [TurnOutput(thought="Thinking...", text="Hello, world!")]
        backend = MockBackend(responses=responses)
        streamed_chunks = []

        class TestStreamingCallbacks(AgentCallbacks):
            def on_stream_chunk(self, text, thought):
                streamed_chunks.append((text, thought))

        agent = Agent(
            sys_prompt="System Prompt",
            backend=backend,
            callbacks=TestStreamingCallbacks()
        )
        
        agent.send_input(TurnInput(perm_text=["Hi"]))
        time.sleep(0.1)
        agent.stop()

        self.assertEqual(streamed_chunks, [
            (None, "Thinking..."),
            ("Hello, world!", None)
        ])

if __name__ == "__main__":
    unittest.main()
