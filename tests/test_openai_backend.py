import unittest
import threading
import time
from PIL import Image
from apikeys import openai_key
from agentlib import (
    Agent,
    AgentCallbacks,
    OpenAIBackend,
    Tool,
    TurnInput,
    ImageInput
)

class TestOpenAIBackend(unittest.TestCase):
    
    def setUp(self):
        self.backend = OpenAIBackend(
            api_key=openai_key,
            model_name="gpt-5.5",
            base_url='http://127.0.0.1:8080/v1/',
        )

    def test_regular_conversation(self):
        class WaitCallback(AgentCallbacks):
            def __init__(self):
                self.event = threading.Event()
                self.output = None
                
            def on_final_output(self, turn_output):
                self.output = turn_output
                self.event.set()

        cb = WaitCallback()
        agent = Agent(
            sys_prompt="You are a helpful assistant.",
            backend=self.backend,
            callbacks=cb
        )

        agent.send_input(TurnInput(perm_text=["Respond with only the word 'Hello'."]))
        
        # Wait up to 30 seconds for OpenAI response
        self.assertTrue(cb.event.wait(timeout=30))
        agent.stop()

        self.assertIsNotNone(cb.output)
        self.assertIsNotNone(cb.output.text)
        self.assertIn("hello", cb.output.text.lower())

    def test_tool_call(self):
        def add_numbers(a: int, b: int) -> int:
            """Adds two numbers a and b."""
            return a + b

        math_tool = Tool.from_function(add_numbers)

        class WaitCallback(AgentCallbacks):
            def __init__(self):
                self.event = threading.Event()
                self.output = None
                self.tool_calls_received = []
                self.tool_responses_received = []

            def on_context_updated(self, context, update_type, data):
                if update_type == "tool_calls":
                    self.tool_calls_received.extend(data)
                elif update_type == "tool_responses":
                    self.tool_responses_received.extend(data)

            def on_final_output(self, turn_output):
                self.output = turn_output
                self.event.set()

        cb = WaitCallback()
        agent = Agent(
            sys_prompt="Calculate using the tools provided. Respond with the final answer.",
            backend=self.backend,
            tools=[math_tool],
            callbacks=cb
        )

        agent.send_input(TurnInput(perm_text=["Calculate 1324 + 5678."]))
        
        self.assertTrue(cb.event.wait(timeout=30))
        agent.stop()

        self.assertIsNotNone(cb.output)
        # Verify tool calls were triggered and matched our parameters
        self.assertTrue(len(cb.tool_calls_received) > 0)
        self.assertEqual(cb.tool_calls_received[0].name, "add_numbers")
        self.assertEqual(cb.tool_calls_received[0].args, {"a": 1324, "b": 5678})
        # Verify tool responses were returned
        self.assertTrue(len(cb.tool_responses_received) > 0)
        self.assertEqual(cb.tool_responses_received[0].content, "7002")

    def test_image_input(self):
        class WaitCallback(AgentCallbacks):
            def __init__(self):
                self.event = threading.Event()
                self.output = None
                
            def on_final_output(self, turn_output):
                self.output = turn_output
                self.event.set()

        cb = WaitCallback()
        agent = Agent(
            sys_prompt="Identify what is in the image.",
            backend=self.backend,
            callbacks=cb
        )

        # Open test image
        img = Image.open("tests/example_img.jpg")
        image_input = ImageInput(image=img)

        agent.send_input(TurnInput(
            imgs=[image_input],
            perm_text=["What is this object or scene in this picture? Limit answer to 5 words."]
        ))
        
        self.assertTrue(cb.event.wait(timeout=30))
        agent.stop()

        self.assertIsNotNone(cb.output)
        self.assertIsNotNone(cb.output.text)
        self.assertTrue(len(cb.output.text) > 0)

if __name__ == "__main__":
    unittest.main()
