import os
import sys
import time
import datetime
import math
import threading
from PIL import Image

from apikeys import googleai_key
from agentlib import (
    Agent,
    AgentCallbacks,
    GoogleBackend,
    Tool,
    TurnInput,
    TurnOutput,
    ImageInput
)

# 1. Define two simple tools
def get_system_time() -> str:
    """Returns the current system date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """
    Evaluates a basic mathematical expression safely.
    Supported functions: sqrt, sin, cos, tan, pi, e
    """
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        # Basic sanitization
        sanitized = expression.replace("__", "").replace("import", "")
        result = eval(sanitized, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

# 2. Build the Tool objects
time_tool = Tool.from_function(get_system_time)
calc_tool = Tool.from_function(calculate)

# 3. Create the callbacks to track intermediate steps
class ConsoleChatCallbacks(AgentCallbacks):
    def __init__(self, finished_event: threading.Event):
        self.finished_event = finished_event

    def on_input_received(self, turn_input: TurnInput) -> None:
        print("\n=== [Event: Input Queued] ===")
        if turn_input.perm_text:
            print(f" > Perm Text: {turn_input.perm_text}")
        if turn_input.imgs:
            print(f" > Attached Images: {len(turn_input.imgs)}")

    def on_input_processing(self, turn_input: TurnInput) -> TurnInput:
        print("=== [Event: Input Processing (RAG hook)] ===")
        # We can add temporary context here for demonstration
        turn_input.temp_text.append("[System Inject: User is running in interactive console mode]")
        return turn_input

    def on_context_updated(self, context, update_type: str, data: Any) -> None:
        print(f"=== [Event: Context Updated - {update_type}] ===")
        if update_type == "user_input":
            if data.perm_text:
                print(f" > User Text: {data.perm_text}")
            if data.imgs:
                print(f" > Attached Images: {len(data.imgs)}")
        elif update_type == "model_response":
            if data.thought:
                print(f" > Model Thought:\n{data.thought}")
            if data.text:
                print(f" > Model Response Text:\n{data.text}")
        elif update_type == "tool_calls":
            for call in data:
                print(f" > Model Requesting Tool: '{call.name}' with args: {call.args}")
        elif update_type == "tool_responses":
            for resp in data:
                print(f" > Tool execution output: '{resp.name}' -> {resp.content}")

    def on_final_output(self, turn_output: TurnOutput) -> None:
        print("=== [Event: Final Output Ready] ===")
        print(f"\nAssistant: {turn_output.text}")
        # Unblock the main input thread
        self.finished_event.set()


def main():
    print("Initializing Google Gemini Agent Backend...")
    # Initialize the backend using the specified model
    backend = GoogleBackend(
        api_key=googleai_key,
        model_name="gemini-3.1-flash-lite"
    )

    finished_event = threading.Event()
    callbacks = ConsoleChatCallbacks(finished_event)

    # Initialize the Agent
    agent = Agent(
        sys_prompt="You are a helpful assistant. Use the tools provided when appropriate.",
        backend=backend,
        tools=[time_tool, calc_tool],
        callbacks=callbacks
    )

    print("\n==============================================")
    print("Welcome to the AgentLib Interactive Chat CLI!")
    print("Commands:")
    print("  /image <path> [prompt] - Load an image and optional prompt text")
    print("  /exit or /quit         - Exit the chat")
    print("==============================================")

    try:
        while True:
            # Wait for any background execution to complete
            finished_event.set() 
            
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit"):
                break

            finished_event.clear()

            # Handle /image command
            if user_input.startswith("/image "):
                parts = user_input.split(" ", 2)
                if len(parts) < 2:
                    print("Usage: /image <path_to_image> [prompt]")
                    finished_event.set()
                    continue
                
                img_path = parts[1]
                prompt_text = parts[2] if len(parts) > 2 else "Describe this image."
                
                if not os.path.exists(img_path):
                    print(f"Error: File '{img_path}' does not exist.")
                    finished_event.set()
                    continue

                try:
                    img = Image.open(img_path)
                    turn_input = TurnInput(
                        imgs=[ImageInput(image=img)],
                        perm_text=[prompt_text]
                    )
                    agent.send_input(turn_input)
                except Exception as e:
                    print(f"Error loading image: {e}")
                    finished_event.set()
                    
            else:
                # Regular text input
                turn_input = TurnInput(perm_text=[user_input])
                agent.send_input(turn_input)

            # Block input prompt until model responses and tools execution complete
            finished_event.wait()

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        agent.stop()
        print("Agent thread stopped. Goodbye!")

if __name__ == "__main__":
    main()
