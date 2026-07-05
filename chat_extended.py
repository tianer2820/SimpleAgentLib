import os
import sys
import time
import datetime
import math
import argparse
import threading
from typing import Any, Union
from PIL import Image

from apikeys import googleai_key, tavily_key
from agentlib import (
    ExtendedAgent,
    AgentCallbacks,
    GoogleBackend,
    OpenAIBackend,
    Tool,
    TurnInput,
    TurnOutput,
    ImageInput,
    discover_skills,
    WorkspaceTools,
    WebTools,
)

# 1. Define two simple tools (reused from chat.py)
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

# Build the Tool objects
time_tool = Tool.from_function(get_system_time)
calc_tool = Tool.from_function(calculate)

# 2. Callbacks to track intermediate steps
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


def parse_args():
    parser = argparse.ArgumentParser(description="Extended Agent Interactive Chat CLI")
    parser.add_argument(
        "--rules", "-r",
        help="Rules to pass to the agent, either a comma-separated list or a path to a file containing rules (one per line)."
    )
    parser.add_argument(
        "--workspace", "-w",
        default=".",
        help="Path to the workspace root directory. Confines file operations here. Skills are loaded from <workspace>/.agent."
    )
    parser.add_argument(
        "--max-context-length", "-m",
        type=int,
        help="Maximum conversation context length (in turns) before compression."
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite",
        help="Google Gemini model name to use."
    )
    parser.add_argument(
        "--workspace-type",
        choices=["local", "podman"],
        default="local",
        help="Workspace environment type (default: local)."
    )
    parser.add_argument(
        "--podman-container",
        default=None,
        help="Podman container name or ID."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Process rules argument
    rules = []
    if args.rules:
        if os.path.exists(args.rules):
            try:
                with open(args.rules, "r", encoding="utf-8") as f:
                    rules = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error reading rules file: {e}")
                sys.exit(1)
        else:
            rules = [r.strip() for r in args.rules.split(",") if r.strip()]

    # Process workspace and skills
    temp_workspace_dir = None
    workspace_for_skills = args.workspace
    if args.workspace_type == "podman":
        workspace = args.workspace
        # Copy container .agent to host temp folder for skill discovery
        import subprocess
        import tempfile
        import shutil
        agent_dir_in_container = f"{workspace}/.agent"
        res = subprocess.run(
            ["podman", "exec", args.podman_container, "test", "-d", agent_dir_in_container],
            capture_output=True
        )
        if res.returncode == 0:
            temp_dir = tempfile.mkdtemp()
            temp_workspace_dir = temp_dir
            cp_cmd = ["podman", "cp", f"{args.podman_container}:{agent_dir_in_container}", temp_dir]
            res_cp = subprocess.run(cp_cmd, capture_output=True)
            if res_cp.returncode == 0:
                workspace_for_skills = temp_dir
            else:
                shutil.rmtree(temp_dir)
                temp_workspace_dir = None
                workspace_for_skills = ""
    else:
        workspace = os.path.abspath(args.workspace)
        workspace_for_skills = workspace

    skills = []
    skills_dir = ""
    if workspace_for_skills:
        skills_dir = os.path.join(workspace_for_skills, ".agent")
        if os.path.isdir(skills_dir):
            skills = discover_skills(skills_dir)
            if args.workspace_type == "podman" and temp_workspace_dir:
                import posixpath
                for skill in skills:
                    if skill.path and skill.path.startswith(temp_workspace_dir):
                        rel_path = os.path.relpath(skill.path, temp_workspace_dir)
                        skill.path = posixpath.normpath(posixpath.join(workspace, rel_path))
            
    if temp_workspace_dir:
        import shutil
        shutil.rmtree(temp_workspace_dir)

    # Setup tools confirmation callback for exec_cmd
    def confirm_callback(cmd: str) -> Union[bool, str]:
        try:
            sys.stdout.flush()
            answer = input(f"\n[Confirm Execution] Run command: '{cmd}'?\nConfirm (y/n) or enter feedback for the model: ").strip()
            answer_lower = answer.lower()
            if answer_lower in ("y", "yes"):
                return True
            elif answer_lower in ("n", "no") or not answer:
                return False
            else:
                return answer
        except Exception:
            return False

    # Initialize Workspace and Web Tools
    ws_tools = WorkspaceTools(
        workspace_root=workspace,
        sandbox_enabled=True,
        confirm_cmd_callback=confirm_callback,
        workspace_type=args.workspace_type,
        podman_container=args.podman_container
    )
    web_tools = WebTools(api_key=tavily_key)
    
    agent_tools = [time_tool, calc_tool] + ws_tools.get_tools() + web_tools.get_tools()

    print("Initializing Google Gemini Agent Backend...")
    # backend = GoogleBackend(
    #     api_key=googleai_key,
    #     model_name=args.model
    # )

    backend = OpenAIBackend(
        api_key="no key needed",
        model_name=args.model,
        base_url="http://127.0.0.1:8080/v1",
    )

    finished_event = threading.Event()
    callbacks = ConsoleChatCallbacks(finished_event)

    # Initialize the ExtendedAgent
    agent = ExtendedAgent(
        sys_prompt="You are a helpful assistant. Use the tools and skills provided when appropriate.",
        backend=backend,
        tools=agent_tools,
        callbacks=callbacks,
        rules=rules,
        skills=skills,
        max_context_length=args.max_context_length
    )

    print("\n==============================================")
    print("Welcome to the Extended Agent Interactive Chat CLI!")
    print(f"Active Model: {args.model}")
    print(f"Workspace Root: {workspace}")
    if rules:
        print(f"Loaded {len(rules)} system rules.")
    if skills:
        print(f"Discovered {len(skills)} skills from '{skills_dir}'.")
    if args.max_context_length:
        print(f"Max Context Length set to {args.max_context_length} turns.")
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
