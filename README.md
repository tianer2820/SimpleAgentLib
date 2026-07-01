# AgentLib

A highly customizable, thread-safe, general-purpose Python agent framework with strict multi-turn chat context structures, tool call loop execution, priority-queue concurrency management, and robust event callbacks.

## Features

- **Strict Context Abstraction**: Manages multi-turn conversation context including images (via PIL), temporary RAG/context text, permanent user message texts, and nested tool calls/responses.
- **Priority Concurrency Limits**: Limit the max concurrent requests to backends. Blocks extra calls and processes them in order of priority (lower priority integers wake up first).
- **Callback Interfaces**: Dynamic hook hooks to intercept and mutate inputs (e.g. for custom RAG), watch context updates, or catch final turn outputs.
- **Automatic Tool Conversion**: Simply pass any python function and `Tool.from_function` generates the JSON schema parameter description dynamically using type hints and docstrings.
- **Modular Backends**: Built-in support for OpenAI and Google Gemini APIs alongside a Mock simulator for offline test execution.

---

## Installation

Install core dependencies (requires Pillow):
```bash
pip install -r requirements.txt
```

---

## Quick Start

See [basic_agent.py](file:///home/toby/Downloads/handtest/AgentLib/examples/basic_agent.py) for a runnable setup. Below is a simplified integration example:

```python
from agentlib import Agent, Tool, MockBackend, TurnInput, TurnOutput, ToolCall

# 1. Define custom tools
def calculate_tax(amount: float, rate: float = 0.15) -> float:
    """Calculates tax for a given amount and rate."""
    return amount * rate

# Convert Python function to a Tool schema automatically
tax_tool = Tool.from_function(calculate_tax)

# 2. Setup a backend (e.g., Mock or OpenAI/Google)
backend = MockBackend(responses=[
    TurnOutput(
        text="Processing transaction.",
        tool_calls=[ToolCall(id="tc_1", name="calculate_tax", args={"amount": 100.0})]
    ),
    TurnOutput(text="Tax calculated successfully.")
])

# 3. Initialize Agent
agent = Agent(
    sys_prompt="You are a financial advisor assistant.",
    backend=backend,
    tools=[tax_tool],
    terminating_tool_calls=["calculate_tax"]
)

# 4. Dispatch Input
agent.send_input(TurnInput(perm_text=["Please check the tax for 100 dollars."]))

# Stop the loop thread gracefully when complete
agent.stop()
```

---

## Developer Guide

### Callbacks
Customize your agent logic by subclassing `AgentCallbacks`:
- `on_input_received`: Fired immediately when `send_input` is called.
- `on_input_processing`: Fired right before text is formatted for backend prompt generation. Ideal place for custom RAG injection (returns updated input).
- `on_context_updated`: Triggers whenever inputs are merged, tool requests are generated, or tool results are added.
- `on_final_output`: Triggered when model completes the execution loop and returns final output text.

### Testing
Verify all functionality with:
```bash
python -m unittest tests/test_agent.py
```
