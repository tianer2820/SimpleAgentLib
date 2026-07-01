from .context import (
    ImageInput,
    ToolCall,
    ToolResponse,
    TurnInput,
    TurnOutput,
    ChatTurn,
    ChatContext,
)
from .skills_tools import Tool, Skill, discover_skills
from .extended_agent import ExtendedAgent
from .model_backend import ModelBackend, PriorityConcurrencyLock
from .agent import Agent, AgentCallbacks
from .backends.mock import MockBackend
from .essential_tools import WorkspaceTools, WebTools

# Conditional/lazy imports for GoogleBackend
try:
    from .backends.google import GoogleBackend
except (ImportError, ModuleNotFoundError):
    class GoogleBackend:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GoogleBackend is unavailable because the 'google-genai' package is not installed. "
                "Install it using: pip install google-genai"
            )

# Conditional/lazy imports for OpenAIBackend
try:
    from .backends.openai import OpenAIBackend
except (ImportError, ModuleNotFoundError):
    class OpenAIBackend:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "OpenAIBackend is unavailable because the 'openai' package is not installed. "
                "Install it using: pip install openai"
            )

__all__ = [
    "ImageInput",
    "ToolCall",
    "ToolResponse",
    "TurnInput",
    "TurnOutput",
    "ChatTurn",
    "ChatContext",
    "Tool",
    "Skill",
    "discover_skills",
    "ModelBackend",
    "PriorityConcurrencyLock",
    "Agent",
    "ExtendedAgent",
    "AgentCallbacks",
    "GoogleBackend",
    "OpenAIBackend",
    "MockBackend",
    "WorkspaceTools",
    "WebTools",
]
