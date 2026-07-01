from dataclasses import dataclass, field
from typing import List, Optional, Any
from PIL import Image

@dataclass
class ImageInput:
    """Wrapper for PIL Image object."""
    image: Image.Image

# Note: AudioInput is left unimplemented as per instructions.

@dataclass
class ToolCall:
    """Represents a request from the model to execute a tool."""
    id: str
    name: str
    args: dict
    thought_signature: Optional[Any] = None

@dataclass
class ToolResponse:
    """Represents the output from executing a tool."""
    tool_call_id: str
    name: str
    content: str

@dataclass
class TurnInput:
    """Input parameters for a single chat turn."""
    imgs: List[ImageInput] = field(default_factory=list)
    temp_text: List[str] = field(default_factory=list)
    perm_text: List[str] = field(default_factory=list)
    tool_responses: List[ToolResponse] = field(default_factory=list)

    def get_concatenated_text(self, is_last_turn: bool) -> str:
        """
        Concatenates the text inputs.
        If it's the last turn, both temp_text and perm_text are visible.
        Otherwise, only perm_text is visible.
        """
        texts = []
        if is_last_turn:
            texts.extend(self.temp_text)
        texts.extend(self.perm_text)
        return "\n".join(texts)

@dataclass
class TurnOutput:
    """Output received from the model for a single turn."""
    thought: Optional[str] = None
    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)

@dataclass
class ChatTurn:
    """A single turn containing input and optional output."""
    turn_input: TurnInput
    turn_output: Optional[TurnOutput] = None

@dataclass
class ChatContext:
    """The full conversation context containing system prompt and historical turns."""
    system_prompt: str
    conversations: List[ChatTurn] = field(default_factory=list)
