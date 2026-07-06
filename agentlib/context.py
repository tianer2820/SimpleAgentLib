from dataclasses import dataclass, field
from typing import List, Optional, Any
from PIL import Image

@dataclass
class ImageInput:
    """Wrapper for PIL Image object."""
    image: Image.Image

    def to_dict(self) -> dict:
        import io
        import base64
        buffered = io.BytesIO()
        self.image.save(buffered, format="JPEG")
        return {
            "image_b64": base64.b64encode(buffered.getvalue()).decode('utf-8')
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageInput":
        import io
        import base64
        from PIL import Image
        img_data = base64.b64decode(d.get("image_b64", ""))
        image = Image.open(io.BytesIO(img_data))
        return cls(image=image)

# Note: AudioInput is left unimplemented as per instructions.

@dataclass
class ToolCall:
    """Represents a request from the model to execute a tool."""
    id: str
    name: str
    args: dict
    thought_signature: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "args": self.args
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        return cls(
            id=d.get("id", ""),
            name=d.get("name", ""),
            args=d.get("args", {})
        )

@dataclass
class ToolResponse:
    """Represents the output from executing a tool."""
    tool_call_id: str
    name: str
    content: str

    def to_dict(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolResponse":
        return cls(
            tool_call_id=d.get("tool_call_id", ""),
            name=d.get("name", ""),
            content=d.get("content", "")
        )

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

    def to_dict(self) -> dict:
        return {
            "imgs": [img.to_dict() for img in self.imgs],
            "temp_text": self.temp_text,
            "perm_text": self.perm_text,
            "tool_responses": [tr.to_dict() for tr in self.tool_responses]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TurnInput":
        return cls(
            imgs=[ImageInput.from_dict(img_d) for img_d in d.get("imgs", [])],
            temp_text=d.get("temp_text", []),
            perm_text=d.get("perm_text", []),
            tool_responses=[ToolResponse.from_dict(tr_d) for tr_d in d.get("tool_responses", [])]
        )

@dataclass
class TurnOutput:
    """Output received from the model for a single turn."""
    thought: Optional[str] = None
    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "thought": self.thought,
            "text": self.text,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TurnOutput":
        return cls(
            thought=d.get("thought"),
            text=d.get("text"),
            tool_calls=[ToolCall.from_dict(tc_d) for tc_d in d.get("tool_calls", [])]
        )

@dataclass
class ChatTurn:
    """A single turn containing input and optional output."""
    turn_input: TurnInput
    turn_output: Optional[TurnOutput] = None

    def to_dict(self) -> dict:
        return {
            "turn_input": self.turn_input.to_dict(),
            "turn_output": self.turn_output.to_dict() if self.turn_output else None
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChatTurn":
        turn_input = TurnInput.from_dict(d.get("turn_input", {}))
        to_dict = d.get("turn_output")
        turn_output = TurnOutput.from_dict(to_dict) if to_dict else None
        return cls(
            turn_input=turn_input,
            turn_output=turn_output
        )

@dataclass
class ChatContext:
    """The full conversation context containing system prompt and historical turns."""
    system_prompt: str
    conversations: List[ChatTurn] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "conversations": [turn.to_dict() for turn in self.conversations]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChatContext":
        return cls(
            system_prompt=d.get("system_prompt", ""),
            conversations=[ChatTurn.from_dict(t_d) for t_d in d.get("conversations", [])]
        )
