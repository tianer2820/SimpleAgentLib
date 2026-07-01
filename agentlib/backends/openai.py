import io
import base64
import json
from typing import Optional, List, Any
import openai
from ..model_backend import ModelBackend
from ..context import ChatContext, TurnOutput, ToolCall, ToolResponse
from ..skills_tools import Tool

def encode_pil_to_base64(image) -> str:
    """Helper to convert a PIL Image into a base64 encoded string for OpenAI multi-modal messages."""
    buffered = io.BytesIO()
    # Ensure image is in RGB format for JPEG compatibility
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class OpenAIBackend(ModelBackend):
    """OpenAI backend integration using the openai Python SDK."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o", max_concurrency: int = -1, **kwargs):
        super().__init__(max_concurrency=max_concurrency)
        self.api_key = api_key
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)

    def _chat_impl(self, context: ChatContext, tools: Optional[List[Tool]] = None) -> TurnOutput:
        """Translates custom context structures into OpenAI API format and executes the call."""
        
        # 1. Map Tool objects to OpenAI tool definitions
        openai_tools = []
        if tools:
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    }
                })

        # 2. Build conversation history messages
        messages = [{"role": "system", "content": context.system_prompt}]
        for i, turn in enumerate(context.conversations):
            is_last = (i == len(context.conversations) - 1)
            
            # Map TurnInput (user query / multimodal assets)
            text_content = turn.turn_input.get_concatenated_text(is_last_turn=is_last)
            content_blocks = []
            
            if text_content:
                content_blocks.append({"type": "text", "text": text_content})
                
            for img_input in turn.turn_input.imgs:
                base64_str = encode_pil_to_base64(img_input.image)
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                })
                
            if content_blocks:
                messages.append({"role": "user", "content": content_blocks})

            # Map Tool response messages (role="tool" for each response)
            for resp in turn.turn_input.tool_responses:
                messages.append({
                    "role": "tool",
                    "tool_call_id": resp.tool_call_id,
                    "name": resp.name,
                    "content": resp.content
                })

            # Map TurnOutput (assistant reasoning / tool calls / response text)
            if turn.turn_output:
                assistant_message = {"role": "assistant"}
                
                # Capture reasoning content (thoughts) if available
                # (Can be passed as standard text or reasoning fields on compatible models)
                if turn.turn_output.text:
                    assistant_message["content"] = turn.turn_output.text
                
                if turn.turn_output.tool_calls:
                    openai_calls = []
                    for call in turn.turn_output.tool_calls:
                        openai_calls.append({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.args)
                            }
                        })
                    assistant_message["tool_calls"] = openai_calls
                
                messages.append(assistant_message)

        # 3. Call ChatCompletion endpoint
        kwargs = {"model": self.model_name, "messages": messages}
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = self.client.chat.completions.create(**kwargs)

        # 4. Parse response into a TurnOutput
        choice = response.choices[0].message
        text = choice.content
        
        # Check if reasoning_content is returned (for o1/o3-type models)
        thought = getattr(choice, "reasoning_content", None)

        tool_calls = []
        if choice.tool_calls:
            for call in choice.tool_calls:
                tool_calls.append(ToolCall(
                    id=call.id,
                    name=call.function.name,
                    args=json.loads(call.function.arguments)
                ))

        return TurnOutput(thought=thought, text=text, tool_calls=tool_calls)
