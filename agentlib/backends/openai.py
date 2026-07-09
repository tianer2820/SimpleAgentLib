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
        self.client = openai.OpenAI(api_key=api_key, **kwargs)

    def _chat_impl(
        self,
        context: ChatContext,
        tools: Optional[List[Tool]] = None,
        on_stream_chunk: Optional[Callable[[Optional[str], Optional[str]], None]] = None,
    ) -> TurnOutput:
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
        from typing import Callable
        if on_stream_chunk:
            kwargs = {"model": self.model_name, "messages": messages, "stream": True}
            if openai_tools:
                kwargs["tools"] = openai_tools

            response = self.client.chat.completions.create(**kwargs)
            
            text_parts = []
            reasoning_parts = []
            tool_calls_accum = {}
            
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                
                # Text content
                chunk_text = getattr(delta, "content", None)
                if chunk_text:
                    text_parts.append(chunk_text)
                
                # Reasoning / thoughts
                chunk_thought = getattr(delta, "reasoning_content", None)
                if chunk_thought:
                    reasoning_parts.append(chunk_thought)
                
                # Trigger callback
                if chunk_text or chunk_thought:
                    on_stream_chunk(chunk_text, chunk_thought)
                
                # Tool calls delta
                chunk_tool_calls = getattr(delta, "tool_calls", None)
                if chunk_tool_calls:
                    for tc_delta in chunk_tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_accum:
                            tool_calls_accum[idx] = {
                                "id": "",
                                "name": "",
                                "arguments_parts": []
                            }
                        
                        if getattr(tc_delta, "id", None):
                            tool_calls_accum[idx]["id"] = tc_delta.id
                        if getattr(tc_delta, "function", None):
                            fn_delta = tc_delta.function
                            if getattr(fn_delta, "name", None):
                                tool_calls_accum[idx]["name"] = fn_delta.name
                            if getattr(fn_delta, "arguments", None):
                                tool_calls_accum[idx]["arguments_parts"].append(fn_delta.arguments)
            
            # Reconstruct tool calls list
            import time
            tool_calls = []
            for idx in sorted(tool_calls_accum.keys()):
                accum = tool_calls_accum[idx]
                args_str = "".join(accum["arguments_parts"])
                try:
                    args = json.loads(args_str) if args_str else {}
                except Exception:
                    args = {}
                tool_calls.append(ToolCall(
                    id=accum["id"] or f"call_{int(time.time())}_{idx}",
                    name=accum["name"],
                    args=args
                ))
                
            full_text = "".join(text_parts) if text_parts else None
            full_thought = "".join(reasoning_parts) if reasoning_parts else None
            return TurnOutput(thought=full_thought, text=full_text, tool_calls=tool_calls)

        else:
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
