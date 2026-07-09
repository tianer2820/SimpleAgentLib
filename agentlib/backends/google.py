import io
import time
from typing import Optional, List, Any
from google import genai
from google.genai import types
from ..model_backend import ModelBackend
from ..context import ChatContext, TurnOutput, ToolCall, ToolResponse
from ..skills_tools import Tool

class GoogleBackend(ModelBackend):
    """Google Gemini backend integration using the new google-genai SDK."""
    
    def __init__(self, api_key: str, model_name: str, max_concurrency: int = -1, **kwargs):
        super().__init__(max_concurrency=max_concurrency)
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key, **kwargs)

    def _chat_impl(
        self,
        context: ChatContext,
        tools: Optional[List[Tool]] = None,
        on_stream_chunk: Optional[Callable[[Optional[str], Optional[str]], None]] = None,
    ) -> TurnOutput:
        """Translates custom context structures into Gemini API formats and executes the call."""
        
        # 1. Map Tool objects to google.genai.types.Tool formats
        gemini_tools = []
        if tools:
            declarations = []
            for tool in tools:
                declarations.append(types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters
                ))
            gemini_tools.append(types.Tool(function_declarations=declarations))

        # 2. Build conversation history contents
        gemini_contents = []
        for i, turn in enumerate(context.conversations):
            is_last = (i == len(context.conversations) - 1)
            
            # Translate TurnInput
            input_parts = []
            role = "user"
            
            text_content = turn.turn_input.get_concatenated_text(is_last_turn=is_last)
            if text_content:
                input_parts.append(types.Part.from_text(text=text_content))
                
            for img_input in turn.turn_input.imgs:
                buffered = io.BytesIO()
                img = img_input.image
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(buffered, format="JPEG")
                img_bytes = buffered.getvalue()
                input_parts.append(types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/jpeg"
                ))
                
            if turn.turn_input.tool_responses:
                role = "tool"
                for resp in turn.turn_input.tool_responses:
                    input_parts.append(types.Part.from_function_response(
                        name=resp.name,
                        response={"result": resp.content}
                    ))
                
            if input_parts:
                gemini_contents.append(types.Content(role=role, parts=input_parts))

            # Translate TurnOutput (Model role)
            if turn.turn_output:
                output_parts = []
                if turn.turn_output.text:
                    output_parts.append(types.Part.from_text(text=turn.turn_output.text))
                for call in turn.turn_output.tool_calls:
                    fc_part = types.Part(
                        function_call=types.FunctionCall(
                            name=call.name,
                            args=call.args,
                            id=call.id
                        ),
                        thought_signature=call.thought_signature
                    )
                    output_parts.append(fc_part)
                if output_parts:
                    gemini_contents.append(types.Content(role="model", parts=output_parts))

        # 3. Call Gemini generate content API
        config = types.GenerateContentConfig(
            system_instruction=context.system_prompt,
            tools=gemini_tools if gemini_tools else None
        )

        from typing import Callable
        if on_stream_chunk:
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=gemini_contents,
                config=config
            )
            text_parts = []
            reasoning_parts = []
            tool_calls = []
            
            for chunk in response:
                chunk_text = None
                chunk_thought = None
                
                # Check for reasoning_content (thoughts)
                if chunk.candidates:
                    cand = chunk.candidates[0]
                    if hasattr(cand, "reasoning_content") and cand.reasoning_content:
                        chunk_thought = cand.reasoning_content
                        reasoning_parts.append(chunk_thought)
                
                # Check for text in parts
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    p_texts = [p.text for p in chunk.candidates[0].content.parts if p.text]
                    if p_texts:
                        chunk_text = "".join(p_texts)
                        text_parts.append(chunk_text)
                
                # Trigger callback
                if chunk_text or chunk_thought:
                    on_stream_chunk(chunk_text, chunk_thought)
                
                # Check for function calls in parts
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if part.function_call:
                            call = part.function_call
                            call_id = getattr(call, "id", None) or f"call_{int(time.time())}"
                            thought_sig = getattr(part, "thought_signature", None)
                            tool_calls.append(ToolCall(
                                id=call_id,
                                name=call.name,
                                args=call.args,
                                thought_signature=thought_sig
                            ))
                elif hasattr(chunk, "function_calls") and chunk.function_calls:
                    for call in chunk.function_calls:
                        call_id = getattr(call, "id", None) or f"call_{int(time.time())}"
                        tool_calls.append(ToolCall(
                            id=call_id,
                            name=call.name,
                            args=call.args,
                            thought_signature=None
                        ))
            
            full_text = "".join(text_parts) if text_parts else None
            full_thought = "".join(reasoning_parts) if reasoning_parts else None
            return TurnOutput(thought=full_thought, text=full_text, tool_calls=tool_calls)

        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=gemini_contents,
                config=config
            )

            # 4. Parse response parts into TurnOutput
            text = None
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                text_parts = [part.text for part in response.candidates[0].content.parts if part.text]
                if text_parts:
                    text = "".join(text_parts)

            thought = None
            # Check if candidate has reasoning content
            if response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, "reasoning_content"):
                    thought = cand.reasoning_content

            tool_calls = []
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        call = part.function_call
                        call_id = getattr(call, "id", None) or f"call_{int(time.time())}"
                        thought_sig = getattr(part, "thought_signature", None)
                        tool_calls.append(ToolCall(
                            id=call_id,
                            name=call.name,
                            args=call.args,
                            thought_signature=thought_sig
                        ))
            elif response.function_calls:
                for call in response.function_calls:
                    call_id = getattr(call, "id", None) or f"call_{int(time.time())}"
                    tool_calls.append(ToolCall(
                        id=call_id,
                        name=call.name,
                        args=call.args,
                        thought_signature=None
                    ))

            return TurnOutput(thought=thought, text=text, tool_calls=tool_calls)
