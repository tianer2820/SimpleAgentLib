from typing import List, Optional, Any
from .agent import Agent, AgentCallbacks
from .skills_tools import Tool, Skill
from .model_backend import ModelBackend

class ExtendedAgent(Agent):
    """
    Subclass of Agent that supports:
    1. System rules as a list of strings.
    2. Skills dynamically loaded.
    3. Context auto-compression.
    """
    
    def __init__(
        self,
        sys_prompt: str,
        backend: ModelBackend,
        tools: Optional[List[Tool]] = None,
        callbacks: Optional[AgentCallbacks] = None,
        terminating_tool_calls: Optional[List[str]] = None,
        rules: Optional[List[str]] = None,
        skills: Optional[List[Skill]] = None,
        max_context_length: Optional[int] = None,
    ):
        self.max_context_length = max_context_length
        self.available_skills = {}

        # 1. Format rules if provided and append to system prompt
        if rules:
            rules_formatted = "\n".join(f"- {rule}" for rule in rules)
            sys_prompt = f"{sys_prompt}\n\nRules:\n{rules_formatted}"

        # 2. Format skills if provided and append to system prompt
        if skills:
            for skill in skills:
                self.available_skills[skill.name] = skill
            
            skills_formatted = "\n".join(f"- {s.name}: {s.description}" for s in skills)
            sys_prompt = f"{sys_prompt}\n\nAvailable Skills:\n{skills_formatted}"

        # 3. Call super constructor
        super().__init__(
            sys_prompt=sys_prompt,
            backend=backend,
            tools=tools,
            callbacks=callbacks,
            terminating_tool_calls=terminating_tool_calls,
        )

        # 4. Add the load_skill tool if skills are available
        if self.available_skills:
            load_skill_tool = Tool.from_function(self.load_skill)
            self.tools.append(load_skill_tool)

    def load_skill(self, skill_name: str) -> str:
        """Loads the full document of a skill by name, making it readable in context."""
        if skill_name not in self.available_skills:
            return f"Error: Skill '{skill_name}' not found."
        
        skill = self.available_skills[skill_name]
        path_str = f"Folder Path: {skill.path}\n" if skill.path else ""
        return f"Skill '{skill.name}'\nSkill Folder: {skill.path}\nSkill Document: {skill.document}"

    def _trigger_callback(self, name: str, *args, **kwargs) -> Any:
        res = super()._trigger_callback(name, *args, **kwargs)
        if name == "on_final_output":
            self._compress_context_if_needed()
        return res

    def _compress_context_if_needed(self) -> None:
        if self.max_context_length is not None and len(self.context.conversations) >= self.max_context_length:
            target_len = self.max_context_length // 2
            start_idx = len(self.context.conversations) - target_len
            
            # Make sure we are not cutting off half of the tool calling loop
            while start_idx > 0 and self.context.conversations[start_idx].turn_input.tool_responses:
                start_idx -= 1
                
            if start_idx > 0:
                self.context.conversations = self.context.conversations[start_idx:]
