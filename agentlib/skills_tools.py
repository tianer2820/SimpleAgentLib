import os
import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, get_type_hints, List, Optional

class Tool:
    """Wraps a python function as an agent tool with JSON schema descriptions."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema dictionary
        self.func = func

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)

    @staticmethod
    def from_function(func: Callable) -> "Tool":
        """
        Creates a Tool object automatically from a Python function's
        name, docstring, and signature type hints.
        """
        name = func.__name__
        description = func.__doc__.strip() if func.__doc__ else ""
        
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        # Get type hints, falling back to signature annotations
        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}
        
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            dict: "object",
            list: "array"
        }
        
        for param_name, param in sig.parameters.items():
            # Skip self, cls, *args, **kwargs
            if param_name in ("self", "cls"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
                
            param_type = type_hints.get(param_name, param.annotation)
            json_type = type_mapping.get(param_type, "string")
            
            properties[param_name] = {
                "type": json_type
            }
            
            # If there's no default value, it's required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
                
        parameters = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters["required"] = required
            
        return Tool(name=name, description=description, parameters=parameters, func=func)


@dataclass
class Skill:
    """Wraps a skill's name, short description, and full documentation."""
    name: str
    description: str
    document: str
    path: Optional[str] = None


def parse_skill_markdown(content: str) -> Optional[dict]:
    """Parses YAML frontmatter and body content from SKILL.md."""
    lines = content.strip().splitlines()
    if not lines or lines[0].strip() != '---':
        return None
        
    frontmatter_lines = []
    body_start_idx = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == '---':
            body_start_idx = idx + 1
            break
        frontmatter_lines.append(line)
        
    if body_start_idx == -1:
        return None
        
    metadata = {}
    for line in frontmatter_lines:
        if ':' in line:
            key, val = line.split(':', 1)
            metadata[key.strip()] = val.strip().strip('"').strip("'")
            
    body = "\n".join(lines[body_start_idx:])
    
    return {
        "name": metadata.get("name"),
        "description": metadata.get("description"),
        "document": body.strip()
    }


def discover_skills(folder_path: str) -> List[Skill]:
    """
    Discovers and loads skills from the specified folder path.
    Looks inside `folder_path/skills` if it exists, otherwise looks directly in `folder_path`.
    Each skill is expected to be a subdirectory containing a `SKILL.md` file.
    """
    skills = []
    if not os.path.exists(folder_path):
        return skills
        
    target_dir = os.path.join(folder_path, "skills")
    if not os.path.isdir(target_dir):
        target_dir = folder_path
        
    for entry in os.listdir(target_dir):
        sub_path = os.path.join(target_dir, entry)
        if os.path.isdir(sub_path):
            skill_md_path = os.path.join(sub_path, "SKILL.md")
            if os.path.exists(skill_md_path):
                try:
                    with open(skill_md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    parsed = parse_skill_markdown(content)
                    if parsed and parsed.get("name") and parsed.get("description"):
                        skills.append(Skill(
                            name=parsed["name"],
                            description=parsed["description"],
                            document=parsed["document"],
                            path=sub_path
                        ))
                except Exception:
                    pass
    return skills

