import unittest
import inspect
from typing import List, Dict, Any, Optional
from agentlib.skills_tools import Tool

class TestToolFromFunction(unittest.TestCase):
    
    def test_basic_function(self):
        def my_func(a: int, b: str, c: float = 1.0) -> str:
            """
            This is my test function.
            It does things.
            """
            return f"{a} {b} {c}"
            
        tool = Tool.from_function(my_func)
        
        # Verify basic fields
        self.assertEqual(tool.name, "my_func")
        self.assertEqual(tool.description, "This is my test function.\nIt does things.")
        
        # Verify schema parameters
        params = tool.parameters
        self.assertEqual(params["type"], "object")
        
        # Verify properties type mapping
        properties = params["properties"]
        self.assertEqual(properties["a"]["type"], "integer")
        self.assertEqual(properties["b"]["type"], "string")
        self.assertEqual(properties["c"]["type"], "number")
        
        # Verify required parameters (a and b are required, c has a default so it is not)
        self.assertIn("a", params["required"])
        self.assertIn("b", params["required"])
        self.assertNotIn("c", params["required"])
        self.assertEqual(len(params["required"]), 2)
        
        # Verify callability
        self.assertEqual(tool(1, "hello", c=2.5), "1 hello 2.5")

    def test_complex_types(self):
        def complex_func(flag: bool, items: list, mapping: dict) -> None:
            pass
            
        tool = Tool.from_function(complex_func)
        properties = tool.parameters["properties"]
        self.assertEqual(properties["flag"]["type"], "boolean")
        self.assertEqual(properties["items"]["type"], "array")
        self.assertEqual(properties["mapping"]["type"], "object")

    def test_no_docstring(self):
        def no_doc(x: int):
            pass
            
        tool = Tool.from_function(no_doc)
        self.assertEqual(tool.description, "")

    def test_special_parameters(self):
        class MyClass:
            def method(self, x: int, *args, **kwargs):
                pass
                
            @classmethod
            def class_method(cls, y: str):
                pass
                
        # For instance method
        tool = Tool.from_function(MyClass.method)
        self.assertNotIn("self", tool.parameters["properties"])
        self.assertNotIn("args", tool.parameters["properties"])
        self.assertNotIn("kwargs", tool.parameters["properties"])
        self.assertIn("x", tool.parameters["properties"])
        
        # For class method
        tool_cls = Tool.from_function(MyClass.class_method)
        self.assertNotIn("cls", tool_cls.parameters["properties"])
        self.assertIn("y", tool_cls.parameters["properties"])

    def test_no_required_parameters(self):
        def all_optional(x: int = 1, y: str = "default"):
            pass
            
        tool = Tool.from_function(all_optional)
        self.assertNotIn("required", tool.parameters)

    def test_no_parameters(self):
        def no_params():
            pass
            
        tool = Tool.from_function(no_params)
        self.assertEqual(tool.parameters["properties"], {})
        self.assertNotIn("required", tool.parameters)

    def test_fallback_type_mapping(self):
        # Unknown/complex type annotation defaults to "string"
        def unknown_type_func(custom: object, opt: Optional[int]):
            pass
            
        tool = Tool.from_function(unknown_type_func)
        properties = tool.parameters["properties"]
        self.assertEqual(properties["custom"]["type"], "string")
        self.assertEqual(properties["opt"]["type"], "string")

    def test_schema_openai_google_compatibility(self):
        def calculate(expression: str, precision: int = 4) -> float:
            """Evaluates a math expression with a given precision."""
            return round(eval(expression), precision)
            
        tool = Tool.from_function(calculate)
        
        # 1. OpenAI format compatibility
        # OpenAI expects a 'type': 'function' tool, with a 'function' object containing 'name', 'description', and 'parameters'.
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        }
        self.assertEqual(openai_tool["type"], "function")
        self.assertEqual(openai_tool["function"]["name"], "calculate")
        self.assertEqual(openai_tool["function"]["description"], "Evaluates a math expression with a given precision.")
        
        params = openai_tool["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertEqual(params["properties"]["expression"]["type"], "string")
        self.assertEqual(params["properties"]["precision"]["type"], "integer")
        self.assertEqual(params["required"], ["expression"])
        
        # 2. Google Gemini format compatibility
        # google.genai.types.FunctionDeclaration expects name, description, and parameters (representing function params schema)
        google_decl = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        }
        self.assertEqual(google_decl["name"], "calculate")
        self.assertEqual(google_decl["description"], "Evaluates a math expression with a given precision.")
        self.assertEqual(google_decl["parameters"]["type"], "object")
        self.assertEqual(google_decl["parameters"]["properties"]["expression"]["type"], "string")
        self.assertEqual(google_decl["parameters"]["properties"]["precision"]["type"], "integer")
        self.assertEqual(google_decl["parameters"]["required"], ["expression"])

if __name__ == "__main__":
    unittest.main()
