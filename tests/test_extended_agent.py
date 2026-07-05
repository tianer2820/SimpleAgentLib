import unittest
import time
import tempfile
import os
from agentlib import (
    ExtendedAgent,
    MockBackend,
    Tool,
    Skill,
    discover_skills,
    TurnInput,
    TurnOutput,
    ToolCall,
    ToolResponse,
    ChatTurn
)

class TestExtendedAgent(unittest.TestCase):
    
    def test_rules_appending(self):
        backend = MockBackend(responses=[TurnOutput(text="Hello")])
        agent = ExtendedAgent(
            sys_prompt="You are a helpful assistant.",
            backend=backend,
            rules=["Rule 1: Be polite.", "Rule 2: Keep it short."]
        )
        self.assertIn("Rules:\n- Rule 1: Be polite.\n- Rule 2: Keep it short.", agent.context.system_prompt)
        agent.stop()

    def test_skills_parsing_and_prompt(self):
        backend = MockBackend(responses=[TurnOutput(text="Hello")])
        skill = Skill(
            name="test_skill",
            description="A test skill short description.",
            document="This is the full documentation for test_skill."
        )
        agent = ExtendedAgent(
            sys_prompt="You are a helpful assistant.",
            backend=backend,
            skills=[skill]
        )
        # Verify prompt update
        self.assertIn("Available Skills:\n- test_skill: A test skill short description.", agent.context.system_prompt)
        # Verify load_skill tool is registered
        tool_names = [t.name for t in agent.tools]
        self.assertIn("load_skill", tool_names)
        agent.stop()

    def test_dynamic_load_skill(self):
        backend = MockBackend(responses=[TurnOutput(text="Hello")])
        skill = Skill(
            name="test_skill",
            description="A test skill short description.",
            document="This is the full documentation for test_skill.",
            path="/path/to/test_skill"
        )
        agent = ExtendedAgent(
            sys_prompt="You are a helpful assistant.",
            backend=backend,
            skills=[skill]
        )
        
        # Load skill via method call
        result = agent.load_skill("test_skill")
        self.assertIn("This is the full documentation for test_skill.", result)
        self.assertIn("Skill Folder: /path/to/test_skill", result)
        
        # Loading non-existent skill
        result2 = agent.load_skill("unknown_skill")
        self.assertIn("Error", result2)
        agent.stop()

    def test_discover_skills(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create skills/my_skill/SKILL.md
            skills_dir = os.path.join(tmpdir, "skills")
            os.makedirs(os.path.join(skills_dir, "my_skill"))
            skill_md = """---
name: my_skill
description: This is my skill.
---
Full content of my skill document.
"""
            with open(os.path.join(skills_dir, "my_skill", "SKILL.md"), "w", encoding="utf-8") as f:
                f.write(skill_md)
                
            skills = discover_skills(tmpdir)
            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "my_skill")
            self.assertEqual(skills[0].description, "This is my skill.")
            self.assertEqual(skills[0].document, "Full content of my skill document.")
            self.assertEqual(skills[0].path, os.path.join(skills_dir, "my_skill"))

    def test_context_compression_simple(self):
        # We want to compress after 4 turns
        # Using a mock backend that returns simple text
        responses = [
            TurnOutput(text="Response 1"),
            TurnOutput(text="Response 2"),
            TurnOutput(text="Response 3"),
            TurnOutput(text="Response 4"),
            TurnOutput(text="Response 5"),
        ]
        backend = MockBackend(responses=responses)
        agent = ExtendedAgent(
            sys_prompt="System",
            backend=backend,
            max_context_length=4
        )
        
        # Send 4 inputs. Each should create 1 turn (User Input -> Model Response in-place).
        # We wait after each to ensure processing is complete.
        agent.send_input(TurnInput(perm_text=["Input 1"]))
        time.sleep(0.05)
        self.assertEqual(len(agent.context.conversations), 1)

        agent.send_input(TurnInput(perm_text=["Input 2"]))
        time.sleep(0.05)
        self.assertEqual(len(agent.context.conversations), 2)

        agent.send_input(TurnInput(perm_text=["Input 3"]))
        time.sleep(0.05)
        self.assertEqual(len(agent.context.conversations), 3)

        # Sending the 4th input will make the length 4 at the end of the turn,
        # which triggers compression to 4 // 2 = 2 turns.
        agent.send_input(TurnInput(perm_text=["Input 4"]))
        time.sleep(0.05)
        
        # Verify context is compressed to 2 turns
        self.assertEqual(len(agent.context.conversations), 2)
        self.assertEqual(agent.context.conversations[0].turn_input.perm_text, ["Input 3"])
        self.assertEqual(agent.context.conversations[1].turn_input.perm_text, ["Input 4"])
        
        agent.stop()

    def test_context_compression_with_tool_loop_protection(self):
        # Setup: max_context_length = 4, target = 2.
        # Conversations will have:
        # Turn 0: User Input 1 -> Text Response (Length 1)
        # Turn 1: User Input 2 -> Tool Call (Start of tool loop)
        # Turn 2: Tool Response -> Text Response (End of tool loop)
        # Turn 3: User Input 3 -> Text Response (Length 4 at end of this turn)
        # Compression target is 2 (start_idx = 4 - 2 = 2).
        # Turn 2 has tool responses. It must NOT be cut off.
        # It should decrement start_idx until 1 (User Input 2 has no tool responses).
        # So it should retain Turn 1, Turn 2, and Turn 3 (retaining 3 turns instead of 2).
        
        def mock_tool() -> str:
            return "ok"
            
        tool_obj = Tool.from_function(mock_tool)
        
        responses = [
            TurnOutput(text="Response 1"),
            # Interaction 2 tool loop:
            TurnOutput(tool_calls=[ToolCall(id="t1", name="mock_tool", args={})]),
            TurnOutput(text="Response 2 final"),
            # Interaction 3:
            TurnOutput(text="Response 3"),
        ]
        backend = MockBackend(responses=responses)
        agent = ExtendedAgent(
            sys_prompt="System",
            backend=backend,
            tools=[tool_obj],
            max_context_length=4
        )
        
        # Interaction 1 (Creates Turn 0)
        agent.send_input(TurnInput(perm_text=["Input 1"]))
        time.sleep(0.05)
        self.assertEqual(len(agent.context.conversations), 1)

        # Interaction 2 (Creates Turn 1 and Turn 2 due to tool call loop)
        agent.send_input(TurnInput(perm_text=["Input 2"]))
        time.sleep(0.1)
        # Total turns should be 3 (Turn 0: Input 1, Turn 1: Input 2 -> Tool Call, Turn 2: Tool Response -> Response 2 final)
        self.assertEqual(len(agent.context.conversations), 3)

        # Interaction 3 (Creates Turn 3. Length reaches 4 at the end of the turn)
        agent.send_input(TurnInput(perm_text=["Input 3"]))
        time.sleep(0.05)
        
        # Compression should trigger because len is 4.
        # It wants to keep 2 turns (start_idx = 2).
        # Turn 2 has tool_responses, so start_idx decrements to 1.
        # Turn 1 does not have tool_responses.
        # Slices to conversations[1:]. Length becomes 3.
        self.assertEqual(len(agent.context.conversations), 3)
        self.assertEqual(agent.context.conversations[0].turn_input.perm_text, ["Input 2"])
        self.assertEqual(agent.context.conversations[1].turn_input.tool_responses[0].content, "ok")
        self.assertEqual(agent.context.conversations[2].turn_input.perm_text, ["Input 3"])
        
        agent.stop()

if __name__ == "__main__":
    unittest.main()
