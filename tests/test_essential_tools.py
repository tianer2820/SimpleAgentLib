import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from agentlib.essential_tools import WorkspaceTools, WebTools


class TestWorkspaceTools(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to serve as our sandbox workspace root
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_path = Path(self.temp_dir.name).resolve()
        self.tools = WorkspaceTools(workspace_root=str(self.workspace_path), sandbox_enabled=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_path_resolution_and_sandboxing(self):
        # Path inside workspace
        inside_path = self.tools._resolve_path("sub/file.txt")
        self.assertEqual(inside_path, self.workspace_path / "sub" / "file.txt")

        # Absolute path inside workspace
        abs_inside = str(self.workspace_path / "another.txt")
        resolved_abs = self.tools._resolve_path(abs_inside)
        self.assertEqual(resolved_abs, Path(abs_inside))

        # Relative path escaping workspace
        with self.assertRaises(PermissionError):
            self.tools._resolve_path("../outside.txt")

        # Absolute path escaping workspace
        with self.assertRaises(PermissionError):
            self.tools._resolve_path("/etc/passwd")

        # Check when sandbox is disabled
        unsandboxed_tools = WorkspaceTools(workspace_root=str(self.workspace_path), sandbox_enabled=False)
        resolved_outside = unsandboxed_tools._resolve_path("../outside.txt")
        self.assertEqual(resolved_outside, (self.workspace_path / "../outside.txt").resolve())

    def test_write_and_read_file(self):
        test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        file_path = "subfolder/test_file.txt"

        # Write file (should auto-create subfolder)
        result = self.tools.write_file(file_path, test_content)
        self.assertIn("Successfully wrote", result)

        # Read entire file
        read_all = self.tools.read_file(file_path)
        self.assertEqual(read_all, test_content)

        # Read line range
        read_range = self.tools.read_file(file_path, start_line=2, end_line=4)
        self.assertEqual(read_range, "Line 2\nLine 3\nLine 4")

        # Read out of bounds
        read_empty = self.tools.read_file(file_path, start_line=10, end_line=15)
        self.assertEqual(read_empty, "")

        # Try to overwrite without permission
        with self.assertRaises(FileExistsError):
            self.tools.write_file(file_path, "new content", overwrite=False)

        # Try reading non-existent file
        with self.assertRaises(FileNotFoundError):
            self.tools.read_file("nonexistent.txt")

    def test_apply_patch(self):
        file_path = "patch_test.txt"
        original_content = (
            "def hello():\n"
            "    print('Hello World')\n"
            "\n"
            "def greet(name):\n"
            "    return f'Hi, {name}'\n"
        )
        self.tools.write_file(file_path, original_content)

        # 1. Apply single search/replace block patch
        patch_content = (
            "<<<<<<< SEARCH\n"
            "def greet(name):\n"
            "    return f'Hi, {name}'\n"
            "=======\n"
            "def greet(name):\n"
            "    print(f'Greeting {name}')\n"
            "    return f'Hello, {name}'\n"
            ">>>>>>> REPLACE"
        )

        res = self.tools.apply_patch(file_path, patch_content)
        self.assertIn("Successfully applied patch", res)

        updated_content = self.tools.read_file(file_path)
        expected_content = (
            "def hello():\n"
            "    print('Hello World')\n"
            "\n"
            "def greet(name):\n"
            "    print(f'Greeting {name}')\n"
            "    return f'Hello, {name}'"
        )
        self.assertEqual(updated_content, expected_content)

        # 2. Apply multi-block patch
        multi_patch = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('Hello World')\n"
            "=======\n"
            "def hello_v2():\n"
            "    print('Hello Everyone')\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "def greet(name):\n"
            "    print(f'Greeting {name}')\n"
            "=======\n"
            "def greet_v2(name):\n"
            "    print(f'Welcoming {name}')\n"
            ">>>>>>> REPLACE"
        )
        
        self.tools.apply_patch(file_path, multi_patch)
        final_content = self.tools.read_file(file_path)
        
        expected_final = (
            "def hello_v2():\n"
            "    print('Hello Everyone')\n"
            "\n"
            "def greet_v2(name):\n"
            "    print(f'Welcoming {name}')\n"
            "    return f'Hello, {name}'"
        )
        self.assertEqual(final_content, expected_final)

    def test_apply_patch_errors(self):
        file_path = "error_patch.txt"
        content = "line 1\nline 2\nline 3\nline 2\n"
        self.tools.write_file(file_path, content)

        # Search block not found
        missing_patch = (
            "<<<<<<< SEARCH\n"
            "line 99\n"
            "=======\n"
            "new line\n"
            ">>>>>>> REPLACE"
        )
        with self.assertRaises(ValueError) as ctx:
            self.tools.apply_patch(file_path, missing_patch)
        self.assertIn("not found in file", str(ctx.exception))

        # Ambiguous search block
        ambiguous_patch = (
            "<<<<<<< SEARCH\n"
            "line 2\n"
            "=======\n"
            "replaced line 2\n"
            ">>>>>>> REPLACE"
        )
        with self.assertRaises(ValueError) as ctx:
            self.tools.apply_patch(file_path, ambiguous_patch)
        self.assertIn("is ambiguous: found 2 occurrences", str(ctx.exception))

        # Empty search block
        empty_patch = (
            "<<<<<<< SEARCH\n"
            "=======\n"
            "new content\n"
            ">>>>>>> REPLACE"
        )
        with self.assertRaises(ValueError) as ctx:
            self.tools.apply_patch(file_path, empty_patch)
        self.assertIn("cannot be empty", str(ctx.exception))

        # Malformed format (missing REPLACE marker)
        malformed_patch = (
            "<<<<<<< SEARCH\n"
            "line 1\n"
            "=======\n"
            "new line\n"
        )
        with self.assertRaises(ValueError) as ctx:
            self.tools.apply_patch(file_path, malformed_patch)
        self.assertIn("missing '>>>>>>> REPLACE' marker line.", str(ctx.exception))

    def test_list_dir(self):
        # Create nested file structure
        self.tools.write_file("file1.txt", "1")
        self.tools.write_file("file2.txt", "12")
        self.tools.write_file("sub/file3.txt", "123")
        self.tools.write_file("sub/deep/file4.txt", "1234")

        # Flat listing of root directory
        flat_list = self.tools.list_dir(".")
        self.assertIn("file1.txt (1 bytes)", flat_list)
        self.assertIn("file2.txt (2 bytes)", flat_list)
        self.assertIn("sub/ [DIR]", flat_list)
        self.assertNotIn("sub/file3.txt", flat_list)

        # Recursive listing
        recursive_list = self.tools.list_dir(".", recursive=True, max_depth=3)
        self.assertIn("file1.txt (1 bytes)", recursive_list)
        self.assertIn("sub/file3.txt (3 bytes)", recursive_list)
        self.assertIn("sub/deep/file4.txt (4 bytes)", recursive_list)

        # Test listing file instead of directory raises NotADirectoryError
        with self.assertRaises(NotADirectoryError):
            self.tools.list_dir("file1.txt")

    def test_exec_cmd(self):
        # Run standard echo
        result = self.tools.exec_cmd("echo 'Hello standard exec'")
        self.assertIn("Exit Code: 0", result)
        self.assertIn("Hello standard exec", result)

        # Run command in a subfolder and verify working directory boundary
        self.tools.write_file("sub/mark.txt", "inside sub")
        result_dir = self.tools.exec_cmd("cat mark.txt", cwd="sub")
        self.assertIn("Exit Code: 0", result_dir)
        self.assertIn("inside sub", result_dir)

        # Command failure
        result_fail = self.tools.exec_cmd("non_existent_command_12345")
        self.assertNotEqual(self._get_exit_code(result_fail), 0)

        # Command timeout
        # Using a sleep command that exceeds 1 second
        timeout_res = self.tools.exec_cmd("sleep 5", timeout=1)
        self.assertIn("timed out after 1 seconds", timeout_res)
        self.assertIn("Exit Code: Timeout", timeout_res)

    def test_get_tools(self):
        tools = self.tools.get_tools()
        self.assertEqual(len(tools), 5)
        names = {t.name for t in tools}
        expected_names = {"read_file", "write_file", "apply_patch", "list_dir", "exec_cmd"}
        self.assertEqual(names, expected_names)

    def _get_exit_code(self, result_str: str) -> int:
        for line in result_str.splitlines():
            if line.startswith("Exit Code:"):
                try:
                    return int(line.split(":")[1].strip())
                except ValueError:
                    return -1
        return -1

    def test_exec_cmd_confirm_callback(self):
        # 1. Approved callback
        approved_calls = []
        def approved_cb(cmd: str) -> bool:
            approved_calls.append(cmd)
            return True

        tools_approved = WorkspaceTools(
            workspace_root=str(self.workspace_path),
            confirm_cmd_callback=approved_cb
        )
        res = tools_approved.exec_cmd("echo 'approved'")
        self.assertIn("Exit Code: 0", res)
        self.assertIn("approved", res)
        self.assertEqual(approved_calls, ["echo 'approved'"])

        # 2. Denied callback
        denied_calls = []
        def denied_cb(cmd: str) -> bool:
            denied_calls.append(cmd)
            return False

        tools_denied = WorkspaceTools(
            workspace_root=str(self.workspace_path),
            confirm_cmd_callback=denied_cb
        )
        res_denied = tools_denied.exec_cmd("echo 'denied'")
        self.assertIn("Error: Command execution denied by user.", res_denied)
        self.assertEqual(denied_calls, ["echo 'denied'"])


class TestWebTools(unittest.TestCase):

    @patch("agentlib.essential_tools.TavilyClient")
    def test_web_search(self, mock_tavily):
        mock_client = MagicMock()
        mock_tavily.return_value = mock_client
        mock_client.search.return_value = {"results": [{"title": "Test Title", "url": "http://test.com"}]}

        web_tools = WebTools(api_key="test_key")
        res = web_tools.web_search("test query")
        
        mock_client.search.assert_called_once_with(query="test query", search_depth="advanced")
        self.assertEqual(res, {"results": [{"title": "Test Title", "url": "http://test.com"}]})

    @patch("agentlib.essential_tools.TavilyClient")
    def test_web_extract(self, mock_tavily):
        mock_client = MagicMock()
        mock_tavily.return_value = mock_client
        mock_client.extract.return_value = {
            "results": [
                {"title": "Page Title", "url": "http://example.com/page", "raw_content": "Page Body Content"}
            ]
        }

        web_tools = WebTools(api_key="test_key")
        res = web_tools.web_extract("http://example.com/page", "relevant info")
        
        mock_client.extract.assert_called_once_with(urls=["http://example.com/page"], query="relevant info")
        self.assertEqual(res, {
            "results": [
                {"title": "Page Title", "url": "http://example.com/page", "content": "Page Body Content"}
            ]
        })

    def test_web_tools_no_key(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}):
            web_tools = WebTools()
            res = web_tools.web_search("test")
            self.assertIn("error", res)
            self.assertIn("Tavily API key is not configured", res["error"])

    def test_web_get_tools(self):
        web_tools = WebTools(api_key="test_key")
        tools = web_tools.get_tools()
        self.assertEqual(len(tools), 2)
        names = {t.name for t in tools}
        self.assertEqual(names, {"web_search", "web_extract"})


if __name__ == "__main__":
    unittest.main()
