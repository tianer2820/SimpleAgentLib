import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Union, Literal
from tavily import TavilyClient


from .skills_tools import Tool

logger = logging.getLogger(__name__)

class WorkspaceTools:
    """
    Provides a set of secure, workspace-confined file system and command execution tools.
    """
    
    def __init__(
        self,
        workspace_root: Optional[str] = None,
        sandbox_enabled: bool = True,
        confirm_cmd_callback: Optional[Callable[[str], Union[bool, str]]] = None,
        workspace_type: Literal["local", "podman"] = "local",
        podman_container: Optional[str] = None
    ):
        """
        Initializes the WorkspaceTools wrapper.
        
        Args:
            workspace_root: Absolute path to the allowed workspace root. Defaults to current directory.
            sandbox_enabled: If True, blocks accessing files/folders outside the workspace root.
            confirm_cmd_callback: Optional callback accepting the command string and returning a bool or feedback string.
            workspace_type: 'local' or 'podman'.
            podman_container: Container ID or Name if workspace_type is 'podman'.
        """
        self.workspace_type = workspace_type
        self.podman_container = podman_container
        
        if self.workspace_type == "podman":
            # find the default cwd in the container
            default_cwd = "/workspace"
            if self.podman_container:
                try:
                    res = subprocess.run(
                        ["podman", "exec", self.podman_container, "pwd"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        default_cwd = res.stdout.strip()
                except Exception:
                    pass
            import posixpath
            ws_root_str = (workspace_root or ".")
            if not posixpath.isabs(ws_root_str):
                resolved_ws_root = posixpath.normpath(posixpath.join(default_cwd, ws_root_str))
            else:
                resolved_ws_root = posixpath.normpath(ws_root_str)
            self.workspace_root = Path(resolved_ws_root)
        else:
            self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
            
        self.sandbox_enabled = sandbox_enabled
        self.confirm_cmd_callback = confirm_cmd_callback

    def _resolve_path(self, relative_or_absolute_path: str) -> Path:
        """
        Resolves a path to an absolute Path object and checks sandbox constraints.
        """
        if self.workspace_type == "podman":
            import posixpath
            path_str = relative_or_absolute_path
            if not posixpath.isabs(path_str):
                resolved_str = posixpath.normpath(posixpath.join(str(self.workspace_root), path_str))
            else:
                resolved_str = posixpath.normpath(path_str)

            if self.sandbox_enabled:
                rel = posixpath.relpath(resolved_str, str(self.workspace_root))
                if rel.startswith("..") or posixpath.isabs(rel):
                    raise PermissionError(
                        f"Access denied: Path '{relative_or_absolute_path}' resolved to '{resolved_str}', "
                        f"which is outside the workspace root '{self.workspace_root}'"
                    )
            return Path(resolved_str)

        path = Path(relative_or_absolute_path)
        if not path.is_absolute():
            resolved = (self.workspace_root / path).resolve()
        else:
            resolved = path.resolve()

        if self.sandbox_enabled:
            try:
                # Ensure the resolved path starts with the workspace root path
                resolved.relative_to(self.workspace_root)
            except ValueError:
                raise PermissionError(
                    f"Access denied: Path '{relative_or_absolute_path}' resolved to '{resolved}', "
                    f"which is outside the workspace root '{self.workspace_root}'"
                )
        return resolved

    def _container_path_exists(self, path: str) -> bool:
        if not self.podman_container:
            return False
        res = subprocess.run(
            ["podman", "exec", self.podman_container, "test", "-e", path],
            capture_output=True
        )
        return res.returncode == 0

    def _container_path_is_dir(self, path: str) -> bool:
        if not self.podman_container:
            return False
        res = subprocess.run(
            ["podman", "exec", self.podman_container, "test", "-d", path],
            capture_output=True
        )
        return res.returncode == 0

    def _container_path_is_file(self, path: str) -> bool:
        if not self.podman_container:
            return False
        res = subprocess.run(
            ["podman", "exec", self.podman_container, "test", "-f", path],
            capture_output=True
        )
        return res.returncode == 0

    def read_file(self, path: str, start_line: int = 1, end_line: int = -1) -> str:
        """
        Reads the contents of a text file within the workspace. Supports optional line ranges.
        
        Args:
            path: Relative or absolute path of the file to read.
            start_line: Line number to start reading from (1-indexed, inclusive). Defaults to 1.
            end_line: Line number to stop reading at (1-indexed, inclusive). Use -1 to read to the end of the file.
        """
        resolved_path = self._resolve_path(path)
        if self.workspace_type == "podman":
            if not self._container_path_is_file(str(resolved_path)):
                raise FileNotFoundError(f"File not found in container: '{path}'")
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_name = tmp.name
            try:
                cp_cmd = ["podman", "cp", f"{self.podman_container}:{resolved_path}", tmp_name]
                res = subprocess.run(cp_cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise IOError(f"Failed to copy file from container: {res.stderr}")
                with open(tmp_name, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        else:
            if not resolved_path.is_file():
                raise FileNotFoundError(f"File not found: '{path}'")

            with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        lines = content.splitlines()
        total_lines = len(lines)
        
        # Normalize bounds
        start = max(1, start_line)
        end = total_lines if end_line == -1 else min(end_line, total_lines)

        if start > total_lines or start > end:
            return ""

        return "\n".join(lines[start - 1:end])

    def write_file(self, path: str, content: str, overwrite: bool = True, append: bool = False) -> str:
        """
        Writes, appends, or creates a text file with the given content. Creates parent directories automatically.
        
        Args:
            path: Relative or absolute path of the file to write.
            content: Text content to write into the file.
            overwrite: If True, overwrites the existing file content (ignored if append is True). If False, raises an error if the file exists.
            append: If True, appends content to the end of the file. Defaults to False.
        """
        resolved_path = self._resolve_path(path)
        if self.workspace_type == "podman":
            exists = self._container_path_exists(str(resolved_path))
            if exists:
                if self._container_path_is_dir(str(resolved_path)):
                    raise IsADirectoryError(f"Cannot overwrite a directory '{path}' with a file.")
                if not overwrite and not append:
                    raise FileExistsError(f"File already exists in container: '{path}'")

            # Automatically create any parent directories in container
            import posixpath
            parent_dir = posixpath.dirname(str(resolved_path))
            if parent_dir:
                assert self.podman_container
                subprocess.run(
                    ["podman", "exec", self.podman_container, "mkdir", "-p", parent_dir],
                    capture_output=True
                )

            import tempfile
            if append and exists:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp_name = tmp.name
                try:
                    cp_cmd = ["podman", "cp", f"{self.podman_container}:{resolved_path}", tmp_name]
                    res = subprocess.run(cp_cmd, capture_output=True, text=True)
                    if res.returncode != 0:
                        raise IOError(f"Failed to copy file from container for appending: {res.stderr}")
                    with open(tmp_name, "a", encoding="utf-8") as f:
                        f.write(content)
                    
                    cp_back_cmd = ["podman", "cp", tmp_name, f"{self.podman_container}:{resolved_path}"]
                    res = subprocess.run(cp_back_cmd, capture_output=True, text=True)
                    if res.returncode != 0:
                        raise IOError(f"Failed to copy appended file back to container: {res.stderr}")
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
            else:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as tmp:
                    tmp.write(content)
                    tmp_name = tmp.name
                try:
                    cp_cmd = ["podman", "cp", tmp_name, f"{self.podman_container}:{resolved_path}"]
                    res = subprocess.run(cp_cmd, capture_output=True, text=True)
                    if res.returncode != 0:
                        raise IOError(f"Failed to copy file to container: {res.stderr}")
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
        else:
            if resolved_path.exists():
                if resolved_path.is_dir():
                    raise IsADirectoryError(f"Cannot overwrite a directory '{path}' with a file.")
                if not overwrite and not append:
                    raise FileExistsError(f"File already exists: '{path}'")

            # Automatically create any parent directories if missing
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(resolved_path, mode, encoding="utf-8") as f:
                f.write(content)

        action = "appended to" if append else "wrote to"
        return f"Successfully {action} '{path}'"

    def apply_patch(self, path: str, patch: str) -> str:
        """
        Applies a SEARCH/REPLACE block patch to a file. Multiple blocks can be specified in a single patch.
        
        The format for each block must be:
        <<<<<<< SEARCH
        exact lines to search for
        =======
        lines to replace with
        >>>>>>> REPLACE
        
        Args:
            path: Relative or absolute path of the file to modify.
            patch: The SEARCH/REPLACE patch block string.
        """
        resolved_path = self._resolve_path(path)
        if self.workspace_type == "podman":
            if not self._container_path_is_file(str(resolved_path)):
                raise FileNotFoundError(f"File not found in container to apply patch: '{path}'")
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_name = tmp.name
            try:
                cp_cmd = ["podman", "cp", f"{self.podman_container}:{resolved_path}", tmp_name]
                res = subprocess.run(cp_cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise IOError(f"Failed to copy file from container to apply patch: {res.stderr}")
                with open(tmp_name, "r", encoding="utf-8", errors="replace") as f:
                    file_content = f.read()
            except Exception as e:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
                raise e
        else:
            if not resolved_path.is_file():
                raise FileNotFoundError(f"File not found to apply patch: '{path}'")

            with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
                file_content = f.read()

        # Normalize line endings for internal consistency during matching
        original_crlf = "\r\n" in file_content
        normalized_content = file_content.replace("\r\n", "\n")
        normalized_patch = patch.replace("\r\n", "\n")

        # Parse the patch content into individual SEARCH/REPLACE blocks
        blocks = []
        lines = normalized_patch.splitlines(keepends=True)
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            if line.startswith("<<<<<<< SEARCH"):
                search_lines = []
                i += 1
                while i < n and not lines[i].startswith("======="):
                    search_lines.append(lines[i])
                    i += 1
                
                if i >= n:
                    raise ValueError("Malformed patch: missing '=======' separator line.")
                i += 1  # Skip the '=======' separator line
                
                replace_lines = []
                while i < n and not lines[i].startswith(">>>>>>> REPLACE"):
                    replace_lines.append(lines[i])
                    i += 1
                
                if i >= n:
                    raise ValueError("Malformed patch: missing '>>>>>>> REPLACE' marker line.")
                
                search_str = "".join(search_lines)
                replace_str = "".join(replace_lines)
                
                blocks.append((search_str, replace_str))
            i += 1

        if not blocks:
            raise ValueError("No SEARCH/REPLACE blocks found in the patch content.")

        # Apply blocks sequentially
        current_content = normalized_content
        for idx, (search_block, replace_block) in enumerate(blocks):
            if not search_block.strip():
                raise ValueError(f"Search block #{idx + 1} cannot be empty or contain only whitespace.")

            count = current_content.count(search_block)
            if count == 0:
                raise ValueError(
                    f"Search block #{idx + 1} not found in file '{path}'. "
                    f"Ensure exact match including spacing and indentation. Searched block:\n{search_block!r}"
                )
            elif count > 1:
                raise ValueError(
                    f"Search block #{idx + 1} is ambiguous: found {count} occurrences in file '{path}'."
                )

            current_content = current_content.replace(search_block, replace_block, 1)

        # Restore original line endings if they were CRLF
        if original_crlf:
            current_content = current_content.replace("\n", "\r\n")

        if self.workspace_type == "podman":
            try:
                with open(tmp_name, "w", encoding="utf-8") as f:
                    f.write(current_content)
                cp_back_cmd = ["podman", "cp", tmp_name, f"{self.podman_container}:{resolved_path}"]
                res = subprocess.run(cp_back_cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise IOError(f"Failed to copy patched file back to container: {res.stderr}")
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        else:
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(current_content)

        return f"Successfully applied patch with {len(blocks)} block(s) to '{path}'"

    def list_dir(self, path: str = ".", recursive: bool = False, max_depth: int = 3) -> str:
        """
        Lists files and directories under the specified path, formatting names and file sizes.
        
        Args:
            path: Relative or absolute path of the directory to list. Defaults to '.'.
            recursive: If True, lists files recursively.
            max_depth: Maximum depth to recurse if recursive is True. Defaults to 3.
        """
        resolved_path = self._resolve_path(path)
        
        if self.workspace_type == "podman":
            if not self._container_path_is_dir(str(resolved_path)):
                raise NotADirectoryError(f"Not a directory inside container: '{path}'")
                
            import json
            python_script = """
import os, sys, json
def walk(current_dir, current_depth, max_depth, workspace_root, recursive):
    if current_depth > max_depth:
        return []
    results = []
    try:
        entries = sorted(os.scandir(current_dir), key=lambda e: (not e.is_dir(), e.name.lower()))
    except Exception as e:
        try:
            rel = os.path.relpath(current_dir, workspace_root)
        except Exception:
            rel = current_dir
        return [{"path": rel, "type": "error", "message": str(e)}]
    for entry in entries:
        try:
            rel_path = os.path.relpath(entry.path, workspace_root)
        except Exception:
            rel_path = entry.path
        if entry.is_dir():
            results.append({"path": rel_path, "type": "dir"})
            if recursive:
                results.extend(walk(entry.path, current_depth + 1, max_depth, workspace_root, recursive))
        else:
            try:
                size = entry.stat().st_size
                results.append({"path": rel_path, "type": "file", "size": size})
            except Exception:
                results.append({"path": rel_path, "type": "file", "error": True})
    return results
print(json.dumps(walk(sys.argv[1], 1, int(sys.argv[2]), sys.argv[3], sys.argv[4] == 'True')))
"""
            run_cmd = [
                "podman", "exec", "-i", self.podman_container,
                "python3", "-c", python_script,
                str(resolved_path), str(max_depth), str(self.workspace_root), str(recursive)
            ]
            res = subprocess.run(run_cmd, capture_output=True, text=True)
            entries = None
            if res.returncode == 0:
                try:
                    entries = json.loads(res.stdout.strip())
                except Exception:
                    pass
                    
            if entries is None:
                # Fallback to standard POSIX shell script
                shell_script = f"""
find "{resolved_path}" -maxdepth {max_depth if recursive else 1} | while read -r line; do
    if [ "$line" = "{resolved_path}" ]; then continue; fi
    if [ -d "$line" ]; then
        echo "DIR:$line"
    else
        size=$(wc -c < "$line" 2>/dev/null || echo "unknown")
        echo "FILE:$size:$line"
    fi
done
"""
                run_cmd = [
                    "podman", "exec", "-i", self.podman_container,
                    "/bin/sh", "-c", shell_script
                ]
                res = subprocess.run(run_cmd, capture_output=True, text=True)
                entries = []
                if res.returncode == 0:
                    import posixpath
                    for line in res.stdout.splitlines():
                        line = line.strip()
                        if line.startswith("DIR:"):
                            full_path = line[4:]
                            rel = posixpath.relpath(full_path, str(self.workspace_root))
                            entries.append({"path": rel, "type": "dir"})
                        elif line.startswith("FILE:"):
                            parts = line[5:].split(":", 1)
                            if len(parts) == 2:
                                size_str, full_path = parts
                                rel = posixpath.relpath(full_path, str(self.workspace_root))
                                size = int(size_str) if size_str.isdigit() else None
                                entries.append({"path": rel, "type": "file", "size": size})

            results = []
            for entry in entries:
                path_val = entry["path"]
                if entry.get("type") == "dir":
                    results.append(f"{path_val}/ [DIR]")
                elif entry.get("type") == "error":
                    results.append(f"{path_val}/ [Permission Denied]")
                else:
                    size = entry.get("size")
                    if size is not None:
                        results.append(f"{path_val} ({size} bytes)")
                    else:
                        results.append(f"{path_val} [Error reading file details]")
        else:
            if not resolved_path.is_dir():
                raise NotADirectoryError(f"Not a directory: '{path}'")

            results = []

            def _walk(current_dir: Path, current_depth: int):
                if current_depth > max_depth:
                    return

                try:
                    # Sort: directories first, then files alphabetically
                    entries_list = sorted(os.scandir(current_dir), key=lambda e: (not e.is_dir(), e.name.lower()))
                except PermissionError:
                    try:
                        rel_dir = current_dir.relative_to(self.workspace_root)
                    except ValueError:
                        rel_dir = current_dir
                    results.append(f"{rel_dir}/ [Permission Denied]")
                    return

                for entry in entries_list:
                    try:
                        rel_path = Path(entry.path).relative_to(self.workspace_root)
                    except ValueError:
                        rel_path = Path(entry.path)

                    if entry.is_dir():
                        results.append(f"{rel_path}/ [DIR]")
                        if recursive:
                           _walk(Path(entry.path), current_depth + 1)
                    else:
                        try:
                            size = entry.stat().st_size
                            results.append(f"{rel_path} ({size} bytes)")
                        except Exception:
                            results.append(f"{rel_path} [Error reading file details]")

            _walk(resolved_path, 1)

        if not results:
            return f"Directory '{path}' is empty."

        max_entries = 1000
        if len(results) > max_entries:
            truncated = results[:max_entries]
            truncated.append(f"... and {len(results) - max_entries} more items.")
            return "\n".join(truncated)

        return "\n".join(results)

    def exec_cmd(self, command: str, cwd: str = "", timeout: int = 30) -> str:
        """
        Executes a shell command in the specified directory. Confined to workspace.
        
        Args:
            command: The command to execute in the shell.
            cwd: Working directory for the command. If empty, defaults to the workspace root.
            timeout: Maximum time in seconds to wait for execution to complete. Defaults to 30.
        """
        if self.confirm_cmd_callback is not None:
            callback_result = self.confirm_cmd_callback(command)
            if callback_result is False:
                return "Error: Command execution denied by user."
            elif isinstance(callback_result, str):
                return f"Error: Command execution denied by user. Feedback: {callback_result}"

        if self.workspace_type == "podman":
            if cwd:
                run_cwd = self._resolve_path(cwd)
                if not self._container_path_is_dir(str(run_cwd)):
                    raise NotADirectoryError(f"Cwd path is not a directory in container: '{cwd}'")
            else:
                run_cwd = self.workspace_root

            run_args = ["podman", "exec", "-i"]
            if run_cwd:
                run_args.extend(["-w", str(run_cwd)])
            run_args.extend([self.podman_container, "/bin/sh", "-c", command])
            shell_mode = False
        else:
            run_cwd = self.workspace_root
            if cwd:
                run_cwd = self._resolve_path(cwd)
                if not run_cwd.is_dir():
                    raise NotADirectoryError(f"Cwd path is not a directory: '{cwd}'")
            run_args = command
            shell_mode = True

        def _decode_stream(stream) -> str:
            if stream is None:
                return ""
            if isinstance(stream, str):
                return stream
            return stream.decode("utf-8", errors="replace")

        try:
            result = subprocess.run(
                run_args,
                cwd=None if self.workspace_type == "podman" else run_cwd,
                shell=shell_mode,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = [f"Exit Code: {result.returncode}"]
            if result.stdout:
                output.append("Stdout:")
                output.append(result.stdout)
            if result.stderr:
                output.append("Stderr:")
                output.append(result.stderr)
            return "\n".join(output)

        except subprocess.TimeoutExpired as e:
            stdout = _decode_stream(e.stdout)
            stderr = _decode_stream(e.stderr)
            output = [
                f"Error: Command timed out after {timeout} seconds.",
                "Exit Code: Timeout"
            ]
            if stdout:
                output.append("Stdout (captured before timeout):")
                output.append(stdout)
            if stderr:
                output.append("Stderr (captured before timeout):")
                output.append(stderr)
            return "\n".join(output)
        except Exception as e:
            return f"Error executing command: {e}"

    def get_tools(self) -> List[Tool]:
        """
        Returns WorkspaceTools methods as a list of Tool objects ready for the Agent.
        """
        return [
            Tool.from_function(self.read_file),
            Tool.from_function(self.write_file),
            Tool.from_function(self.apply_patch),
            Tool.from_function(self.list_dir),
            Tool.from_function(self.exec_cmd)
        ]


class WebTools:
    """
    Provides a set of web search and extraction tools using the Tavily API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes WebTools with a Tavily API key.
        If not provided, attempts to load the key from the TAVILY_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")

    def _get_client(self) -> "TavilyClient":
        if TavilyClient is None:
            raise ImportError(
                "TavilyClient is unavailable because the 'tavily-python' package is not installed. "
                "Install it using: pip install tavily-python"
            )
        if not self.api_key:
            raise ValueError(
                "Tavily API key is not configured. "
                "Please pass it to WebTools or set the TAVILY_API_KEY environment variable."
            )
        return TavilyClient(api_key=self.api_key)

    def web_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search using the Tavily API to retrieve up-to-date information.

        Args:
            query: The search query string.

        Returns:
            A dictionary containing the search results.
        """
        logger.info(f"Tool call web_search: query='{query}'")
        try:
            client = self._get_client()
            response = client.search(
                query=query,
                search_depth="advanced"
            )
            return response
        except Exception as e:
            logger.error(f"Tavily search failed for query '{query}': {e}", exc_info=True)
            return {"error": f"Search failed: {str(e)}"}

    def web_extract(self, url: str, query: str) -> Dict[str, Any]:
        """
        Extract detailed information from a webpage URL using the Tavily API.

        Args:
            url: The webpage URL to extract content from.
            query: A query describing the relevant information to focus the extraction on.

        Returns:
            A dictionary containing the webpage extract (title, url, and content).
        """
        logger.info(f"Tool call web_extract: url='{url}', query='{query}'")
        try:
            client = self._get_client()
            response = client.extract(
                urls=[url],
                query=query
            )

            results = response.get("results", [])
            extracted_results = []
            for res in results:
                extracted_results.append({
                    "title": res.get("title"),
                    "url": res.get("url"),
                    "content": res.get("raw_content")
                })

            return {
                "results": extracted_results
            }
        except Exception as e:
            logger.error(f"Tavily web_extract failed for url '{url}': {e}", exc_info=True)
            return {"error": f"Extraction failed: {str(e)}"}

    def get_tools(self) -> List[Tool]:
        """
        Returns WebTools methods as a list of Tool objects ready for the Agent.
        """
        return [
            Tool.from_function(self.web_search),
            Tool.from_function(self.web_extract)
        ]
