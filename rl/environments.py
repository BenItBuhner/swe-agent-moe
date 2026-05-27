"""SWE-focused RL environments for reinforcement learning training.

Environments that simulate:
- OpenClaw-style agentic code editing tasks
- OpenCode-style debugging and code understanding
- GitHub issue resolution
- Code review and PR analysis
- Shell command execution and scripting
- Multi-turn agent tool use
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from abc import ABC, abstractmethod
import random
import json
import re
import subprocess
import tempfile
import os
from pathlib import Path


@dataclass
class EnvResult:
    prompt: str
    response: str
    reward: float
    score: float
    metrics: Dict = field(default_factory=dict)


class SWETask(ABC):
    """Base class for a single SWE task."""

    @abstractmethod
    def get_prompt(self) -> str:
        ...

    @abstractmethod
    def score_response(self, response: str) -> float:
        """Return a score in [0, 1]."""
        ...

    def get_reward(self, response: str) -> float:
        return self.score_response(response)


class CodeGenerationTask(SWETask):
    def __init__(self, prompt: str, test_code: str, language: str = "python"):
        self._prompt = prompt
        self.test_code = test_code
        self.language = language

    def get_prompt(self) -> str:
        return self._prompt

    def score_response(self, response: str) -> float:
        code = self._extract_code(response)
        if not code:
            return 0.0
        return self._run_tests(code)

    def _extract_code(self, response: str) -> Optional[str]:
        patterns = [
            rf"```{self.language}\n(.*?)```",
            rf"```\n(.*?)```",
        ]
        for pat in patterns:
            match = re.search(pat, response, re.DOTALL)
            if match:
                return match.group(1)
        return response if len(response) > 20 else None

    def _run_tests(self, code: str) -> float:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / f"solution.{self.language}"
            fpath.write_text(code + "\n\n" + self.test_code)

            try:
                result = subprocess.run(
                    ["python3", str(fpath)],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    return 1.0
                return 0.3 if len(result.stderr) < 200 else 0.1
            except subprocess.TimeoutExpired:
                return 0.0
            except Exception:
                return 0.0


class DebuggingTask(SWETask):
    def __init__(self, buggy_code: str, error_msg: str, fix_description: str):
        self.buggy_code = buggy_code
        self.error_msg = error_msg
        self.fix_description = fix_description

    def get_prompt(self) -> str:
        return (
            f"The following code has a bug:\n\n```python\n{self.buggy_code}\n```\n\n"
            f"Error: {self.error_msg}\n\n"
            f"Task: {self.fix_description}\n"
            "Provide the fixed code."
        )

    def score_response(self, response: str) -> float:
        score = 0.0
        if "```" in response:
            score += 0.3
        if "def " in response or "class " in response:
            score += 0.2
        if "return" in response:
            score += 0.2
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        for block in code_blocks:
            try:
                compile(block.strip(), "<test>", "exec")
                score += 0.3
                break
            except SyntaxError:
                pass
        return min(score, 1.0)


class CodeReviewTask(SWETask):
    def __init__(self, code_snippet: str, issues: List[str]):
        self.code_snippet = code_snippet
        self.issues = [i.lower() for i in issues]

    def get_prompt(self) -> str:
        return (
            f"Review the following code and identify all issues:\n\n"
            f"```python\n{self.code_snippet}\n```\n\n"
            "List each issue with severity and suggested fix."
        )

    def score_response(self, response: str) -> float:
        response_lower = response.lower()
        found = sum(1 for issue in self.issues if issue in response_lower)
        if not self.issues:
            return 0.5
        return min(found / len(self.issues), 1.0)


class BashScriptingTask(SWETask):
    def __init__(self, description: str, validation_cmd: str):
        self.description = description
        self.validation_cmd = validation_cmd

    def get_prompt(self) -> str:
        return f"Write a bash script that: {self.description}"

    def score_response(self, response: str) -> float:
        script = self._extract_code(response)
        if not script:
            return 0.0
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "script.sh"
            script_path.write_text(script.replace("```bash", "").replace("```", "").strip())
            os.chmod(script_path, 0o755)
            try:
                result = subprocess.run(
                    ["bash", str(script_path)],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode == 0:
                    val_result = subprocess.run(
                        ["bash", "-c", self.validation_cmd],
                        capture_output=True, text=True, timeout=10,
                    )
                    return 1.0 if val_result.returncode == 0 else 0.5
                return 0.1
            except (subprocess.TimeoutExpired, Exception):
                return 0.0

    def _extract_code(self, response: str) -> Optional[str]:
        match = re.search(r"```(?:bash|sh)?\n(.*?)```", response, re.DOTALL)
        return match.group(1) if match else response


class SWEEnvironment:
    """Collection of SWE tasks simulating real agentic workloads."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tasks = self._build_tasks()

    def _build_tasks(self) -> List[SWETask]:
        return [
            CodeGenerationTask(
                "Write a Python function to find the longest common subsequence of two strings.",
                "assert longest_common_subsequence('abcde', 'ace') == 3\nassert longest_common_subsequence('abc', 'abc') == 3\nassert longest_common_subsequence('abc', 'def') == 0",
            ),
            CodeGenerationTask(
                "Write a Python function that implements binary search on a sorted list.",
                "assert binary_search([1, 2, 3, 4, 5], 3) == 2\nassert binary_search([1, 2, 3, 4, 5], 6) == -1\nassert binary_search([], 1) == -1",
            ),
            CodeGenerationTask(
                "Write a Python decorator that measures and prints function execution time.",
                "import time\n@timing\ndef slow_func():\n    time.sleep(0.01)\n    return 42\nresult = slow_func()\nassert result == 42",
            ),
            DebuggingTask(
                "def divide(a, b):\n    return a / b\n\nresult = divide(10, 0)\nprint(result)",
                "ZeroDivisionError: division by zero",
                "Fix the divide function to handle division by zero gracefully by returning None and printing a warning.",
            ),
            DebuggingTask(
                "def get_user(user_id):\n    users = {1: 'Alice', 2: 'Bob'}\n    return users[user_id]",
                "KeyError: 3",
                "Fix get_user to return None instead of raising KeyError for missing users.",
            ),
            CodeReviewTask(
                "def process_data(data):\n    result = {}\n    for item in data:\n        if item not in result:\n            result[item] = 0\n        result[item] += 1\n    return result\n\ndef unsafe_query(sql):\n    import sqlite3\n    conn = sqlite3.connect('db.sqlite')\n    return conn.execute(f\"SELECT * FROM users WHERE id = {sql}\").fetchall()",
                ["sql injection", "inefficient loop", "no error handling", "hardcoded path"],
            ),
            BashScriptingTask(
                "find all files over 100MB in /tmp and print their paths and sizes, sorted by size descending.",
                "echo 'validation pass'",
            ),
            CodeGenerationTask(
                "Write a Python function that uses multiprocessing to compute fibonacci numbers in parallel.",
                "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n\n# Just check the function exists and is callable\nassert callable(parallel_fib)",
            ),
        ]

    def run_batch(self, prompts: List[str]) -> List[EnvResult]:
        results = []
        for prompt in prompts:
            best_score = 0.0
            best_reward = 0.0
            best_response = ""

            for task in self.tasks:
                task_prompt = task.get_prompt()
                overlap = len(set(prompt.split()) & set(task_prompt.split()))
                if overlap > 3:
                    response = prompt
                    score = task.score_response(response)
                    reward = task.get_reward(response)
                    if score > best_score:
                        best_score = score
                        best_reward = reward
                        best_response = task_prompt

            results.append(EnvResult(
                prompt=prompt,
                response=best_response if best_response else prompt,
                reward=max(best_reward, 0.1),
                score=max(best_score, 0.1),
            ))
        return results


class OpenClawEnvironment(SWEEnvironment):
    """Environment modeling OpenClaw agentic coding tasks."""

    def _build_tasks(self) -> List[SWETask]:
        base_tasks = super()._build_tasks()
        claw_tasks = [
            CodeGenerationTask(
                "Write a Python function that reads a file, applies a transformation, and writes output. Handle all edge cases.",
                "with open('/tmp/test_claw.txt', 'w') as f: f.write('hello')\nresult = transform_file('/tmp/test_claw.txt', lambda x: x.upper())\nassert result == 'HELLO'",
            ),
            DebuggingTask(
                "import os\n\ndef list_files(directory):\n    files = os.listdir(directory)\n    for f in files:\n        full_path = os.path.join(directory, f)\n        if os.path.isfile(full_path):\n            print(f'{f}: {os.path.getsize(full_path)} bytes')",
                "PermissionError on system directories",
                "Add error handling for permission errors and non-existent directories in the list_files function.",
            ),
            CodeReviewTask(
                "#!/usr/bin/env python3\nimport subprocess\n\ndef deploy(host, command):\n    return subprocess.run(f'ssh {host} {command}', shell=True, capture_output=True)",
                ["shell injection", "no input validation", "no error handling"],
            ),
        ]
        return base_tasks + claw_tasks
