"""Tests for RL environments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.environments import (
    SWEEnvironment, OpenClawEnvironment,
    CodeGenerationTask, DebuggingTask, CodeReviewTask,
    EnvResult,
)


def test_swe_environment():
    env = SWEEnvironment()
    assert len(env.tasks) > 0
    print(f"SWE Environment: {len(env.tasks)} tasks")


def test_env_run_batch():
    env = SWEEnvironment()
    prompts = ["Write a binary search function."]
    results = env.run_batch(prompts)
    assert len(results) == len(prompts)
    for r in results:
        assert isinstance(r, EnvResult)
        assert r.reward >= 0
        assert r.score >= 0
    print(f"Batch run OK: {len(results)} results, avg reward={sum(r.reward for r in results)/len(results):.2f}")


def test_code_generation():
    task = CodeGenerationTask(
        "Write a function that adds two numbers.",
        "assert add(1, 2) == 3",
    )
    score = task.score_response("```python\ndef add(a, b):\n    return a + b\n```")
    print(f"Code generation exact match: score={score}")


def test_code_generation_syntax_error():
    task = CodeGenerationTask(
        "Write a function.",
        "assert foo() == 1",
    )
    score = task.score_response("```python\ndef foo(:\n    return\n```")
    print(f"Code generation syntax error: score={score}")


def test_debugging_task():
    task = DebuggingTask(
        "def divide(a, b): return a / b",
        "ZeroDivisionError",
        "Fix division by zero",
    )
    score = task.score_response("```python\ndef divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n```")
    print(f"Debugging task: score={score}")


def test_code_review():
    task = CodeReviewTask("x = 1\ny = x + '2'", ["type error"])
    score = task.score_response("Issue: type mismatch between int and str")
    print(f"Code review: score={score}")


def test_openclaw_environment():
    env = OpenClawEnvironment()
    assert len(env.tasks) >= len(SWEEnvironment().tasks)
    print(f"OpenClaw Environment: {len(env.tasks)} tasks (>= base SWE)")


def test_empty_reward():
    env = SWEEnvironment()
    results = env.run_batch(["irrelevant text about cooking"])
    for r in results:
        assert r.reward >= 0.0
    print(f"Empty/irrelevant prompt: reward={results[0].reward}")


if __name__ == "__main__":
    test_swe_environment()
    test_env_run_batch()
    test_code_generation()
    test_code_generation_syntax_error()
    test_debugging_task()
    test_code_review()
    test_openclaw_environment()
    test_empty_reward()
    print("\nAll RL tests passed!")
