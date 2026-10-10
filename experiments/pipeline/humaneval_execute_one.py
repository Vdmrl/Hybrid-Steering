"""Run one HumanEval completion inside an OS sandbox; input/output are JSON."""

import contextlib
import json
import os
import resource
import sys


def build_source(payload: dict) -> str:
    """Append the hidden tests after a complete Python line."""
    completion = payload["completion"]
    separator = "" if completion.endswith("\n") else "\n"
    return (
        payload["prompt"]
        + completion
        + separator
        + payload["test"]
        + "\ncheck("
        + payload["entry_point"]
        + ")\n"
    )


def main() -> None:
    payload = json.load(sys.stdin)
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    source = build_source(payload)
    result = {"passed": False, "error": None}
    try:
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            exec(compile(source, "<HumanEval>", "exec"), {})
        result["passed"] = True
    except BaseException as error:
        result["error"] = type(error).__name__
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
