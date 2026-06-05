import json
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


@contextmanager
def group(name: str):
    """Group output in GitHub Actions."""
    print(f"::group::{name}", flush=True)
    try:
        yield
    finally:
        print("::endgroup::", flush=True)


def _display_command(cmd: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def _print_captured_failure(exc: subprocess.CalledProcessError) -> None:
    if exc.stdout:
        print(exc.stdout, end="" if exc.stdout.endswith("\n") else "\n")
    if exc.stderr:
        print(
            exc.stderr,
            end="" if exc.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )


def run(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Union[str, Path]] = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command with optional grouping and error handling."""
    cmd_str = _display_command(cmd)
    should_group = not capture_output

    if should_group:
        print(f"::group::{cmd_str}", flush=True)

    try:
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        result = subprocess.run(
            cmd,
            env=run_env,
            cwd=cwd,
            check=check,
            text=True,
            capture_output=capture_output,
        )
        return result
    except subprocess.CalledProcessError as exc:
        if should_group:
            print(f"Command failed with exit code {exc.returncode}", flush=True)
        elif capture_output:
            _print_captured_failure(exc)
        raise
    finally:
        if should_group:
            print("::endgroup::", flush=True)


def cargo_json_messages(output: str) -> Iterable[Dict[str, Any]]:
    """Yield Cargo JSON messages, ignoring non-JSON tool and test output."""
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict) and "reason" in message:
            yield message


def cargo_root_package_id(cwd: Optional[Union[str, Path]] = None) -> str:
    """Return the package ID for the workspace's single default package."""
    output = run(
        ["cargo", "metadata", "--no-deps", "--format-version=1"],
        cwd=cwd,
        capture_output=True,
    ).stdout
    metadata = json.loads(output)
    default_members = metadata.get("workspace_default_members", [])
    if len(default_members) != 1:
        fail(
            "Expected exactly one default workspace package, found "
            f"{len(default_members)}"
        )
    return str(default_members[0])


def cargo_root_build_script(
    output: str, cwd: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Return the root package's unique build-script-executed message."""
    package_id = cargo_root_package_id(cwd)
    matches = [
        message
        for message in cargo_json_messages(output)
        if message.get("reason") == "build-script-executed"
        and message.get("package_id") == package_id
    ]
    if len(matches) != 1:
        fail(
            "Expected one root build-script message from Cargo, found "
            f"{len(matches)}"
        )
    return matches[0]


def fail(msg: str) -> None:
    """Exit with an error message."""
    sys.exit(f"Error: {msg}")
