#!/usr/bin/env python3
"""
Verify system libopus usage:

- Requires pkg-config to find libopus 1.5.2 or newer.
- Verifies Cargo selected the system-lib build-script path rather than CMake.
- Builds and tests every Rust target with the `system-lib` feature.
"""

import os
from pathlib import Path

import ci_utils

MINIMUM_VERSION = "1.5.2"
TARGET_DIR = os.environ.get("CARGO_TARGET_DIR", "target/ci-system-lib")


def verify_pkg_config() -> str:
    ci_utils.run(
        ["pkg-config", "--print-errors", f"--atleast-version={MINIMUM_VERSION}", "opus"],
        capture_output=True,
    )
    version = ci_utils.run(
        ["pkg-config", "--modversion", "opus"], capture_output=True
    ).stdout.strip()
    print(f"pkg-config opus version: {version}")
    return version


def verify_system_build() -> None:
    env = {"CARGO_TARGET_DIR": TARGET_DIR}
    result = ci_utils.run(
        [
            "cargo",
            "test",
            "--no-run",
            "--all-targets",
            "--features",
            "system-lib",
            "--message-format=json-render-diagnostics",
        ],
        env=env,
        capture_output=True,
    )
    message = ci_utils.cargo_root_build_script(result.stdout)
    cfgs = set(message.get("cfgs", []))
    linked_libs = set(message.get("linked_libs", []))
    linked_names = {lib.split("=", 1)[-1] for lib in linked_libs}
    linked_paths = [str(path) for path in message.get("linked_paths", [])]
    out_dir = Path(message["out_dir"])

    if "opus_codec_system_lib" not in cfgs:
        ci_utils.fail("Cargo build script did not emit opus_codec_system_lib")
    if "opus" not in linked_names:
        ci_utils.fail(f"Unexpected system-lib link directives: {sorted(linked_libs)}")
    if any(str(out_dir) in path for path in linked_paths):
        ci_utils.fail(f"System-lib build linked a bundled OUT_DIR path: {linked_paths}")
    if (out_dir / "build" / "CMakeCache.txt").exists():
        ci_utils.fail(f"System-lib build unexpectedly configured bundled CMake in {out_dir}")

    print(f"Verified system-lib build-script path in {out_dir}")
    ci_utils.run(
        ["cargo", "test", "--all-targets", "--features", "system-lib"], env=env
    )


def main() -> None:
    verify_pkg_config()
    verify_system_build()


if __name__ == "__main__":
    main()
