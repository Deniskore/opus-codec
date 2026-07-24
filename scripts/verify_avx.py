#!/usr/bin/env python3
"""
Build and verify AVX-presume gating for the bundled opus library.

- The generic build must retain AVX2 runtime dispatch while keeping baseline
  source files free of AVX instructions.
- The presume build must define OPUS_X86_PRESUME_AVX2, apply AVX2/FMA flags to
  every Opus C source, and emit AVX instructions in a baseline source file.

Generic libopus archives intentionally contain separately compiled AVX2
dispatch objects, so the archive as a whole must not be required to be AVX-free.
"""

import platform
import re
import shlex
import shutil
from pathlib import Path

import ci_utils


MAY_HAVE_CACHE_KEY = "OPUS_X86_MAY_HAVE_AVX2"
PRESUME_CACHE_KEY = "OPUS_X86_PRESUME_AVX2"
MAY_HAVE_DEFINE = "-DOPUS_X86_MAY_HAVE_AVX2"
PRESUME_DEFINE = "-DOPUS_X86_PRESUME_AVX2"
RTCD_DEFINE = "-DOPUS_HAVE_RTCD"
REQUIRED_AVX_FLAGS = {"-mavx", "-mavx2", "-mfma"}
BASELINE_OBJECT = Path("CMakeFiles/opus.dir/celt/bands.c.o")
DISPATCH_OBJECT = Path("CMakeFiles/opus.dir/celt/x86/pitch_avx.c.o")


def cache_bool(cache: Path, key: str) -> bool:
    pattern = re.compile(rf"^{re.escape(key)}:BOOL=(ON|OFF)$", re.MULTILINE)
    match = pattern.search(cache.read_text(encoding="utf-8"))
    if not match:
        ci_utils.fail(f"{key}:BOOL was not found in {cache}")
    return match.group(1) == "ON"


def make_variable(path: Path, name: str) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    prefix = f"{name} ="
    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        while value.endswith("\\") and index + 1 < len(lines):
            index += 1
            value = f"{value[:-1]} {lines[index].strip()}"
        return shlex.split(value)
    ci_utils.fail(f"{name} was not found in {path}")


def avx_mnemonics(objdump: str, obj: Path) -> set[str]:
    output = ci_utils.run(
        [objdump, "-d", "--no-show-raw-insn", str(obj)],
        capture_output=True,
    ).stdout
    mnemonics = set()
    instruction = re.compile(r"^\s*[0-9a-f]+:\s+([a-z][a-z0-9.]*)", re.IGNORECASE)
    for line in output.splitlines():
        match = instruction.match(line)
        if match and match.group(1).lower().startswith("v"):
            mnemonics.add(match.group(1).lower())
    return mnemonics


def build(target_dir: str, features: str) -> Path:
    cmd = ["cargo", "build", "--release", "--message-format=json-render-diagnostics"]
    if features:
        cmd += ["--features", features]
    result = ci_utils.run(
        cmd,
        env={
            "CARGO_TARGET_DIR": target_dir,
            "CMAKE_GENERATOR": "Unix Makefiles",
        },
        capture_output=True,
    )
    message = ci_utils.cargo_root_build_script(result.stdout)
    out_dir = Path(message["out_dir"])
    print(f"Cargo reported Opus OUT_DIR: {out_dir}")
    return out_dir


def verify(
    ar: str, objdump: str, target_dir: str, features: str, expect_presume: bool
) -> None:
    with ci_utils.group(f"Build {'presume' if expect_presume else 'generic'} Opus"):
        out_dir = build(target_dir, features)

    cmake_build = out_dir / "build"
    cache = cmake_build / "CMakeCache.txt"
    flags_file = cmake_build / "CMakeFiles/opus.dir/flags.make"
    baseline_obj = cmake_build / BASELINE_OBJECT
    dispatch_obj = cmake_build / DISPATCH_OBJECT
    archive = out_dir / "lib" / "libopus.a"
    missing = [
        path
        for path in (cache, flags_file, baseline_obj, dispatch_obj, archive)
        if not path.is_file()
    ]
    if missing:
        ci_utils.fail("Missing expected build artifacts: " + ", ".join(map(str, missing)))

    archive_members = set(
        ci_utils.run([ar, "t", str(archive)], capture_output=True).stdout.splitlines()
    )
    for member in (BASELINE_OBJECT.name, DISPATCH_OBJECT.name):
        if member not in archive_members:
            ci_utils.fail(f"{member} is missing from final archive {archive}")

    if not cache_bool(cache, MAY_HAVE_CACHE_KEY):
        ci_utils.fail(f"{MAY_HAVE_CACHE_KEY} must be ON in {cache}")
    actual_presume = cache_bool(cache, PRESUME_CACHE_KEY)
    if actual_presume != expect_presume:
        ci_utils.fail(
            f"{PRESUME_CACHE_KEY} mismatch in {cache}: "
            f"expected={expect_presume}, got={actual_presume}"
        )

    definitions = set(make_variable(flags_file, "C_DEFINES"))
    c_flags = set(make_variable(flags_file, "C_FLAGS"))
    if MAY_HAVE_DEFINE not in definitions:
        ci_utils.fail(f"{MAY_HAVE_DEFINE} missing from {flags_file}")
    if (PRESUME_DEFINE in definitions) != expect_presume:
        ci_utils.fail(
            f"{PRESUME_DEFINE} mismatch in {flags_file}: expected={expect_presume}"
        )
    if not expect_presume and RTCD_DEFINE not in definitions:
        ci_utils.fail(f"Generic build is missing runtime dispatch define {RTCD_DEFINE}")

    global_avx_flags = REQUIRED_AVX_FLAGS.intersection(c_flags)
    if expect_presume and global_avx_flags != REQUIRED_AVX_FLAGS:
        ci_utils.fail(
            f"Presume build is missing global flags: "
            f"{sorted(REQUIRED_AVX_FLAGS - global_avx_flags)}"
        )
    if not expect_presume and global_avx_flags:
        ci_utils.fail(
            f"Generic baseline unexpectedly has global AVX flags: {sorted(global_avx_flags)}"
        )

    baseline_mnemonics = avx_mnemonics(objdump, baseline_obj)
    dispatch_mnemonics = avx_mnemonics(objdump, dispatch_obj)
    if not dispatch_mnemonics:
        ci_utils.fail(f"No AVX/VEX instructions found in dispatch object {dispatch_obj}")
    if expect_presume and not baseline_mnemonics:
        ci_utils.fail(
            f"No AVX/VEX instructions found in presume baseline object {baseline_obj}"
        )
    if not expect_presume and baseline_mnemonics:
        ci_utils.fail(
            f"Generic baseline object {baseline_obj} contains AVX/VEX instructions: "
            f"{sorted(baseline_mnemonics)}"
        )
    print(
        f"Verified {'presume' if expect_presume else 'generic'} build; "
        f"baseline AVX mnemonic count={len(baseline_mnemonics)}, "
        f"dispatch AVX mnemonic count={len(dispatch_mnemonics)}"
    )


def main() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "amd64",
        "x86_64",
    }:
        ci_utils.fail("verify_avx.py requires x86-64 Linux")
    objdump = shutil.which("objdump")
    if not objdump:
        ci_utils.fail("GNU objdump was not found on PATH")
    ar = shutil.which("ar")
    if not ar:
        ci_utils.fail("GNU ar was not found on PATH")

    verify(ar, objdump, "target/ci-generic", "", False)
    verify(ar, objdump, "target/ci-presume", "presume-avx2", True)


if __name__ == "__main__":
    main()
