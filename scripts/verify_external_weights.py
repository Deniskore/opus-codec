#!/usr/bin/env python3
"""Generate a DNN blob and test runtime-loaded DRED weights end to end."""

import argparse
import hashlib
import os
import shlex
import shutil
import tempfile
from pathlib import Path

import ci_utils


ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = os.environ.get("CARGO_TARGET_DIR", "target/ci-external-weights")
MODEL_ARCHIVE = "opus_data-735117b.tar.gz"
MODEL_SHA256 = "8f34305a299183509d22c7ba66790f67916a0fc56028ebd4c8f7b938458f2801"
MODEL_URL = f"https://media.xiph.org/opus/models/{MODEL_ARCHIVE}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model_archive(archive: Path) -> None:
    failures = []
    wget = shutil.which("wget")
    if wget:
        result = ci_utils.run([wget, "-O", str(archive), MODEL_URL], check=False)
        if result.returncode == 0:
            return
        failures.append(f"wget exited with {result.returncode}")
    else:
        failures.append("wget was not found")

    curl = shutil.which("curl")
    if curl:
        result = ci_utils.run(
            [curl, "--fail", "--location", "--output", str(archive), MODEL_URL],
            check=False,
        )
        if result.returncode == 0:
            return
        failures.append(f"curl exited with {result.returncode}")
    else:
        failures.append("curl was not found")

    ci_utils.fail(
        "failed to download the DRED model archive with wget or curl: "
        + "; ".join(failures)
    )


def ensure_model_archive() -> Path:
    archive = ROOT / "opus" / MODEL_ARCHIVE
    if not archive.exists():
        download_model_archive(archive)

    actual = sha256(archive)
    if actual != MODEL_SHA256:
        ci_utils.fail(
            f"DRED model archive checksum mismatch: expected {MODEL_SHA256}, got {actual}"
        )
    return archive


def generate_weights_blob(archive: Path, temp: Path) -> Path:
    ci_utils.run(["tar", "xzf", str(archive), "-C", str(temp)])

    compiler = shutil.which(os.environ.get("CC", "cc"))
    if not compiler:
        ci_utils.fail(f"C compiler {os.environ.get('CC', 'cc')!r} was not found")

    cflags = shlex.split(os.environ.get("CFLAGS", ""))
    dnn = ROOT / "opus" / "dnn"
    command = [
        compiler,
        *cflags,
        "-O2",
        f"-I{ROOT / 'opus'}",
        f"-I{ROOT / 'opus' / 'include'}",
        f"-I{ROOT / 'opus' / 'celt'}",
        f"-I{dnn}",
        f"-I{temp / 'dnn'}",
        str(dnn / "write_lpcnet_weights.c"),
        str(dnn / "parse_lpcnet_weights.c"),
        "-lm",
        "-o",
        str(temp / "dump_weights_blob"),
    ]
    ci_utils.run(command)
    ci_utils.run([str(temp / "dump_weights_blob")], cwd=temp)

    blob = temp / "weights_blob.bin"
    if not blob.is_file() or blob.stat().st_size == 0:
        ci_utils.fail("weight blob generator produced no data")
    print(f"Generated external DNN blob: {blob.stat().st_size} bytes")
    return blob


def run_test(blob: Path, toolchain: str | None, target: str | None) -> None:
    command = ["cargo"]
    if toolchain:
        command.append(f"+{toolchain}")
    command += ["test"]
    if target:
        command += ["--target", target]
    command += [
        "--features",
        "external-weights",
        "--test",
        "external_weights",
        "--",
        "--ignored",
        "--exact",
        "external_weight_dred_round_trip",
    ]

    env = {
        "CARGO_TARGET_DIR": TARGET_DIR,
        "OPUS_CODEC_DNN_BLOB": str(blob),
    }
    ci_utils.run(command, cwd=ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolchain")
    parser.add_argument("--target")
    args = parser.parse_args()

    archive = ensure_model_archive()
    with tempfile.TemporaryDirectory(prefix="opus-external-weights-") as temp_dir:
        blob = generate_weights_blob(archive, Path(temp_dir))
        run_test(blob, args.toolchain, args.target)


if __name__ == "__main__":
    main()
