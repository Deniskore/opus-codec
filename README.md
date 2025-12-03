# opus-codec

Safe Rust wrappers around libopus for encoding/decoding Opus audio, with tests that validate core functionality against ffmpeg.

## Features

- `presume-avx2`: Build the bundled libopus with `OPUS_X86_PRESUME_AVX2` on x86/x86_64 targets, assuming AVX/AVX2/FMA support. Ignored when linking against a system libopus.
- `dred`: Enable libopus DRED support (downloads the model when building the bundled library). The bundled DRED build currently assumes a Unix-like host with `sh`, `wget`, and `tar`, it is not supported on Windows.
- `system-lib`: Link against a system-provided libopus instead of the bundled sources.

## License

This crate is licensed under either of

- [MIT license](https://opensource.org/licenses/MIT)
- [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

at your option.

## Bundled libopus

The upstream libopus sources are vendored via `git subtree` at tag **v1.5.2** (split commit `ddbe48383984d56acd9e1ab6a090c54ca6b735a6`).
You can verify the copy is pristine by diffing `opus/` against that upstream commit.
