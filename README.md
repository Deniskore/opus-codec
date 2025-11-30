# opus-codec

Safe Rust wrappers around libopus for encoding/decoding Opus audio, with tests that validate core functionality against ffmpeg.

## License

This crate is licensed under either of

- [MIT license](https://opensource.org/licenses/MIT)
- [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

at your option.

## Bundled libopus

The upstream libopus sources are vendored via `git subtree` at tag **v1.5.2** (split commit `ddbe48383984d56acd9e1ab6a090c54ca6b735a6`).
You can verify the copy is pristine by diffing `opus/` against that upstream commit.
