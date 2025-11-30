//! Crate-wide constants and small helpers

use crate::types::SampleRate;

/// Maximum samples per channel in a single Opus frame at 48 kHz.
///
/// 120 ms at 48 kHz = 0.120 * 48000 = 5760 samples.
pub const MAX_FRAME_SAMPLES_48KHZ: usize = 5760;

/// Maximum packet duration in milliseconds.
pub const MAX_PACKET_DURATION_MS: usize = 120;

/// Compute the maximum samples per channel for a frame at the given `sample_rate`.
#[must_use]
pub const fn max_frame_samples_for(sample_rate: SampleRate) -> usize {
    // Scale linearly from the 48 kHz base.
    // sample_rate.as_i32() is always positive given valid SampleRate enum values
    (MAX_FRAME_SAMPLES_48KHZ * (sample_rate as usize)) / 48_000
}
