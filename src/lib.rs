//! Safe, ergonomic wrappers around libopus for encoding/decoding Opus audio.
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]

// Borrowed state wrappers must not expose `&mut` access to their lifetime-erased
// owning handle: doing so would let safe code move that handle out with
// `mem::replace`. Generate explicit forwarding methods instead.
macro_rules! delegate_ref_mut_methods {
    ($(
        $(#[$meta:meta])*
        fn $name:ident($($arg:ident: $arg_ty:ty),* $(,)?) -> $ret:ty;
    )*) => {$(
        $(#[$meta])*
        #[doc = "Calls the corresponding operation on this borrowed libopus state."]
        ///
        /// # Errors
        /// Returns the same errors as the corresponding owning-handle method.
        #[inline]
        pub fn $name(&mut self, $($arg: $arg_ty),*) -> $ret {
            self.inner.$name($($arg),*)
        }
    )*};
}

macro_rules! delegate_ref_unsafe_mut_methods {
    ($(
        $(#[$meta:meta])*
        unsafe fn $name:ident($($arg:ident: $arg_ty:ty),* $(,)?) -> $ret:ty;
    )*) => {$(
        $(#[$meta])*
        #[doc = "Calls the corresponding unsafe operation on this borrowed libopus state."]
        ///
        /// # Safety
        /// The safety requirements of the corresponding owning-handle method apply.
        ///
        /// # Errors
        /// Returns the same errors as the corresponding owning-handle method.
        #[inline]
        pub unsafe fn $name(&mut self, $($arg: $arg_ty),*) -> $ret {
            unsafe { self.inner.$name($($arg),*) }
        }
    )*};
}

// Include the generated bindings
#[allow(warnings)]
#[allow(clippy::all)]
mod bindings {
    include!("bindings.rs");
}

mod alloc;
pub mod constants;
pub mod decoder;
#[cfg(feature = "dred")]
/// Deep Redundancy (DRED) decoder support.
pub mod dred;
pub mod encoder;
pub mod error;
pub mod multistream;
pub mod packet;
pub mod projection;
mod raw;
pub mod repacketizer;
pub mod types;

pub use alloc::AlignedBuffer;
pub use constants::{MAX_FRAME_SAMPLES_48KHZ, MAX_PACKET_DURATION_MS, max_frame_samples_for};
pub use decoder::Decoder;
#[cfg(feature = "dred")]
pub use dred::{DredDecoder, DredState};
pub use encoder::Encoder;
pub use error::{Error, Result};
pub use multistream::{Mapping, MultistreamDecoder, MultistreamEncoder};
pub use packet::{
    packet_bandwidth, packet_channels, packet_frame_count, packet_has_lbrr, packet_parse,
    packet_parse_into, packet_sample_count, packet_samples_per_frame, soft_clip,
};
pub use projection::{ProjectionDecoder, ProjectionEncoder};
pub use repacketizer::Repacketizer;
pub use types::{
    Application, Bandwidth, Bitrate, Channels, Complexity, ExpertFrameDuration, FrameSize,
    SampleRate, Signal,
};

#[doc(hidden)]
pub use bindings::*;

pub(crate) use raw::{RawHandle, checked_non_null};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ownership {
    Owned,
    Borrowed,
}

#[inline]
pub(crate) fn opus_ptr_is_aligned(ptr: *const u8) -> bool {
    // libopus aligns internal state to pointer-sized alignment (opus_private.h align()).
    ptr.addr().is_multiple_of(std::mem::align_of::<usize>())
}

/// Returns the bundled libopus version string of this crate.
#[must_use]
pub fn version() -> &'static str {
    "1.5.2"
}

/// Returns the runtime libopus version string from the linked C library.
#[must_use]
pub fn runtime_version() -> &'static str {
    unsafe {
        let ptr = crate::bindings::opus_get_version_string();
        if ptr.is_null() {
            return "";
        }
        std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
    }
}

/// Returns a human-readable string for a libopus error code (via runtime library).
#[must_use]
pub fn strerror(code: i32) -> &'static str {
    unsafe {
        let ptr = crate::bindings::opus_strerror(code);
        if ptr.is_null() {
            return "";
        }
        std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
    }
}
