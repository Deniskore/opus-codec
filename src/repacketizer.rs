//! Safe wrapper for `OpusRepacketizer` utilities

use crate::bindings::{
    OpusRepacketizer, opus_repacketizer_cat, opus_repacketizer_create, opus_repacketizer_destroy,
    opus_repacketizer_get_nb_frames, opus_repacketizer_init, opus_repacketizer_out,
    opus_repacketizer_out_range,
};
use crate::error::{Error, Result};

/// Repackages Opus frames into packets.
pub struct Repacketizer {
    rp: *mut OpusRepacketizer,
}

unsafe impl Send for Repacketizer {}
unsafe impl Sync for Repacketizer {}

impl Repacketizer {
    /// Create a new repacketizer.
    ///
    /// # Errors
    /// Returns `AllocFail` if allocation fails.
    pub fn new() -> Result<Self> {
        let rp = unsafe { opus_repacketizer_create() };
        if rp.is_null() {
            return Err(Error::AllocFail);
        }
        Ok(Self { rp })
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        unsafe { opus_repacketizer_init(self.rp) };
    }

    /// Add a packet to the current state.
    ///
    /// # Errors
    /// Returns an error if the packet is invalid for the current state.
    pub fn push(&mut self, packet: &[u8]) -> Result<()> {
        if packet.is_empty() {
            return Err(Error::BadArg);
        }
        let len_i32 = i32::try_from(packet.len()).map_err(|_| Error::BadArg)?;
        let r = unsafe { opus_repacketizer_cat(self.rp, packet.as_ptr(), len_i32) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }

    /// Number of frames currently queued.
    #[must_use]
    pub fn frames(&mut self) -> i32 {
        unsafe { opus_repacketizer_get_nb_frames(self.rp) }
    }

    /// Emit a packet containing frames in range [begin, end).
    ///
    /// # Errors
    /// Returns an error if range is invalid or output buffer is too small.
    pub fn out_range(&mut self, begin: i32, end: i32, out: &mut [u8]) -> Result<usize> {
        if out.is_empty() {
            return Err(Error::BadArg);
        }
        if begin < 0 || end <= begin {
            return Err(Error::BadArg);
        }
        let out_len_i32 = i32::try_from(out.len()).map_err(|_| Error::BadArg)?;
        let n = unsafe {
            opus_repacketizer_out_range(self.rp, begin, end, out.as_mut_ptr(), out_len_i32)
        };
        if n < 0 {
            return Err(Error::from_code(n));
        }
        usize::try_from(n).map_err(|_| Error::InternalError)
    }

    /// Emit a packet with all queued frames.
    ///
    /// # Errors
    /// Returns an error if the output buffer is too small.
    pub fn out(&mut self, out: &mut [u8]) -> Result<usize> {
        if out.is_empty() {
            return Err(Error::BadArg);
        }
        let out_len_i32 = i32::try_from(out.len()).map_err(|_| Error::BadArg)?;
        let n = unsafe { opus_repacketizer_out(self.rp, out.as_mut_ptr(), out_len_i32) };
        if n < 0 {
            return Err(Error::from_code(n));
        }
        usize::try_from(n).map_err(|_| Error::InternalError)
    }
}

impl Drop for Repacketizer {
    fn drop(&mut self) {
        unsafe { opus_repacketizer_destroy(self.rp) };
    }
}
