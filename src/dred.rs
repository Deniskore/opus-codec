//! Safe wrappers for libopus Deep Redundancy (DRED) decoder support.
//! This module is available when the `dred` Cargo feature is enabled.

use crate::bindings::{
    OpusDRED, OpusDREDDecoder, opus_decoder_dred_decode, opus_decoder_dred_decode_float,
    opus_dred_alloc, opus_dred_decoder_create, opus_dred_decoder_ctl, opus_dred_decoder_destroy,
    opus_dred_decoder_get_size, opus_dred_decoder_init, opus_dred_free, opus_dred_get_size,
    opus_dred_parse, opus_dred_process,
};
use crate::constants::max_frame_samples_for;
use crate::decoder::Decoder;
use crate::error::{Error, Result};
use crate::types::SampleRate;

/// Managed handle for libopus `OpusDREDDecoder`.
pub struct DredDecoder {
    raw: *mut OpusDREDDecoder,
}

unsafe impl Send for DredDecoder {}
unsafe impl Sync for DredDecoder {}

impl DredDecoder {
    /// Allocate a new DRED decoder.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocFail`] if allocation fails or a mapped libopus error
    /// when decoder creation does not succeed.
    pub fn new() -> Result<Self> {
        let mut err = 0;
        let ptr = unsafe { opus_dred_decoder_create(std::ptr::addr_of_mut!(err)) };
        if err != 0 {
            return Err(Error::from_code(err));
        }
        if ptr.is_null() {
            return Err(Error::AllocFail);
        }
        Ok(Self { raw: ptr })
    }

    /// Initialize an externally allocated decoder buffer.
    ///
    /// # Safety
    ///
    /// Caller must provide a valid pointer to `opus_dred_decoder_get_size()` bytes.
    ///
    /// # Errors
    ///
    /// Returns a mapped libopus error if initialization fails.
    pub unsafe fn init_raw(ptr: *mut OpusDREDDecoder) -> Result<()> {
        if ptr.is_null() {
            return Err(Error::BadArg);
        }
        let r = unsafe { opus_dred_decoder_init(ptr) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }

    /// Borrow the raw decoder pointer.
    pub fn as_mut_ptr(&mut self) -> *mut OpusDREDDecoder {
        self.raw
    }

    /// Size of a decoder object in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InternalError`] if libopus reports an invalid (negative)
    /// size, indicating a mismatch with the bundled headers.
    pub fn size() -> Result<usize> {
        let raw = unsafe { opus_dred_decoder_get_size() };
        usize::try_from(raw).map_err(|_| Error::InternalError)
    }

    /// Run a control request directly.
    ///
    /// # Safety
    ///
    /// The caller must ensure the request and argument combination is valid for the
    /// underlying libopus build and that `arg` satisfies libopus expectations.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus
    /// error when the control call fails.
    pub unsafe fn ctl<T>(&mut self, request: i32, arg: T) -> Result<()>
    where
        T: Copy,
    {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let r = unsafe { opus_dred_decoder_ctl(self.raw, request, arg) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }

    /// Parse DRED payload and update `state`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidState`] if handles are invalid, [`Error::BadArg`] for
    /// size conversion failures, or a mapped libopus error from [`opus_dred_parse`].
    pub fn parse(
        &mut self,
        state: &mut DredState,
        data: &[u8],
        max_dred_samples: usize,
        sampling_rate: SampleRate,
        dred_end: &mut i32,
        defer_processing: bool,
    ) -> Result<usize> {
        if self.raw.is_null() || state.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let len = i32::try_from(data.len()).map_err(|_| Error::BadArg)?;
        let max_samples = i32::try_from(max_dred_samples).map_err(|_| Error::BadArg)?;
        let result = unsafe {
            opus_dred_parse(
                self.raw,
                state.raw,
                data.as_ptr(),
                len,
                max_samples,
                sampling_rate.as_i32(),
                dred_end,
                i32::from(defer_processing),
            )
        };
        if result < 0 {
            return Err(Error::from_code(result));
        }
        usize::try_from(result).map_err(|_| Error::InternalError)
    }

    /// Complete deferred processing between `src` and `dst` states.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidState`] if pointers are invalid, or a mapped libopus
    /// error when [`opus_dred_process`] fails.
    pub fn process(&mut self, src: &DredState, dst: &mut DredState) -> Result<()> {
        if self.raw.is_null() || src.raw.is_null() || dst.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let r = unsafe { opus_dred_process(self.raw, src.raw, dst.raw) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }

    /// Decode redundancy into i16 PCM using a normal Opus decoder.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidState`] if handles are invalid, [`Error::BadArg`] for
    /// invalid buffer sizing, or a mapped libopus error from
    /// [`opus_decoder_dred_decode`].
    pub fn decode_into_i16(
        &mut self,
        decoder: &mut Decoder,
        state: &DredState,
        dred_offset: i32,
        pcm: &mut [i16],
    ) -> Result<usize> {
        if self.raw.is_null() || state.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let channel_count = decoder.channels().as_usize();
        let frame_size = validate_pcm_frame_len(pcm, channel_count, decoder.sample_rate())?;
        let result = unsafe {
            opus_decoder_dred_decode(
                decoder.as_mut_ptr(),
                state.raw,
                dred_offset,
                pcm.as_mut_ptr(),
                frame_size,
            )
        };
        if result < 0 {
            return Err(Error::from_code(result));
        }
        usize::try_from(result).map_err(|_| Error::InternalError)
    }

    /// Decode redundancy into f32 PCM using a normal Opus decoder.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidState`] if handles are invalid, [`Error::BadArg`] for
    /// invalid buffer sizing, or a mapped libopus error from
    /// [`opus_decoder_dred_decode_float`].
    pub fn decode_into_f32(
        &mut self,
        decoder: &mut Decoder,
        state: &DredState,
        dred_offset: i32,
        pcm: &mut [f32],
    ) -> Result<usize> {
        if self.raw.is_null() || state.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let channel_count = decoder.channels().as_usize();
        let frame_size = validate_pcm_frame_len(pcm, channel_count, decoder.sample_rate())?;
        let result = unsafe {
            opus_decoder_dred_decode_float(
                decoder.as_mut_ptr(),
                state.raw,
                dred_offset,
                pcm.as_mut_ptr(),
                frame_size,
            )
        };
        if result < 0 {
            return Err(Error::from_code(result));
        }
        usize::try_from(result).map_err(|_| Error::InternalError)
    }
}

impl Drop for DredDecoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { opus_dred_decoder_destroy(self.raw) };
        }
    }
}

fn validate_pcm_frame_len<T>(
    pcm: &[T],
    channel_count: usize,
    sample_rate: SampleRate,
) -> Result<i32> {
    if channel_count == 0 {
        return Err(Error::InvalidState);
    }
    if pcm.is_empty() {
        return Err(Error::BadArg);
    }
    if pcm.len() % channel_count != 0 {
        return Err(Error::BadArg);
    }
    let frame_size_per_ch = pcm.len() / channel_count;
    if frame_size_per_ch == 0 || frame_size_per_ch > max_frame_samples_for(sample_rate) {
        return Err(Error::BadArg);
    }
    i32::try_from(frame_size_per_ch).map_err(|_| Error::BadArg)
}

/// Managed handle for libopus `OpusDRED` state.
pub struct DredState {
    raw: *mut OpusDRED,
}

unsafe impl Send for DredState {}
unsafe impl Sync for DredState {}

impl DredState {
    /// Allocate a new DRED state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocFail`] if allocation fails or a mapped libopus error when
    /// creation does not succeed.
    pub fn new() -> Result<Self> {
        let mut err = 0;
        let ptr = unsafe { opus_dred_alloc(std::ptr::addr_of_mut!(err)) };
        if err != 0 {
            return Err(Error::from_code(err));
        }
        if ptr.is_null() {
            return Err(Error::AllocFail);
        }
        Ok(Self { raw: ptr })
    }

    /// Size of a DRED state in bytes.
    ///
    /// # Panics
    ///
    /// Panics if libopus reports a negative size, which would indicate a
    /// mismatch with the bundled headers.
    /// Size of a DRED state in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InternalError`] if libopus reports an invalid (negative)
    /// size, indicating a mismatch with the bundled headers.
    pub fn size() -> Result<usize> {
        let raw = unsafe { opus_dred_get_size() };
        usize::try_from(raw).map_err(|_| Error::InternalError)
    }

    /// Borrow the raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut OpusDRED {
        self.raw
    }
}

impl Drop for DredState {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { opus_dred_free(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_pcm_frame_len_checks_arguments() {
        let pcm = vec![0i16; 4];
        assert!(validate_pcm_frame_len(&pcm, 2, SampleRate::Hz48000).is_ok());

        let err = validate_pcm_frame_len(&pcm, 0, SampleRate::Hz48000).unwrap_err();
        assert_eq!(err, Error::InvalidState);

        let err = validate_pcm_frame_len(&pcm[..3], 2, SampleRate::Hz48000).unwrap_err();
        assert_eq!(err, Error::BadArg);

        let err = validate_pcm_frame_len(&[] as &[i16], 2, SampleRate::Hz48000).unwrap_err();
        assert_eq!(err, Error::BadArg);
    }
}
