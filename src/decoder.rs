//! Opus decoder implementation with safe wrappers

#[cfg(feature = "dred")]
use crate::bindings::{
    OPUS_GET_DRED_DURATION_REQUEST, OPUS_SET_DNN_BLOB_REQUEST, OPUS_SET_DRED_DURATION_REQUEST,
};
use crate::bindings::{
    OPUS_GET_FINAL_RANGE_REQUEST, OPUS_GET_GAIN_REQUEST, OPUS_GET_LAST_PACKET_DURATION_REQUEST,
    OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_GET_PITCH_REQUEST,
    OPUS_GET_SAMPLE_RATE_REQUEST, OPUS_RESET_STATE, OPUS_SET_GAIN_REQUEST,
    OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, OpusDecoder, opus_decode, opus_decode_float,
    opus_decoder_create, opus_decoder_ctl, opus_decoder_destroy, opus_decoder_get_nb_samples,
};
use crate::constants::max_frame_samples_for;
use crate::error::{Error, Result};
use crate::packet;
use crate::types::{Bandwidth, Channels, SampleRate};
use std::ptr;

/// Safe wrapper around a libopus `OpusDecoder`.
pub struct Decoder {
    raw: *mut OpusDecoder,
    sample_rate: SampleRate,
    channels: Channels,
}

unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

impl Decoder {
    /// Create a new decoder for a given sample rate and channel layout.
    ///
    /// # Errors
    /// Returns an error if allocation fails or arguments are invalid.
    pub fn new(sample_rate: SampleRate, channels: Channels) -> Result<Self> {
        // Validate sample rate
        if !sample_rate.is_valid() {
            return Err(Error::BadArg);
        }

        let mut error = 0i32;
        let decoder = unsafe {
            opus_decoder_create(
                sample_rate.as_i32(),
                channels.as_i32(),
                std::ptr::addr_of_mut!(error),
            )
        };

        if error != 0 {
            return Err(Error::from_code(error));
        }

        if decoder.is_null() {
            return Err(Error::AllocFail);
        }

        Ok(Self {
            raw: decoder,
            sample_rate,
            channels,
        })
    }

    /// Decode a packet into 16-bit PCM.
    ///
    /// - `input`: Opus packet bytes. Pass empty slice to invoke PLC.
    /// - `output`: Interleaved output buffer sized to `frame_size * channels`.
    /// - `fec`: Enable in-band FEC if available.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is invalid, [`Error::BadArg`]
    /// for invalid buffer sizes or frame sizes, or a mapped libopus error via
    /// [`Error::from_code`].
    pub fn decode(&mut self, input: &[u8], output: &mut [i16], fec: bool) -> Result<usize> {
        // Errors: InvalidState, BadArg, or libopus error mapped.
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }

        // Validate buffer sizes up-front
        if !input.is_empty() && input.len() > i32::MAX as usize {
            return Err(Error::BadArg);
        }
        if output.is_empty() {
            return Err(Error::BadArg);
        }
        if !output.len().is_multiple_of(self.channels.as_usize()) {
            return Err(Error::BadArg);
        }
        let frame_size = output.len() / self.channels.as_usize();
        let max_frame = max_frame_samples_for(self.sample_rate);
        if frame_size == 0 || frame_size > max_frame {
            return Err(Error::BadArg);
        }

        let input_len_i32 = if input.is_empty() {
            0
        } else {
            i32::try_from(input.len()).map_err(|_| Error::BadArg)?
        };
        let frame_size_i32 = i32::try_from(frame_size).map_err(|_| Error::BadArg)?;

        let result = unsafe {
            opus_decode(
                self.raw,
                if input.is_empty() {
                    ptr::null()
                } else {
                    input.as_ptr()
                },
                input_len_i32,
                output.as_mut_ptr(),
                frame_size_i32,
                i32::from(fec),
            )
        };

        if result < 0 {
            return Err(Error::from_code(result));
        }

        usize::try_from(result).map_err(|_| Error::InternalError)
    }

    /// Decode a packet into `f32` PCM.
    ///
    /// See [`Self::decode`] for parameter semantics.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is invalid, [`Error::BadArg`]
    /// for invalid buffer sizes or frame sizes, or a mapped libopus error via
    /// [`Error::from_code`].
    pub fn decode_float(&mut self, input: &[u8], output: &mut [f32], fec: bool) -> Result<usize> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }

        // Validate buffer sizes up-front
        if !input.is_empty() && input.len() > i32::MAX as usize {
            return Err(Error::BadArg);
        }
        if output.is_empty() {
            return Err(Error::BadArg);
        }
        if !output.len().is_multiple_of(self.channels.as_usize()) {
            return Err(Error::BadArg);
        }
        let frame_size = output.len() / self.channels.as_usize();
        let max_frame = max_frame_samples_for(self.sample_rate);
        if frame_size == 0 || frame_size > max_frame {
            return Err(Error::BadArg);
        }

        let input_len_i32 = if input.is_empty() {
            0
        } else {
            i32::try_from(input.len()).map_err(|_| Error::BadArg)?
        };
        let frame_size_i32 = i32::try_from(frame_size).map_err(|_| Error::BadArg)?;

        let result = unsafe {
            opus_decode_float(
                self.raw,
                if input.is_empty() {
                    ptr::null()
                } else {
                    input.as_ptr()
                },
                input_len_i32,
                output.as_mut_ptr(),
                frame_size_i32,
                i32::from(fec),
            )
        };

        if result < 0 {
            return Err(Error::from_code(result));
        }

        usize::try_from(result).map_err(|_| Error::InternalError)
    }

    /// Return the number of samples (per channel) in an Opus `packet` at this decoder's rate.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, [`Error::BadArg`] for
    /// overlong input, or a mapped libopus error.
    pub fn packet_samples(&self, packet: &[u8]) -> Result<usize> {
        // Errors: InvalidState or libopus error mapped.
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }

        if packet.len() > i32::MAX as usize {
            return Err(Error::BadArg);
        }
        let len_i32 = i32::try_from(packet.len()).map_err(|_| Error::BadArg)?;
        let result = unsafe { opus_decoder_get_nb_samples(self.raw, packet.as_ptr(), len_i32) };

        if result < 0 {
            return Err(Error::from_code(result));
        }

        usize::try_from(result).map_err(|_| Error::InternalError)
    }

    /// Return the bandwidth encoded in an Opus `packet`.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or [`Error::InvalidPacket`]
    /// if the packet cannot be parsed.
    pub fn packet_bandwidth(&self, packet: &[u8]) -> Result<Bandwidth> {
        // Errors: InvalidState or InvalidPacket.
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }

        packet::packet_bandwidth(packet)
    }

    /// Return the number of channels described by an Opus `packet`.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or [`Error::InvalidPacket`]
    /// if the packet cannot be parsed.
    pub fn packet_channels(&self, packet: &[u8]) -> Result<Channels> {
        // Errors: InvalidState or InvalidPacket.
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }

        packet::packet_channels(packet)
    }

    /// Reset the decoder to its initial state.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error
    /// if resetting fails.
    pub fn reset(&mut self) -> Result<()> {
        // Errors: InvalidState or request failure.
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }

        // OPUS_RESET_STATE takes no additional argument. Passing extras is undefined behavior.
        let result = unsafe { opus_decoder_ctl(self.raw, OPUS_RESET_STATE as i32) };

        if result != 0 {
            return Err(Error::from_code(result));
        }

        Ok(())
    }

    /// The decoder's configured sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// The decoder's channel configuration.
    #[must_use]
    pub const fn channels(&self) -> Channels {
        self.channels
    }

    #[cfg_attr(not(feature = "dred"), allow(dead_code))]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut OpusDecoder {
        self.raw
    }

    /// Query decoder output sample rate.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error.
    pub fn get_sample_rate(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_SAMPLE_RATE_REQUEST as i32)
    }

    /// Query pitch (fundamental period) of the last decoded frame (in samples at 48 kHz domain).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error.
    pub fn get_pitch(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_PITCH_REQUEST as i32)
    }

    /// Duration (per channel) of the last decoded packet.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error.
    pub fn get_last_packet_duration(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_LAST_PACKET_DURATION_REQUEST as i32)
    }

    /// Final RNG state after the last decode.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error.
    pub fn final_range(&mut self) -> Result<u32> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let mut v: u32 = 0;
        let r = unsafe { opus_decoder_ctl(self.raw, OPUS_GET_FINAL_RANGE_REQUEST as i32, &mut v) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(v)
    }

    /// Set post-decode gain in Q8 dB units.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error.
    pub fn set_gain(&mut self, q8_db: i32) -> Result<()> {
        self.simple_ctl(OPUS_SET_GAIN_REQUEST as i32, q8_db)
    }
    /// Query post-decode gain in Q8 dB units.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error.
    pub fn gain(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_GAIN_REQUEST as i32)
    }

    /// Returns true if phase inversion is disabled (CELT stereo decorrelation).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error.
    pub fn phase_inversion_disabled(&mut self) -> Result<bool> {
        Ok(self.get_int_ctl(OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST as i32)? != 0)
    }

    /// Disable/enable phase inversion (CELT stereo decorrelation).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) -> Result<()> {
        self.simple_ctl(
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST as i32,
            i32::from(disabled),
        )
    }

    #[cfg(feature = "dred")]
    /// Set DRED duration in ms (if libopus built with DRED).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error.
    pub fn set_dred_duration(&mut self, ms: i32) -> Result<()> {
        self.simple_ctl(OPUS_SET_DRED_DURATION_REQUEST as i32, ms)
    }
    #[cfg(feature = "dred")]
    /// Query DRED duration in ms.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error.
    pub fn dred_duration(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_DRED_DURATION_REQUEST as i32)
    }
    #[cfg(feature = "dred")]
    /// Set DNN blob for DRED (feature-gated; will error if unsupported).
    ///
    /// # Safety
    /// Caller must ensure `ptr` is valid for reads as expected by libopus for the duration of the call
    /// and points to a properly formatted DNN blob. Passing an invalid or dangling pointer is UB.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder is invalid, or a mapped libopus error.
    pub unsafe fn set_dnn_blob(&mut self, ptr: *const u8, len: i32) -> Result<()> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        if ptr.is_null() || len <= 0 {
            return Err(Error::BadArg);
        }
        let r = unsafe { opus_decoder_ctl(self.raw, OPUS_SET_DNN_BLOB_REQUEST as i32, ptr, len) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }

    // --- internal helpers for CTLs ---
    fn simple_ctl(&mut self, req: i32, val: i32) -> Result<()> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let r = unsafe { opus_decoder_ctl(self.raw, req, val) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }
    fn get_int_ctl(&mut self, req: i32) -> Result<i32> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let mut v: i32 = 0;
        let r = unsafe { opus_decoder_ctl(self.raw, req, &mut v) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(v)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            opus_decoder_destroy(self.raw);
        }
    }
}
