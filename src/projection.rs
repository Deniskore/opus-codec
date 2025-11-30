//! Safe wrappers for the libopus projection (ambisonics) API

use crate::bindings::{
    OPUS_BITRATE_MAX, OPUS_GET_BITRATE_REQUEST, OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST,
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST, OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
    OPUS_SET_BITRATE_REQUEST, OpusProjectionDecoder, OpusProjectionEncoder,
    opus_projection_ambisonics_encoder_create, opus_projection_decode,
    opus_projection_decode_float, opus_projection_decoder_create, opus_projection_decoder_destroy,
    opus_projection_encode, opus_projection_encode_float, opus_projection_encoder_ctl,
    opus_projection_encoder_destroy,
};
use crate::constants::max_frame_samples_for;
use crate::error::{Error, Result};
use crate::types::{Application, Bitrate, SampleRate};

/// Safe wrapper around `OpusProjectionEncoder`.
pub struct ProjectionEncoder {
    raw: *mut OpusProjectionEncoder,
    sample_rate: SampleRate,
    channels: u8,
    streams: u8,
    coupled_streams: u8,
}

unsafe impl Send for ProjectionEncoder {}
unsafe impl Sync for ProjectionEncoder {}

impl ProjectionEncoder {
    /// Create a new projection encoder using the ambisonics helper.
    ///
    /// Returns [`Error::BadArg`] for unsupported channel/mapping combinations
    /// or propagates libopus allocation failures.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] for invalid arguments or the libopus error produced by
    /// the underlying create call; [`Error::AllocFail`] if libopus returns a null handle.
    pub fn new(
        sample_rate: SampleRate,
        channels: u8,
        mapping_family: i32,
        application: Application,
    ) -> Result<Self> {
        let mut err = 0i32;
        let mut streams = 0i32;
        let mut coupled = 0i32;
        let enc = unsafe {
            opus_projection_ambisonics_encoder_create(
                sample_rate as i32,
                i32::from(channels),
                mapping_family,
                &raw mut streams,
                &raw mut coupled,
                application as i32,
                &raw mut err,
            )
        };
        if err != 0 {
            return Err(Error::from_code(err));
        }
        if enc.is_null() {
            return Err(Error::AllocFail);
        }
        Ok(Self {
            raw: enc,
            sample_rate,
            channels,
            streams: u8::try_from(streams).map_err(|_| Error::BadArg)?,
            coupled_streams: u8::try_from(coupled).map_err(|_| Error::BadArg)?,
        })
    }

    fn validate_frame_size(&self, frame_size_per_ch: usize) -> Result<i32> {
        if frame_size_per_ch == 0 || frame_size_per_ch > max_frame_samples_for(self.sample_rate) {
            return Err(Error::BadArg);
        }
        i32::try_from(frame_size_per_ch).map_err(|_| Error::BadArg)
    }

    fn ensure_pcm_layout(&self, len: usize, frame_size_per_ch: usize) -> Result<()> {
        if len != frame_size_per_ch * self.channels as usize {
            return Err(Error::BadArg);
        }
        Ok(())
    }

    /// Encode interleaved `i16` PCM.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle was freed, [`Error::BadArg`] for
    /// buffer/layout issues, the libopus error mapped via [`Error::from_code`], or
    /// [`Error::InternalError`] if libopus reports an impossible packet length.
    pub fn encode(
        &mut self,
        pcm: &[i16],
        frame_size_per_ch: usize,
        out: &mut [u8],
    ) -> Result<usize> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        if out.is_empty() || out.len() > i32::MAX as usize {
            return Err(Error::BadArg);
        }
        self.ensure_pcm_layout(pcm.len(), frame_size_per_ch)?;
        let frame_size = self.validate_frame_size(frame_size_per_ch)?;
        let out_len = i32::try_from(out.len()).map_err(|_| Error::BadArg)?;
        let n = unsafe {
            opus_projection_encode(
                self.raw,
                pcm.as_ptr(),
                frame_size,
                out.as_mut_ptr(),
                out_len,
            )
        };
        if n < 0 {
            return Err(Error::from_code(n));
        }
        usize::try_from(n).map_err(|_| Error::InternalError)
    }

    /// Encode interleaved `f32` PCM.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle was freed, [`Error::BadArg`] for
    /// buffer/layout issues, the libopus error mapped via [`Error::from_code`], or
    /// [`Error::InternalError`] if libopus reports an impossible packet length.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size_per_ch: usize,
        out: &mut [u8],
    ) -> Result<usize> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        if out.is_empty() || out.len() > i32::MAX as usize {
            return Err(Error::BadArg);
        }
        self.ensure_pcm_layout(pcm.len(), frame_size_per_ch)?;
        let frame_size = self.validate_frame_size(frame_size_per_ch)?;
        let out_len = i32::try_from(out.len()).map_err(|_| Error::BadArg)?;
        let n = unsafe {
            opus_projection_encode_float(
                self.raw,
                pcm.as_ptr(),
                frame_size,
                out.as_mut_ptr(),
                out_len,
            )
        };
        if n < 0 {
            return Err(Error::from_code(n));
        }
        usize::try_from(n).map_err(|_| Error::InternalError)
    }

    /// Set target bitrate for the encoder.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is invalid or a mapped libopus error.
    pub fn set_bitrate(&mut self, bitrate: Bitrate) -> Result<()> {
        self.simple_ctl(OPUS_SET_BITRATE_REQUEST as i32, bitrate.value())
    }

    /// Query current bitrate configuration.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is invalid or a mapped libopus error.
    pub fn bitrate(&mut self) -> Result<Bitrate> {
        let v = self.get_int_ctl(OPUS_GET_BITRATE_REQUEST as i32)?;
        Ok(match v {
            x if x == crate::bindings::OPUS_AUTO => Bitrate::Auto,
            x if x == OPUS_BITRATE_MAX => Bitrate::Max,
            other => Bitrate::Custom(other),
        })
    }

    /// Size in bytes of the current demixing matrix.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is invalid or a mapped libopus error.
    pub fn demixing_matrix_size(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST as i32)
    }

    /// Gain (in Q8 dB) of the demixing matrix.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is invalid or a mapped libopus error.
    pub fn demixing_matrix_gain(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST as i32)
    }

    /// Copy the demixing matrix into `out` and return the number of bytes written.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is invalid, [`Error::BufferTooSmall`]
    /// when `out` cannot fit the matrix, a mapped libopus error, or [`Error::InternalError`]
    /// when libopus reports an invalid matrix size.
    pub fn write_demixing_matrix(&mut self, out: &mut [u8]) -> Result<usize> {
        let size = self.demixing_matrix_size()?;
        if size <= 0 {
            return Err(Error::InternalError);
        }
        let needed = usize::try_from(size).map_err(|_| Error::InternalError)?;
        if out.len() < needed {
            return Err(Error::BufferTooSmall);
        }
        let r = unsafe {
            opus_projection_encoder_ctl(
                self.raw,
                OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST as i32,
                out.as_mut_ptr(),
                size,
            )
        };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(needed)
    }

    /// Convenience helper returning the demixing matrix as a newly allocated buffer.
    ///
    /// # Errors
    /// Propagates errors from [`Self::demixing_matrix_size`] and [`Self::write_demixing_matrix`],
    /// including [`Error::InternalError`] if libopus reports impossible sizes.
    pub fn demixing_matrix_bytes(&mut self) -> Result<Vec<u8>> {
        let size = self.demixing_matrix_size()?;
        let len = usize::try_from(size).map_err(|_| Error::InternalError)?;
        let mut buf = vec![0u8; len];
        self.write_demixing_matrix(&mut buf)?;
        Ok(buf)
    }

    /// Number of coded streams.
    #[must_use]
    pub const fn streams(&self) -> u8 {
        self.streams
    }

    /// Number of coupled (stereo) coded streams.
    #[must_use]
    pub const fn coupled_streams(&self) -> u8 {
        self.coupled_streams
    }

    /// Input channels passed to the encoder.
    #[must_use]
    pub const fn channels(&self) -> u8 {
        self.channels
    }

    /// Encoder sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    fn simple_ctl(&mut self, req: i32, val: i32) -> Result<()> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let r = unsafe { opus_projection_encoder_ctl(self.raw, req, val) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }

    fn get_int_ctl(&mut self, req: i32) -> Result<i32> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let mut v = 0i32;
        let r = unsafe { opus_projection_encoder_ctl(self.raw, req, &mut v) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(v)
    }
}

impl Drop for ProjectionEncoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { opus_projection_encoder_destroy(self.raw) };
        }
    }
}

/// Safe wrapper around `OpusProjectionDecoder`.
pub struct ProjectionDecoder {
    raw: *mut OpusProjectionDecoder,
    sample_rate: SampleRate,
    channels: u8,
    streams: u8,
    coupled_streams: u8,
}

unsafe impl Send for ProjectionDecoder {}
unsafe impl Sync for ProjectionDecoder {}

impl ProjectionDecoder {
    /// Create a projection decoder given the demixing matrix provided by the encoder.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] for invalid inputs, `Error::from_code` for libopus failures,
    /// or [`Error::AllocFail`] if libopus returns a null handle.
    pub fn new(
        sample_rate: SampleRate,
        channels: u8,
        streams: u8,
        coupled_streams: u8,
        demixing_matrix: &[u8],
    ) -> Result<Self> {
        if demixing_matrix.is_empty() {
            return Err(Error::BadArg);
        }
        let matrix_len = i32::try_from(demixing_matrix.len()).map_err(|_| Error::BadArg)?;
        let mut err = 0i32;
        let dec = unsafe {
            opus_projection_decoder_create(
                sample_rate as i32,
                i32::from(channels),
                i32::from(streams),
                i32::from(coupled_streams),
                demixing_matrix.as_ptr().cast_mut(),
                matrix_len,
                &raw mut err,
            )
        };
        if err != 0 {
            return Err(Error::from_code(err));
        }
        if dec.is_null() {
            return Err(Error::AllocFail);
        }
        Ok(Self {
            raw: dec,
            sample_rate,
            channels,
            streams,
            coupled_streams,
        })
    }

    fn validate_frame_size(&self, frame_size_per_ch: usize) -> Result<i32> {
        if frame_size_per_ch == 0 || frame_size_per_ch > max_frame_samples_for(self.sample_rate) {
            return Err(Error::BadArg);
        }
        i32::try_from(frame_size_per_ch).map_err(|_| Error::BadArg)
    }

    fn ensure_output_layout(&self, len: usize, frame_size_per_ch: usize) -> Result<()> {
        if len != frame_size_per_ch * self.channels as usize {
            return Err(Error::BadArg);
        }
        Ok(())
    }

    /// Decode into interleaved `i16` PCM.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle was freed, [`Error::BadArg`] for
    /// buffer/layout issues, a mapped libopus error, or [`Error::InternalError`] if libopus
    /// reports an impossible decoded sample count.
    pub fn decode(
        &mut self,
        packet: &[u8],
        out: &mut [i16],
        frame_size_per_ch: usize,
        fec: bool,
    ) -> Result<usize> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        self.ensure_output_layout(out.len(), frame_size_per_ch)?;
        let frame_size = self.validate_frame_size(frame_size_per_ch)?;
        let packet_len = if packet.is_empty() {
            0
        } else {
            i32::try_from(packet.len()).map_err(|_| Error::BadArg)?
        };
        let n = unsafe {
            opus_projection_decode(
                self.raw,
                if packet.is_empty() {
                    std::ptr::null()
                } else {
                    packet.as_ptr()
                },
                packet_len,
                out.as_mut_ptr(),
                frame_size,
                i32::from(fec),
            )
        };
        if n < 0 {
            return Err(Error::from_code(n));
        }
        usize::try_from(n).map_err(|_| Error::InternalError)
    }

    /// Decode into interleaved `f32` PCM.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle was freed, [`Error::BadArg`] for
    /// buffer/layout issues, a mapped libopus error, or [`Error::InternalError`] if libopus
    /// reports an impossible decoded sample count.
    pub fn decode_float(
        &mut self,
        packet: &[u8],
        out: &mut [f32],
        frame_size_per_ch: usize,
        fec: bool,
    ) -> Result<usize> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        self.ensure_output_layout(out.len(), frame_size_per_ch)?;
        let frame_size = self.validate_frame_size(frame_size_per_ch)?;
        let packet_len = if packet.is_empty() {
            0
        } else {
            i32::try_from(packet.len()).map_err(|_| Error::BadArg)?
        };
        let n = unsafe {
            opus_projection_decode_float(
                self.raw,
                if packet.is_empty() {
                    std::ptr::null()
                } else {
                    packet.as_ptr()
                },
                packet_len,
                out.as_mut_ptr(),
                frame_size,
                i32::from(fec),
            )
        };
        if n < 0 {
            return Err(Error::from_code(n));
        }
        usize::try_from(n).map_err(|_| Error::InternalError)
    }

    /// Output channel count.
    #[must_use]
    pub const fn channels(&self) -> u8 {
        self.channels
    }

    /// Number of coded streams expected in the input bitstream.
    #[must_use]
    pub const fn streams(&self) -> u8 {
        self.streams
    }

    /// Number of coupled coded streams expected in the input bitstream.
    #[must_use]
    pub const fn coupled_streams(&self) -> u8 {
        self.coupled_streams
    }

    /// Decoder sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }
}

impl Drop for ProjectionDecoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { opus_projection_decoder_destroy(self.raw) };
        }
    }
}
