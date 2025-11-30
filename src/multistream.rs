//! Safe wrappers for the Opus Multistream API (surround and channel-mapped streams)

use crate::bindings::{
    OPUS_AUTO, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND, OPUS_BITRATE_MAX,
    OPUS_GET_BANDWIDTH_REQUEST, OPUS_GET_BITRATE_REQUEST, OPUS_GET_COMPLEXITY_REQUEST,
    OPUS_GET_DTX_REQUEST, OPUS_GET_FINAL_RANGE_REQUEST, OPUS_GET_FORCE_CHANNELS_REQUEST,
    OPUS_GET_GAIN_REQUEST, OPUS_GET_IN_DTX_REQUEST, OPUS_GET_INBAND_FEC_REQUEST,
    OPUS_GET_LAST_PACKET_DURATION_REQUEST, OPUS_GET_LOOKAHEAD_REQUEST,
    OPUS_GET_MAX_BANDWIDTH_REQUEST, OPUS_GET_PACKET_LOSS_PERC_REQUEST,
    OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_GET_PITCH_REQUEST,
    OPUS_GET_SAMPLE_RATE_REQUEST, OPUS_GET_SIGNAL_REQUEST, OPUS_GET_VBR_CONSTRAINT_REQUEST,
    OPUS_GET_VBR_REQUEST, OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST,
    OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST, OPUS_RESET_STATE, OPUS_SET_BANDWIDTH_REQUEST,
    OPUS_SET_BITRATE_REQUEST, OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_DTX_REQUEST,
    OPUS_SET_FORCE_CHANNELS_REQUEST, OPUS_SET_GAIN_REQUEST, OPUS_SET_INBAND_FEC_REQUEST,
    OPUS_SET_MAX_BANDWIDTH_REQUEST, OPUS_SET_PACKET_LOSS_PERC_REQUEST,
    OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_SET_SIGNAL_REQUEST,
    OPUS_SET_VBR_CONSTRAINT_REQUEST, OPUS_SET_VBR_REQUEST, OPUS_SIGNAL_MUSIC, OPUS_SIGNAL_VOICE,
    OpusDecoder, OpusEncoder, OpusMSDecoder, OpusMSEncoder, opus_multistream_decode,
    opus_multistream_decode_float, opus_multistream_decoder_create, opus_multistream_decoder_ctl,
    opus_multistream_decoder_destroy, opus_multistream_encode, opus_multistream_encode_float,
    opus_multistream_encoder_create, opus_multistream_encoder_ctl,
    opus_multistream_encoder_destroy, opus_multistream_surround_encoder_create,
};
use crate::error::{Error, Result};
use crate::types::{Application, Bandwidth, Bitrate, Channels, Complexity, SampleRate, Signal};

/// Describes the multistream mapping configuration.
#[derive(Debug, Clone, Copy)]
pub struct Mapping<'a> {
    /// Total input/output channels.
    pub channels: u8,
    /// Number of uncoupled mono streams.
    pub streams: u8,
    /// Number of coupled stereo streams (each counts as 2 channels).
    pub coupled_streams: u8,
    /// Channel-to-stream mapping table (length == channels).
    pub mapping: &'a [u8],
}

impl Mapping<'_> {
    /// Validate that mapping length matches channels.
    fn validate(&self) -> Result<()> {
        let channel_count = usize::from(self.channels);
        if channel_count == 0 {
            return Err(Error::BadArg);
        }
        if self.mapping.len() != channel_count {
            return Err(Error::BadArg);
        }

        let streams = usize::from(self.streams);
        let coupled = usize::from(self.coupled_streams);
        if streams + coupled == 0 {
            return Err(Error::BadArg);
        }
        if streams > channel_count {
            return Err(Error::BadArg);
        }
        if coupled > channel_count / 2 {
            return Err(Error::BadArg);
        }
        let total_streams = streams + coupled;
        let mut assignments = vec![0usize; total_streams];
        for &entry in self.mapping {
            if entry == u8::MAX {
                continue;
            }
            let idx = usize::from(entry);
            if idx >= total_streams {
                return Err(Error::BadArg);
            }
            assignments[idx] += 1;
            if idx < streams {
                if assignments[idx] > 1 {
                    return Err(Error::BadArg);
                }
            } else if assignments[idx] > 2 {
                return Err(Error::BadArg);
            }
        }
        Ok(())
    }
}

/// Safe wrapper around `OpusMSEncoder`.
pub struct MSEncoder {
    raw: *mut OpusMSEncoder,
    sample_rate: SampleRate,
    channels: u8,
    streams: u8,
    coupled_streams: u8,
}

unsafe impl Send for MSEncoder {}
unsafe impl Sync for MSEncoder {}

impl MSEncoder {
    /// Create a new multistream encoder.
    ///
    /// The `mapping.mapping` array describes how input channels are assigned to streams.
    /// See libopus docs for standard surround layouts.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] when the mapping dimensions are inconsistent, or
    /// propagates allocation/configuration failures from libopus.
    pub fn new(sr: SampleRate, app: Application, mapping: Mapping<'_>) -> Result<Self> {
        mapping.validate()?;
        let mut err = 0i32;
        let enc = unsafe {
            opus_multistream_encoder_create(
                sr as i32,
                i32::from(mapping.channels),
                i32::from(mapping.streams),
                i32::from(mapping.coupled_streams),
                mapping.mapping.as_ptr(),
                app as i32,
                std::ptr::addr_of_mut!(err),
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
            sample_rate: sr,
            channels: mapping.channels,
            streams: mapping.streams,
            coupled_streams: mapping.coupled_streams,
        })
    }

    /// Encode interleaved i16 PCM into a multistream Opus packet.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is invalid, [`Error::BadArg`]
    /// for buffer mismatches, or the mapped libopus error code.
    #[allow(clippy::missing_panics_doc)]
    pub fn encode(
        &mut self,
        pcm: &[i16],
        frame_size_per_ch: usize,
        out: &mut [u8],
    ) -> Result<usize> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        if pcm.len() != frame_size_per_ch * self.channels as usize {
            return Err(Error::BadArg);
        }
        if out.is_empty() || out.len() > i32::MAX as usize {
            return Err(Error::BadArg);
        }
        let n = unsafe {
            opus_multistream_encode(
                self.raw,
                pcm.as_ptr(),
                i32::try_from(frame_size_per_ch).map_err(|_| Error::BadArg)?,
                out.as_mut_ptr(),
                i32::try_from(out.len()).map_err(|_| Error::BadArg)?,
            )
        };
        if n < 0 {
            return Err(Error::from_code(n));
        }
        usize::try_from(n).map_err(|_| Error::InternalError)
    }

    /// Encode interleaved f32 PCM into a multistream Opus packet.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is invalid, [`Error::BadArg`]
    /// for buffer mismatches, or the mapped libopus error code.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size_per_ch: usize,
        out: &mut [u8],
    ) -> Result<usize> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        if pcm.len() != frame_size_per_ch * self.channels as usize {
            return Err(Error::BadArg);
        }
        if out.is_empty() || out.len() > i32::MAX as usize {
            return Err(Error::BadArg);
        }
        let n = unsafe {
            opus_multistream_encode_float(
                self.raw,
                pcm.as_ptr(),
                i32::try_from(frame_size_per_ch).map_err(|_| Error::BadArg)?,
                out.as_mut_ptr(),
                i32::try_from(out.len()).map_err(|_| Error::BadArg)?,
            )
        };
        if n < 0 {
            return Err(Error::from_code(n));
        }
        usize::try_from(n).map_err(|_| Error::InternalError)
    }

    /// Final RNG state from the last encode.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] when the encoder handle is null or
    /// propagates the libopus error.
    pub fn final_range(&mut self) -> Result<u32> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let mut v: u32 = 0;
        let r = unsafe {
            opus_multistream_encoder_ctl(self.raw, OPUS_GET_FINAL_RANGE_REQUEST as i32, &mut v)
        };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(v)
    }

    /// Set target bitrate for the encoder.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_bitrate(&mut self, bitrate: Bitrate) -> Result<()> {
        self.simple_ctl(OPUS_SET_BITRATE_REQUEST as i32, bitrate.value())
    }

    /// Query the current bitrate target.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null, [`Error::InternalError`]
    /// if the returned value cannot be represented, or propagates any error reported by
    /// libopus.
    pub fn bitrate(&mut self) -> Result<Bitrate> {
        let v = self.get_int_ctl(OPUS_GET_BITRATE_REQUEST as i32)?;
        Ok(match v {
            x if x == OPUS_AUTO => Bitrate::Auto,
            x if x == OPUS_BITRATE_MAX => Bitrate::Max,
            other => Bitrate::Custom(other),
        })
    }

    /// Set encoder complexity in the range 0..=10.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_complexity(&mut self, complexity: Complexity) -> Result<()> {
        self.simple_ctl(
            OPUS_SET_COMPLEXITY_REQUEST as i32,
            complexity.value() as i32,
        )
    }

    /// Query encoder complexity.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null, [`Error::InternalError`]
    /// if the response is outside the valid range, or propagates any error reported by libopus.
    pub fn complexity(&mut self) -> Result<Complexity> {
        let v = self.get_int_ctl(OPUS_GET_COMPLEXITY_REQUEST as i32)?;
        Ok(Complexity::new(
            u32::try_from(v).map_err(|_| Error::InternalError)?,
        ))
    }

    /// Enable/disable discontinuous transmission (DTX).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_dtx(&mut self, enabled: bool) -> Result<()> {
        self.simple_ctl(OPUS_SET_DTX_REQUEST as i32, i32::from(enabled))
    }

    /// Query whether DTX is enabled.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn dtx(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_DTX_REQUEST as i32)
    }

    /// Query whether the encoder is currently in DTX.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn in_dtx(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_IN_DTX_REQUEST as i32)
    }

    /// Enable/disable in-band FEC generation.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_inband_fec(&mut self, enabled: bool) -> Result<()> {
        self.simple_ctl(OPUS_SET_INBAND_FEC_REQUEST as i32, i32::from(enabled))
    }

    /// Query whether in-band FEC is enabled.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn inband_fec(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_INBAND_FEC_REQUEST as i32)
    }

    /// Set expected packet loss percentage (0..=100).
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] when `perc` is outside `0..=100`, [`Error::InvalidState`] if
    /// the encoder handle is null, or propagates any error reported by libopus.
    pub fn set_packet_loss_perc(&mut self, perc: i32) -> Result<()> {
        if !(0..=100).contains(&perc) {
            return Err(Error::BadArg);
        }
        self.simple_ctl(OPUS_SET_PACKET_LOSS_PERC_REQUEST as i32, perc)
    }

    /// Query expected packet loss percentage.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn packet_loss_perc(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_PACKET_LOSS_PERC_REQUEST as i32)
    }

    /// Enable/disable variable bitrate.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_vbr(&mut self, enabled: bool) -> Result<()> {
        self.simple_ctl(OPUS_SET_VBR_REQUEST as i32, i32::from(enabled))
    }

    /// Query VBR status.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn vbr(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_VBR_REQUEST as i32)
    }

    /// Constrain VBR to reduce instantaneous bitrate swings.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_vbr_constraint(&mut self, constrained: bool) -> Result<()> {
        self.simple_ctl(
            OPUS_SET_VBR_CONSTRAINT_REQUEST as i32,
            i32::from(constrained),
        )
    }

    /// Query VBR constraint flag.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn vbr_constraint(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_VBR_CONSTRAINT_REQUEST as i32)
    }

    /// Set the maximum bandwidth the encoder may use.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_max_bandwidth(&mut self, bw: Bandwidth) -> Result<()> {
        self.simple_ctl(OPUS_SET_MAX_BANDWIDTH_REQUEST as i32, bw as i32)
    }

    /// Query the configured maximum bandwidth.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null, [`Error::InternalError`]
    /// if the value cannot be represented, or propagates any error reported by libopus.
    pub fn max_bandwidth(&mut self) -> Result<Bandwidth> {
        self.get_bandwidth_ctl(OPUS_GET_MAX_BANDWIDTH_REQUEST as i32)
    }

    /// Force a specific output bandwidth (overrides automatic selection).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_bandwidth(&mut self, bw: Bandwidth) -> Result<()> {
        self.simple_ctl(OPUS_SET_BANDWIDTH_REQUEST as i32, bw as i32)
    }

    /// Query the current forced bandwidth, if any.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or [`Error::InternalError`]
    /// if the value is outside the known set, and propagates any error reported by libopus.
    pub fn bandwidth(&mut self) -> Result<Bandwidth> {
        self.get_bandwidth_ctl(OPUS_GET_BANDWIDTH_REQUEST as i32)
    }

    /// Force mono/stereo output for coupled streams, or `None` for automatic.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_force_channels(&mut self, channels: Option<Channels>) -> Result<()> {
        let value = match channels {
            Some(Channels::Mono) => 1,
            Some(Channels::Stereo) => 2,
            None => OPUS_AUTO,
        };
        self.simple_ctl(OPUS_SET_FORCE_CHANNELS_REQUEST as i32, value)
    }

    /// Query forced channel configuration (if any).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn force_channels(&mut self) -> Result<Option<Channels>> {
        let v = self.get_int_ctl(OPUS_GET_FORCE_CHANNELS_REQUEST as i32)?;
        Ok(match v {
            1 => Some(Channels::Mono),
            2 => Some(Channels::Stereo),
            x if x == OPUS_AUTO => None,
            _ => None,
        })
    }

    /// Hint the type of content being encoded (voice/music).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_signal(&mut self, signal: Signal) -> Result<()> {
        self.simple_ctl(OPUS_SET_SIGNAL_REQUEST as i32, signal as i32)
    }

    /// Query the current signal hint.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null, [`Error::InternalError`]
    /// if the response is not recognized, or propagates any error reported by libopus.
    pub fn signal(&mut self) -> Result<Signal> {
        let v = self.get_int_ctl(OPUS_GET_SIGNAL_REQUEST as i32)?;
        match v {
            x if x == OPUS_SIGNAL_VOICE as i32 => Ok(Signal::Voice),
            x if x == OPUS_SIGNAL_MUSIC as i32 => Ok(Signal::Music),
            _ => Err(Error::InternalError),
        }
    }

    /// Query the algorithmic lookahead in samples at 48 kHz.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn lookahead(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_LOOKAHEAD_REQUEST as i32)
    }

    /// Reset the encoder state (retaining configuration).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is null or propagates any error
    /// reported by libopus.
    pub fn reset(&mut self) -> Result<()> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let r = unsafe { opus_multistream_encoder_ctl(self.raw, OPUS_RESET_STATE as i32) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }

    /// Channels of this encoder (interleaved input).
    #[must_use]
    pub const fn channels(&self) -> u8 {
        self.channels
    }
    /// Input sampling rate.
    #[must_use]
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }
    /// Number of mono streams.
    #[must_use]
    pub const fn streams(&self) -> u8 {
        self.streams
    }
    /// Number of coupled streams.
    #[must_use]
    pub const fn coupled_streams(&self) -> u8 {
        self.coupled_streams
    }

    /// Create a multistream encoder using libopus surround mapping helpers.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] for invalid channel counts or the mapped libopus
    /// error when surround initialisation fails.
    pub fn new_surround(
        sr: SampleRate,
        channels: u8,
        mapping_family: i32,
        app: Application,
    ) -> Result<(Self, Vec<u8>)> {
        if channels == 0 {
            return Err(Error::BadArg);
        }
        let mut err = 0i32;
        let mut streams = 0i32;
        let mut coupled = 0i32;
        let mut mapping = vec![0u8; channels as usize];
        let enc = unsafe {
            opus_multistream_surround_encoder_create(
                sr as i32,
                i32::from(channels),
                mapping_family,
                std::ptr::addr_of_mut!(streams),
                std::ptr::addr_of_mut!(coupled),
                mapping.as_mut_ptr(),
                app as i32,
                std::ptr::addr_of_mut!(err),
            )
        };
        if err != 0 {
            return Err(Error::from_code(err));
        }
        if enc.is_null() {
            return Err(Error::AllocFail);
        }
        let streams_u8 = u8::try_from(streams).map_err(|_| Error::BadArg)?;
        let coupled_u8 = u8::try_from(coupled).map_err(|_| Error::BadArg)?;
        Ok((
            Self {
                raw: enc,
                sample_rate: sr,
                channels,
                streams: streams_u8,
                coupled_streams: coupled_u8,
            },
            mapping,
        ))
    }

    /// Borrow a pointer to an individual underlying encoder state for CTLs.
    ///
    /// # Safety
    /// Caller must not outlive the multistream encoder and must ensure the
    /// returned pointer is only used for immediate FFI calls.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder handle is invalid or propagates the
    /// libopus error if retrieving the state fails.
    pub unsafe fn encoder_state_ptr(&mut self, stream_index: i32) -> Result<*mut OpusEncoder> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let mut state: *mut OpusEncoder = std::ptr::null_mut();
        let r = unsafe {
            opus_multistream_encoder_ctl(
                self.raw,
                OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST as i32,
                stream_index,
                &mut state,
            )
        };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        if state.is_null() {
            return Err(Error::InternalError);
        }
        Ok(state)
    }

    fn simple_ctl(&mut self, req: i32, val: i32) -> Result<()> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let r = unsafe { opus_multistream_encoder_ctl(self.raw, req, val) };
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
        let r = unsafe { opus_multistream_encoder_ctl(self.raw, req, &mut v) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(v)
    }

    fn get_bool_ctl(&mut self, req: i32) -> Result<bool> {
        Ok(self.get_int_ctl(req)? != 0)
    }

    fn get_bandwidth_ctl(&mut self, req: i32) -> Result<Bandwidth> {
        let v = u32::try_from(self.get_int_ctl(req)?).map_err(|_| Error::InternalError)?;
        match v {
            x if x == OPUS_BANDWIDTH_NARROWBAND => Ok(Bandwidth::Narrowband),
            x if x == OPUS_BANDWIDTH_MEDIUMBAND => Ok(Bandwidth::Mediumband),
            x if x == OPUS_BANDWIDTH_WIDEBAND => Ok(Bandwidth::Wideband),
            x if x == OPUS_BANDWIDTH_SUPERWIDEBAND => Ok(Bandwidth::SuperWideband),
            x if x == OPUS_BANDWIDTH_FULLBAND => Ok(Bandwidth::Fullband),
            _ => Err(Error::InternalError),
        }
    }
}

impl Drop for MSEncoder {
    fn drop(&mut self) {
        unsafe { opus_multistream_encoder_destroy(self.raw) }
    }
}

/// Safe wrapper around `OpusMSDecoder`.
pub struct MSDecoder {
    raw: *mut OpusMSDecoder,
    sample_rate: SampleRate,
    channels: u8,
}

unsafe impl Send for MSDecoder {}
unsafe impl Sync for MSDecoder {}

impl MSDecoder {
    /// Create a new multistream decoder.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] when the mapping dimensions are inconsistent, or
    /// propagates allocation/configuration failures from libopus.
    pub fn new(sr: SampleRate, mapping: Mapping<'_>) -> Result<Self> {
        mapping.validate()?;
        let mut err = 0i32;
        let dec = unsafe {
            opus_multistream_decoder_create(
                sr as i32,
                i32::from(mapping.channels),
                i32::from(mapping.streams),
                i32::from(mapping.coupled_streams),
                mapping.mapping.as_ptr(),
                std::ptr::addr_of_mut!(err),
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
            sample_rate: sr,
            channels: mapping.channels,
        })
    }

    /// Decode into interleaved i16 PCM (`frame_size` is per-channel).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is invalid, [`Error::BadArg`]
    /// for buffer mismatches, or the mapped libopus error code.
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
        if out.len() != frame_size_per_ch * self.channels as usize {
            return Err(Error::BadArg);
        }
        let n = unsafe {
            opus_multistream_decode(
                self.raw,
                if packet.is_empty() {
                    std::ptr::null()
                } else {
                    packet.as_ptr()
                },
                if packet.is_empty() {
                    0
                } else {
                    i32::try_from(packet.len()).map_err(|_| Error::BadArg)?
                },
                out.as_mut_ptr(),
                i32::try_from(frame_size_per_ch).map_err(|_| Error::BadArg)?,
                i32::from(fec),
            )
        };
        if n < 0 {
            return Err(Error::from_code(n));
        }
        usize::try_from(n).map_err(|_| Error::InternalError)
    }

    /// Decode into interleaved f32 PCM (`frame_size` is per-channel).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is invalid, [`Error::BadArg`]
    /// for buffer mismatches, or the mapped libopus error code.
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
        if out.len() != frame_size_per_ch * self.channels as usize {
            return Err(Error::BadArg);
        }
        let n = unsafe {
            opus_multistream_decode_float(
                self.raw,
                if packet.is_empty() {
                    std::ptr::null()
                } else {
                    packet.as_ptr()
                },
                if packet.is_empty() {
                    0
                } else {
                    i32::try_from(packet.len()).map_err(|_| Error::BadArg)?
                },
                out.as_mut_ptr(),
                i32::try_from(frame_size_per_ch).map_err(|_| Error::BadArg)?,
                i32::from(fec),
            )
        };
        if n < 0 {
            return Err(Error::from_code(n));
        }
        usize::try_from(n).map_err(|_| Error::InternalError)
    }

    /// Final RNG state from the last decode.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] when the decoder handle is null or
    /// propagates the libopus error.
    pub fn final_range(&mut self) -> Result<u32> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let mut v: u32 = 0;
        let r = unsafe {
            opus_multistream_decoder_ctl(self.raw, OPUS_GET_FINAL_RANGE_REQUEST as i32, &mut v)
        };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(v)
    }

    /// Reset the decoder to its initial state.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is null or propagates any error
    /// reported by libopus.
    pub fn reset(&mut self) -> Result<()> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let r = unsafe { opus_multistream_decoder_ctl(self.raw, OPUS_RESET_STATE as i32) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }

    /// Set post-decode gain in Q8 dB units.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_gain(&mut self, q8_db: i32) -> Result<()> {
        self.simple_ctl(OPUS_SET_GAIN_REQUEST as i32, q8_db)
    }

    /// Query post-decode gain in Q8 dB units.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is null or propagates any error
    /// reported by libopus.
    pub fn gain(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_GAIN_REQUEST as i32)
    }

    /// Disable or enable phase inversion (CELT stereo decorrelation).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is null or propagates any error
    /// reported by libopus.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) -> Result<()> {
        self.simple_ctl(
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST as i32,
            i32::from(disabled),
        )
    }

    /// Query the phase inversion disabled flag.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is null or propagates any error
    /// reported by libopus.
    pub fn phase_inversion_disabled(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST as i32)
    }

    /// Query decoder output sample rate.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is null or propagates any error
    /// reported by libopus.
    pub fn get_sample_rate(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_SAMPLE_RATE_REQUEST as i32)
    }

    /// Query the pitch (fundamental period) of the last decoded frame.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is null or propagates any error
    /// reported by libopus.
    pub fn get_pitch(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_PITCH_REQUEST as i32)
    }

    /// Query the duration (per channel) of the last decoded packet.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is null or propagates any error
    /// reported by libopus.
    pub fn get_last_packet_duration(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_LAST_PACKET_DURATION_REQUEST as i32)
    }

    /// Output channels (interleaved).
    #[must_use]
    pub const fn channels(&self) -> u8 {
        self.channels
    }
    /// Output sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// Create a multistream decoder using libopus surround mapping helpers.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] for invalid channel counts or the mapped libopus
    /// error when decoder initialisation fails.
    pub fn new_surround(
        sr: SampleRate,
        channels: u8,
        mapping_family: i32,
    ) -> Result<(Self, Vec<u8>, u8, u8)> {
        if channels == 0 {
            return Err(Error::BadArg);
        }
        let mut err = 0i32;
        let mut streams = 0i32;
        let mut coupled = 0i32;
        let mut mapping = vec![0u8; channels as usize];
        // libopus exposes surround helper creation only for encoders; callers
        // should use the returned mapping/stream counts to configure this decoder.
        let enc = unsafe {
            opus_multistream_surround_encoder_create(
                sr as i32,
                i32::from(channels),
                mapping_family,
                std::ptr::addr_of_mut!(streams),
                std::ptr::addr_of_mut!(coupled),
                mapping.as_mut_ptr(),
                Application::Audio as i32,
                std::ptr::addr_of_mut!(err),
            )
        };
        if !enc.is_null() {
            unsafe { opus_multistream_encoder_destroy(enc) };
        }
        if err != 0 {
            return Err(Error::from_code(err));
        }
        let dec = unsafe {
            opus_multistream_decoder_create(
                sr as i32,
                i32::from(channels),
                streams,
                coupled,
                mapping.as_ptr(),
                std::ptr::addr_of_mut!(err),
            )
        };
        if err != 0 {
            return Err(Error::from_code(err));
        }
        if dec.is_null() {
            return Err(Error::AllocFail);
        }
        Ok((
            Self {
                raw: dec,
                sample_rate: sr,
                channels,
            },
            mapping,
            u8::try_from(streams).map_err(|_| Error::BadArg)?,
            u8::try_from(coupled).map_err(|_| Error::BadArg)?,
        ))
    }

    /// Borrow a pointer to an individual underlying decoder state for CTLs.
    ///
    /// # Safety
    /// Caller must not outlive the multistream decoder and must ensure the
    /// returned pointer is only used for immediate FFI calls.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the decoder handle is invalid or propagates the
    /// libopus error when retrieving the per-stream state fails.
    pub unsafe fn decoder_state_ptr(&mut self, stream_index: i32) -> Result<*mut OpusDecoder> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let mut state: *mut OpusDecoder = std::ptr::null_mut();
        let r = unsafe {
            opus_multistream_decoder_ctl(
                self.raw,
                OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST as i32,
                stream_index,
                &mut state,
            )
        };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        if state.is_null() {
            return Err(Error::InternalError);
        }
        Ok(state)
    }

    fn simple_ctl(&mut self, req: i32, val: i32) -> Result<()> {
        if self.raw.is_null() {
            return Err(Error::InvalidState);
        }
        let r = unsafe { opus_multistream_decoder_ctl(self.raw, req, val) };
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
        let r = unsafe { opus_multistream_decoder_ctl(self.raw, req, &mut v) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(v)
    }

    fn get_bool_ctl(&mut self, req: i32) -> Result<bool> {
        Ok(self.get_int_ctl(req)? != 0)
    }
}

impl Drop for MSDecoder {
    fn drop(&mut self) {
        unsafe { opus_multistream_decoder_destroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_allows_dropped_channels() {
        let mapping = Mapping {
            channels: 6,
            streams: 1,
            coupled_streams: 2,
            mapping: &[0, 1, 1, 2, 2, u8::MAX],
        };
        assert!(mapping.validate().is_ok());
    }

    #[test]
    fn mapping_rejects_duplicate_mono_assignments() {
        let mapping = Mapping {
            channels: 3,
            streams: 1,
            coupled_streams: 1,
            mapping: &[0, 0, 1],
        };
        assert!(mapping.validate().is_err());
    }
}
