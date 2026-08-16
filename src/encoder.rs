//! Opus encoder implementation with safe wrappers

use crate::bindings::{
    OPUS_AUTO, OPUS_BANDWIDTH_FULLBAND, OPUS_BITRATE_MAX, OPUS_GET_BANDWIDTH_REQUEST,
    OPUS_GET_BITRATE_REQUEST, OPUS_GET_COMPLEXITY_REQUEST, OPUS_GET_DTX_REQUEST,
    OPUS_GET_EXPERT_FRAME_DURATION_REQUEST, OPUS_GET_FINAL_RANGE_REQUEST,
    OPUS_GET_FORCE_CHANNELS_REQUEST, OPUS_GET_IN_DTX_REQUEST, OPUS_GET_INBAND_FEC_REQUEST,
    OPUS_GET_LOOKAHEAD_REQUEST, OPUS_GET_LSB_DEPTH_REQUEST, OPUS_GET_MAX_BANDWIDTH_REQUEST,
    OPUS_GET_PACKET_LOSS_PERC_REQUEST, OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_GET_PREDICTION_DISABLED_REQUEST, OPUS_GET_SIGNAL_REQUEST, OPUS_GET_VBR_CONSTRAINT_REQUEST,
    OPUS_GET_VBR_REQUEST, OPUS_SET_BANDWIDTH_REQUEST, OPUS_SET_BITRATE_REQUEST,
    OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_DTX_REQUEST, OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
    OPUS_SET_FORCE_CHANNELS_REQUEST, OPUS_SET_INBAND_FEC_REQUEST, OPUS_SET_LSB_DEPTH_REQUEST,
    OPUS_SET_MAX_BANDWIDTH_REQUEST, OPUS_SET_PACKET_LOSS_PERC_REQUEST,
    OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_SET_PREDICTION_DISABLED_REQUEST,
    OPUS_SET_SIGNAL_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST, OPUS_SET_VBR_REQUEST, OpusEncoder,
    opus_encode, opus_encode_float, opus_encoder_create, opus_encoder_ctl, opus_encoder_destroy,
    opus_encoder_get_size, opus_encoder_init,
};
#[cfg(feature = "dred")]
use crate::bindings::{
    OPUS_GET_DRED_DURATION_REQUEST, OPUS_SET_DNN_BLOB_REQUEST, OPUS_SET_DRED_DURATION_REQUEST,
};
use crate::constants::max_frame_samples_for;
use crate::error::{Error, Result};
use crate::types::{
    Application, Bandwidth, Bitrate, Channels, Complexity, ExpertFrameDuration, SampleRate, Signal,
};
use crate::{AlignedBuffer, Ownership, RawHandle};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::ptr::NonNull;

#[cfg(feature = "dred")]
struct RetainedDnnBlob {
    data: Box<[u32]>,
    len: i32,
}

#[cfg(feature = "dred")]
impl RetainedDnnBlob {
    fn parts(&self) -> (*const u8, i32) {
        (self.data.as_ptr().cast::<u8>(), self.len)
    }
}

/// Safe wrapper around a libopus `OpusEncoder`.
pub struct Encoder {
    raw: RawHandle<OpusEncoder>,
    sample_rate: SampleRate,
    channels: Channels,
    // External-weight builds retain pointers into DNN blobs. Keep each copy,
    // including copies used by failed non-transactional load attempts, until
    // after the C encoder has been destroyed. Field declaration order makes
    // `raw` drop before this storage.
    #[cfg(feature = "dred")]
    dnn_blobs: Vec<RetainedDnnBlob>,
    #[cfg(feature = "dred")]
    active_dnn_blob: Option<usize>,
}

unsafe impl Send for Encoder {}

/// Borrowed wrapper around an encoder state.
///
/// The owning handle cannot be moved out of this borrowed wrapper:
///
/// ```compile_fail
/// use opus_codec::encoder::EncoderRef;
/// use opus_codec::Encoder;
/// fn extract<'a>(state: &mut EncoderRef<'a>, replacement: Encoder) -> Encoder {
///     std::mem::replace(&mut **state, replacement)
/// }
/// ```
pub struct EncoderRef<'a> {
    inner: Encoder,
    #[cfg(feature = "dred")]
    active_dnn_blob: Option<(*const u8, i32)>,
    _marker: PhantomData<&'a mut OpusEncoder>,
}

unsafe impl Send for EncoderRef<'_> {}

impl Encoder {
    fn from_raw(
        ptr: NonNull<OpusEncoder>,
        sample_rate: SampleRate,
        channels: Channels,
        ownership: Ownership,
    ) -> Self {
        Self {
            raw: RawHandle::new(ptr, ownership, opus_encoder_destroy),
            sample_rate,
            channels,
            #[cfg(feature = "dred")]
            dnn_blobs: Vec::new(),
            #[cfg(feature = "dred")]
            active_dnn_blob: None,
        }
    }

    /// Size in bytes of an encoder state for external allocation.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] if the channel count is invalid or libopus reports
    /// an impossible size.
    pub fn size(channels: Channels) -> Result<usize> {
        let raw = unsafe { opus_encoder_get_size(channels.as_i32()) };
        if raw <= 0 {
            return Err(Error::BadArg);
        }
        usize::try_from(raw).map_err(|_| Error::InternalError)
    }

    /// Initialize a previously allocated encoder state.
    ///
    /// # Safety
    /// The caller must provide a valid pointer to `Encoder::size()` bytes,
    /// aligned to at least `align_of::<usize>()` (malloc-style alignment).
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] if `ptr` is null, or a mapped libopus error.
    pub unsafe fn init_in_place(
        ptr: *mut OpusEncoder,
        sample_rate: SampleRate,
        channels: Channels,
        application: Application,
    ) -> Result<()> {
        if ptr.is_null() {
            return Err(Error::BadArg);
        }
        if !crate::opus_ptr_is_aligned(ptr.cast()) {
            return Err(Error::BadArg);
        }
        let r = unsafe {
            opus_encoder_init(
                ptr,
                sample_rate.as_i32(),
                channels.as_i32(),
                application as i32,
            )
        };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }

    /// Create a new encoder.
    ///
    /// # Errors
    /// Returns an error if allocation fails or arguments are invalid.
    pub fn new(
        sample_rate: SampleRate,
        channels: Channels,
        application: Application,
    ) -> Result<Self> {
        // Validate sample rate
        if !sample_rate.is_valid() {
            return Err(Error::BadArg);
        }

        let mut error = 0i32;
        let encoder = unsafe {
            opus_encoder_create(
                sample_rate.as_i32(),
                channels.as_i32(),
                application as i32,
                std::ptr::addr_of_mut!(error),
            )
        };

        if error != 0 {
            return Err(Error::from_code(error));
        }

        let encoder = NonNull::new(encoder).ok_or(Error::AllocFail)?;

        Ok(Self::from_raw(
            encoder,
            sample_rate,
            channels,
            Ownership::Owned,
        ))
    }

    /// Encode 16-bit PCM into an Opus packet.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, [`Error::BadArg`] for
    /// invalid buffer sizes or frame size, or a mapped libopus error.
    pub fn encode(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize> {
        // Validate input buffer size
        if input.is_empty() {
            return Err(Error::BadArg);
        }

        // Ensure input buffer is properly sized for the number of channels
        if !input.len().is_multiple_of(self.channels.as_usize()) {
            return Err(Error::BadArg);
        }

        let frame_size = input.len() / self.channels.as_usize();
        let frame_size = NonZeroUsize::new(frame_size).ok_or(Error::BadArg)?;
        // Validate frame size is within Opus limits for the configured sample rate
        if frame_size.get() > max_frame_samples_for(self.sample_rate) {
            return Err(Error::BadArg);
        }

        // Validate output buffer size
        if output.is_empty() {
            return Err(Error::BadArg);
        }
        if output.len() > i32::MAX as usize {
            return Err(Error::BadArg);
        }

        let frame_size_i32 = i32::try_from(frame_size.get()).map_err(|_| Error::BadArg)?;
        let out_len_i32 = i32::try_from(output.len()).map_err(|_| Error::BadArg)?;
        let result = unsafe {
            opus_encode(
                self.raw.as_ptr(),
                input.as_ptr(),
                frame_size_i32,
                output.as_mut_ptr(),
                out_len_i32,
            )
        };

        if result < 0 {
            return Err(Error::from_code(result));
        }

        usize::try_from(result).map_err(|_| Error::InternalError)
    }

    /// Encode 16-bit PCM, capping output to `max_data_bytes`.
    ///
    /// This uses the normal libopus `opus_encode` path and passes `max_data_bytes`
    /// as the output limit, so it constrains packet size without enabling a separate
    /// encoder mode. It does not itself enable FEC; use `set_inband_fec(true)` and
    /// `set_packet_loss_perc(…)` to actually make the encoder produce FEC.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, [`Error::BadArg`] for
    /// invalid buffer sizes or frame size, or a mapped libopus error.
    pub fn encode_limited(
        &mut self,
        input: &[i16],
        output: &mut [u8],
        max_data_bytes: usize,
    ) -> Result<usize> {
        // Validate input buffer size
        if input.is_empty() {
            return Err(Error::BadArg);
        }

        // Ensure input buffer is properly sized for the number of channels
        if !input.len().is_multiple_of(self.channels.as_usize()) {
            return Err(Error::BadArg);
        }

        let frame_size = input.len() / self.channels.as_usize();
        let frame_size = NonZeroUsize::new(frame_size).ok_or(Error::BadArg)?;
        // Validate frame size is within Opus limits for the configured sample rate
        if frame_size.get() > max_frame_samples_for(self.sample_rate) {
            return Err(Error::BadArg);
        }

        // Validate output buffer size
        if output.is_empty() {
            return Err(Error::BadArg);
        }
        if output.len() > i32::MAX as usize {
            return Err(Error::BadArg);
        }
        // Validate max_data_bytes parameter
        if max_data_bytes == 0 || max_data_bytes > output.len() {
            return Err(Error::BadArg);
        }

        let frame_size_i32 = i32::try_from(frame_size.get()).map_err(|_| Error::BadArg)?;
        let max_bytes_i32 = i32::try_from(max_data_bytes).map_err(|_| Error::BadArg)?;
        let result = unsafe {
            opus_encode(
                self.raw.as_ptr(),
                input.as_ptr(),
                frame_size_i32,
                output.as_mut_ptr(),
                max_bytes_i32,
            )
        };

        if result < 0 {
            return Err(Error::from_code(result));
        }

        usize::try_from(result).map_err(|_| Error::InternalError)
    }

    /// Encode f32 PCM into an Opus packet.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, [`Error::BadArg`] for
    /// invalid buffer sizes or frame size, or a mapped libopus error.
    pub fn encode_float(&mut self, input: &[f32], output: &mut [u8]) -> Result<usize> {
        if input.is_empty() {
            return Err(Error::BadArg);
        }
        if !input.len().is_multiple_of(self.channels.as_usize()) {
            return Err(Error::BadArg);
        }
        let frame_size = input.len() / self.channels.as_usize();
        let frame_size = NonZeroUsize::new(frame_size).ok_or(Error::BadArg)?;
        if frame_size.get() > max_frame_samples_for(self.sample_rate) {
            return Err(Error::BadArg);
        }
        if output.is_empty() || output.len() > i32::MAX as usize {
            return Err(Error::BadArg);
        }
        let frame_i32 = i32::try_from(frame_size.get()).map_err(|_| Error::BadArg)?;
        let out_len_i32 = i32::try_from(output.len()).map_err(|_| Error::BadArg)?;
        let n = unsafe {
            opus_encode_float(
                self.raw.as_ptr(),
                input.as_ptr(),
                frame_i32,
                output.as_mut_ptr(),
                out_len_i32,
            )
        };
        if n < 0 {
            return Err(Error::from_code(n));
        }
        usize::try_from(n).map_err(|_| Error::InternalError)
    }

    // ===== Common encoder CTLs =====

    /// Enable/disable in-band FEC generation (decoder can recover from losses).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_inband_fec(&mut self, enabled: bool) -> Result<()> {
        self.simple_ctl(OPUS_SET_INBAND_FEC_REQUEST as i32, i32::from(enabled))
    }
    /// Query in-band FEC setting.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn inband_fec(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_INBAND_FEC_REQUEST as i32)
    }

    /// Hint expected packet loss percentage [0..=100].
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, [`Error::BadArg`] for out-of-range values,
    /// or a mapped libopus error.
    pub fn set_packet_loss_perc(&mut self, perc: i32) -> Result<()> {
        if !(0..=100).contains(&perc) {
            return Err(Error::BadArg);
        }
        self.simple_ctl(OPUS_SET_PACKET_LOSS_PERC_REQUEST as i32, perc)
    }
    /// Query packet loss percentage hint.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn packet_loss_perc(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_PACKET_LOSS_PERC_REQUEST as i32)
    }

    /// Enable/disable DTX (discontinuous transmission).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_dtx(&mut self, enabled: bool) -> Result<()> {
        self.simple_ctl(OPUS_SET_DTX_REQUEST as i32, i32::from(enabled))
    }
    /// Query DTX setting.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn dtx(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_DTX_REQUEST as i32)
    }
    /// Returns true if encoder is currently in DTX.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn in_dtx(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_IN_DTX_REQUEST as i32)
    }

    /// Constrain VBR to reduce instant bitrate swings.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_vbr_constraint(&mut self, constrained: bool) -> Result<()> {
        self.simple_ctl(
            OPUS_SET_VBR_CONSTRAINT_REQUEST as i32,
            i32::from(constrained),
        )
    }
    /// Query VBR constraint setting.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn vbr_constraint(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_VBR_CONSTRAINT_REQUEST as i32)
    }

    /// Set maximum audio bandwidth the encoder may use.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_max_bandwidth(&mut self, bw: Bandwidth) -> Result<()> {
        self.simple_ctl(OPUS_SET_MAX_BANDWIDTH_REQUEST as i32, bw as i32)
    }
    /// Query max bandwidth.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn max_bandwidth(&mut self) -> Result<Bandwidth> {
        self.get_bandwidth_ctl(OPUS_GET_MAX_BANDWIDTH_REQUEST as i32)
    }

    /// Force a specific bandwidth (overrides automatic).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_bandwidth(&mut self, bw: Bandwidth) -> Result<()> {
        self.simple_ctl(OPUS_SET_BANDWIDTH_REQUEST as i32, bw as i32)
    }
    /// Query current forced bandwidth.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn bandwidth(&mut self) -> Result<Bandwidth> {
        self.get_bandwidth_ctl(OPUS_GET_BANDWIDTH_REQUEST as i32)
    }

    /// Force mono/stereo output, or None for automatic.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_force_channels(&mut self, channels: Option<Channels>) -> Result<()> {
        let val = match channels {
            Some(Channels::Mono) => 1,
            Some(Channels::Stereo) => 2,
            None => crate::bindings::OPUS_AUTO,
        };
        self.simple_ctl(OPUS_SET_FORCE_CHANNELS_REQUEST as i32, val)
    }
    /// Query forced channels.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn force_channels(&mut self) -> Result<Option<Channels>> {
        let v = self.get_int_ctl(OPUS_GET_FORCE_CHANNELS_REQUEST as i32)?;
        Ok(match v {
            1 => Some(Channels::Mono),
            2 => Some(Channels::Stereo),
            _ => None,
        })
    }

    /// Hint content type (voice or music).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_signal(&mut self, signal: Signal) -> Result<()> {
        self.simple_ctl(OPUS_SET_SIGNAL_REQUEST as i32, signal as i32)
    }
    /// Query current signal hint.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, [`Error::InternalError`] if the
    /// response is not recognized, or a mapped libopus error.
    pub fn signal(&mut self) -> Result<Signal> {
        let v = self.get_int_ctl(OPUS_GET_SIGNAL_REQUEST as i32)?;
        match v {
            x if x == OPUS_AUTO => Ok(Signal::Auto),
            x if x == crate::bindings::OPUS_SIGNAL_VOICE as i32 => Ok(Signal::Voice),
            x if x == crate::bindings::OPUS_SIGNAL_MUSIC as i32 => Ok(Signal::Music),
            _ => Err(Error::InternalError),
        }
    }

    /// Encoder algorithmic lookahead in samples at this encoder's configured sample rate.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn lookahead(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_LOOKAHEAD_REQUEST as i32)
    }
    /// Final RNG state from the last encode (debugging/bitstream id).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn final_range(&mut self) -> Result<u32> {
        let mut val: u32 = 0;
        let r = unsafe {
            opus_encoder_ctl(
                self.raw.as_ptr(),
                OPUS_GET_FINAL_RANGE_REQUEST as i32,
                &mut val,
            )
        };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(val)
    }

    /// Set input LSB depth (typically 16-24 bits).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, [`Error::BadArg`] for an
    /// out-of-range bit depth, or a mapped libopus error.
    pub fn set_lsb_depth(&mut self, bits: i32) -> Result<()> {
        if !(8..=24).contains(&bits) {
            return Err(Error::BadArg);
        }
        self.simple_ctl(OPUS_SET_LSB_DEPTH_REQUEST as i32, bits)
    }
    /// Query input LSB depth.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn lsb_depth(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_LSB_DEPTH_REQUEST as i32)
    }

    /// Set expert frame duration choice.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_expert_frame_duration(&mut self, dur: ExpertFrameDuration) -> Result<()> {
        self.simple_ctl(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST as i32, dur as i32)
    }
    /// Query expert frame duration.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, [`Error::InternalError`] if the
    /// response is not recognized, or a mapped libopus error.
    pub fn expert_frame_duration(&mut self) -> Result<ExpertFrameDuration> {
        let v = self.get_int_ctl(OPUS_GET_EXPERT_FRAME_DURATION_REQUEST as i32)?;
        let vu = u32::try_from(v).map_err(|_| Error::InternalError)?;
        match vu {
            x if x == crate::bindings::OPUS_FRAMESIZE_ARG => Ok(ExpertFrameDuration::Auto),
            x if x == crate::bindings::OPUS_FRAMESIZE_2_5_MS => Ok(ExpertFrameDuration::Ms2_5),
            x if x == crate::bindings::OPUS_FRAMESIZE_5_MS => Ok(ExpertFrameDuration::Ms5),
            x if x == crate::bindings::OPUS_FRAMESIZE_10_MS => Ok(ExpertFrameDuration::Ms10),
            x if x == crate::bindings::OPUS_FRAMESIZE_20_MS => Ok(ExpertFrameDuration::Ms20),
            x if x == crate::bindings::OPUS_FRAMESIZE_40_MS => Ok(ExpertFrameDuration::Ms40),
            x if x == crate::bindings::OPUS_FRAMESIZE_60_MS => Ok(ExpertFrameDuration::Ms60),
            x if x == crate::bindings::OPUS_FRAMESIZE_80_MS => Ok(ExpertFrameDuration::Ms80),
            x if x == crate::bindings::OPUS_FRAMESIZE_100_MS => Ok(ExpertFrameDuration::Ms100),
            x if x == crate::bindings::OPUS_FRAMESIZE_120_MS => Ok(ExpertFrameDuration::Ms120),
            _ => Err(Error::InternalError),
        }
    }

    /// Disable/enable inter-frame prediction (expert option).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_prediction_disabled(&mut self, disabled: bool) -> Result<()> {
        self.simple_ctl(
            OPUS_SET_PREDICTION_DISABLED_REQUEST as i32,
            i32::from(disabled),
        )
    }
    /// Query prediction disabled flag.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn prediction_disabled(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_PREDICTION_DISABLED_REQUEST as i32)
    }

    /// Disable/enable phase inversion (stereo decorrelation) in CELT (expert option).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) -> Result<()> {
        self.simple_ctl(
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST as i32,
            i32::from(disabled),
        )
    }
    /// Query phase inversion disabled flag.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn phase_inversion_disabled(&mut self) -> Result<bool> {
        self.get_bool_ctl(OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST as i32)
    }

    #[cfg(feature = "dred")]
    /// Set the maximum number of 10-ms Deep Redundancy (DRED) frames.
    ///
    /// A value of zero disables DRED. Libopus validates the supported upper bound.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] if `frames_10ms` is outside the supported range, or a mapped
    /// libopus error if DRED is unavailable in the linked encoder.
    pub fn set_dred_duration(&mut self, frames_10ms: i32) -> Result<()> {
        self.simple_ctl(OPUS_SET_DRED_DURATION_REQUEST as i32, frames_10ms)
    }

    #[cfg(feature = "dred")]
    /// Query the configured maximum number of 10-ms DRED frames.
    ///
    /// # Errors
    /// Returns a mapped libopus error if the encoder is invalid or DRED is unavailable.
    pub fn dred_duration(&mut self) -> Result<i32> {
        self.get_int_ctl(OPUS_GET_DRED_DURATION_REQUEST as i32)
    }

    #[cfg(feature = "dred")]
    /// Load an external DNN model blob into this encoder.
    ///
    /// # Safety
    /// `ptr` must be valid for reads of `len` bytes for the duration of this call and point to a
    /// complete, correctly formatted libopus DNN blob. The bytes are copied into aligned storage
    /// owned by the encoder, so the caller's allocation need not remain alive after this returns.
    /// Some external-weight libopus builds do not safely handle malformed model records.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] if `ptr` is null or `len` is non-positive, or a mapped libopus
    /// error when loading fails. Embedded-weight libopus builds return [`Error::Unimplemented`].
    pub unsafe fn set_dnn_blob(&mut self, ptr: *const u8, len: i32) -> Result<()> {
        let blob_index = unsafe { self.retain_dnn_blob_copy(ptr, len)? };
        let (owned_ptr, owned_len) = self.dnn_blobs[blob_index].parts();
        if let Err(error) = unsafe { self.apply_dnn_blob(owned_ptr, owned_len) } {
            if error == Error::Unimplemented {
                // An unsupported CTL never inspected or retained the pointer.
                // Other failures may leave model fields pointing into the blob.
                let removed = self.dnn_blobs.pop();
                debug_assert!(removed.is_some());
            }
            return Err(error);
        }
        self.active_dnn_blob = Some(blob_index);
        Ok(())
    }

    #[cfg(feature = "dred")]
    unsafe fn retain_dnn_blob_copy(&mut self, ptr: *const u8, len: i32) -> Result<usize> {
        if ptr.is_null() || len <= 0 {
            return Err(Error::BadArg);
        }
        let byte_len = usize::try_from(len).map_err(|_| Error::BadArg)?;
        let word_len = byte_len.div_ceil(std::mem::size_of::<u32>());
        let mut blob = vec![0u32; word_len].into_boxed_slice();
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, blob.as_mut_ptr().cast::<u8>(), byte_len);
        }
        let index = self.dnn_blobs.len();
        self.dnn_blobs.push(RetainedDnnBlob { data: blob, len });
        Ok(index)
    }

    #[cfg(feature = "dred")]
    unsafe fn apply_dnn_blob(&mut self, ptr: *const u8, len: i32) -> Result<()> {
        let r = unsafe {
            opus_encoder_ctl(
                self.raw.as_ptr(),
                OPUS_SET_DNN_BLOB_REQUEST as i32,
                ptr,
                len,
            )
        };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }

    #[cfg(feature = "dred")]
    fn reload_active_dnn_blob(&mut self) -> Result<()> {
        let Some(index) = self.active_dnn_blob else {
            return Ok(());
        };
        let (ptr, len) = self.dnn_blobs[index].parts();
        unsafe { self.apply_dnn_blob(ptr, len) }
    }

    // --- internal helpers ---
    fn simple_ctl(&mut self, req: i32, val: i32) -> Result<()> {
        let r = unsafe { opus_encoder_ctl(self.raw.as_ptr(), req, val) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }
    fn get_bool_ctl(&mut self, req: i32) -> Result<bool> {
        Ok(self.get_int_ctl(req)? != 0)
    }
    fn get_int_ctl(&mut self, req: i32) -> Result<i32> {
        let mut v: i32 = 0;
        let r = unsafe { opus_encoder_ctl(self.raw.as_ptr(), req, &mut v) };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(v)
    }
    fn get_bandwidth_ctl(&mut self, req: i32) -> Result<Bandwidth> {
        let v = self.get_int_ctl(req)?;
        let vu = u32::try_from(v).map_err(|_| Error::InternalError)?;
        match vu {
            x if x == crate::bindings::OPUS_BANDWIDTH_NARROWBAND => Ok(Bandwidth::Narrowband),
            x if x == crate::bindings::OPUS_BANDWIDTH_MEDIUMBAND => Ok(Bandwidth::Mediumband),
            x if x == crate::bindings::OPUS_BANDWIDTH_WIDEBAND => Ok(Bandwidth::Wideband),
            x if x == crate::bindings::OPUS_BANDWIDTH_SUPERWIDEBAND => Ok(Bandwidth::SuperWideband),
            x if x == OPUS_BANDWIDTH_FULLBAND => Ok(Bandwidth::Fullband),
            _ => Err(Error::InternalError),
        }
    }

    /// Set target bitrate.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_bitrate(&mut self, bitrate: Bitrate) -> Result<()> {
        let result = unsafe {
            opus_encoder_ctl(
                self.raw.as_ptr(),
                OPUS_SET_BITRATE_REQUEST as i32,
                bitrate.value(),
            )
        };

        if result != 0 {
            return Err(Error::from_code(result));
        }

        Ok(())
    }

    /// Query current bitrate.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn bitrate(&mut self) -> Result<Bitrate> {
        let mut bitrate = 0i32;
        let result = unsafe {
            opus_encoder_ctl(
                self.raw.as_ptr(),
                OPUS_GET_BITRATE_REQUEST as i32,
                &mut bitrate,
            )
        };

        if result != 0 {
            return Err(Error::from_code(result));
        }

        match bitrate {
            OPUS_AUTO => Ok(Bitrate::Auto),
            OPUS_BITRATE_MAX => Ok(Bitrate::Max),
            bps => Ok(Bitrate::Custom(bps)),
        }
    }

    /// Set encoder complexity [0..=10].
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_complexity(&mut self, complexity: Complexity) -> Result<()> {
        let result = unsafe {
            opus_encoder_ctl(
                self.raw.as_ptr(),
                OPUS_SET_COMPLEXITY_REQUEST as i32,
                complexity.value() as i32,
            )
        };

        if result != 0 {
            return Err(Error::from_code(result));
        }

        Ok(())
    }

    /// Query encoder complexity.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn complexity(&mut self) -> Result<Complexity> {
        let mut complexity = 0i32;
        let result = unsafe {
            opus_encoder_ctl(
                self.raw.as_ptr(),
                OPUS_GET_COMPLEXITY_REQUEST as i32,
                &mut complexity,
            )
        };

        if result != 0 {
            return Err(Error::from_code(result));
        }

        let complexity = u32::try_from(complexity).map_err(|_| Error::InternalError)?;
        Complexity::try_new(complexity).ok_or(Error::InternalError)
    }

    /// Enable or disable VBR.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn set_vbr(&mut self, enabled: bool) -> Result<()> {
        let vbr = i32::from(enabled);
        let result =
            unsafe { opus_encoder_ctl(self.raw.as_ptr(), OPUS_SET_VBR_REQUEST as i32, vbr) };

        if result != 0 {
            return Err(Error::from_code(result));
        }

        Ok(())
    }

    /// Query VBR status.
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn vbr(&mut self) -> Result<bool> {
        let mut vbr = 0i32;
        let result =
            unsafe { opus_encoder_ctl(self.raw.as_ptr(), OPUS_GET_VBR_REQUEST as i32, &mut vbr) };

        if result != 0 {
            return Err(Error::from_code(result));
        }

        Ok(vbr != 0)
    }

    /// The encoder's configured sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// The encoder's channel configuration.
    #[must_use]
    pub const fn channels(&self) -> Channels {
        self.channels
    }

    /// Reset the encoder to its initial state (same config, cleared history).
    ///
    /// # Errors
    /// Returns [`Error::InvalidState`] if the encoder is invalid, or a mapped libopus error.
    pub fn reset(&mut self) -> Result<()> {
        let r = unsafe {
            opus_encoder_ctl(self.raw.as_ptr(), crate::bindings::OPUS_RESET_STATE as i32)
        };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        #[cfg(feature = "dred")]
        self.reload_active_dnn_blob()?;
        Ok(())
    }
}

impl<'a> EncoderRef<'a> {
    /// Wrap an externally-initialized encoder without taking ownership.
    ///
    /// # Safety
    /// - `ptr` must point to valid, initialized memory of at least [`Encoder::size()`] bytes
    /// - `ptr` must be aligned to at least `align_of::<usize>()` (malloc-style alignment)
    /// - `sample_rate` and `channels` must exactly match the encoder state already stored at `ptr`
    /// - The memory must remain valid for the lifetime `'a`
    /// - Caller is responsible for freeing the memory after this wrapper is dropped
    /// - If the external state already uses runtime-loaded DNN weights, register that blob again
    ///   through `EncoderRef::set_dnn_blob` before calling `EncoderRef::reset`
    ///
    /// Passing mismatched metadata is undefined behavior: later safe methods may validate buffer
    /// sizes against the wrong channel/rate and then call libopus with out-of-bounds buffers.
    ///
    /// # Panics
    /// Panics if `ptr` is null or not pointer-aligned.
    ///
    /// Use [`Encoder::init_in_place`] to initialize the memory before calling this.
    #[must_use]
    pub unsafe fn from_raw(
        ptr: *mut OpusEncoder,
        sample_rate: SampleRate,
        channels: Channels,
    ) -> Self {
        let encoder = Encoder::from_raw(
            crate::checked_non_null(ptr, "EncoderRef::from_raw"),
            sample_rate,
            channels,
            Ownership::Borrowed,
        );
        Self {
            inner: encoder,
            #[cfg(feature = "dred")]
            active_dnn_blob: None,
            _marker: PhantomData,
        }
    }

    /// Initialize and wrap an externally allocated buffer.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] if the buffer is too small, or a mapped libopus error.
    pub fn init_in(
        buf: &'a mut AlignedBuffer,
        sample_rate: SampleRate,
        channels: Channels,
        application: Application,
    ) -> Result<Self> {
        let required = Encoder::size(channels)?;
        if buf.capacity_bytes() < required {
            return Err(Error::BadArg);
        }
        let ptr = buf.as_mut_ptr::<OpusEncoder>();
        unsafe { Encoder::init_in_place(ptr, sample_rate, channels, application)? };
        Ok(unsafe { Self::from_raw(ptr, sample_rate, channels) })
    }

    delegate_ref_mut_methods! {
        fn encode(input: &[i16], output: &mut [u8]) -> Result<usize>;
        fn encode_limited(input: &[i16], output: &mut [u8], max_data_bytes: usize) -> Result<usize>;
        fn encode_float(input: &[f32], output: &mut [u8]) -> Result<usize>;
        fn set_inband_fec(enabled: bool) -> Result<()>;
        fn inband_fec() -> Result<bool>;
        fn set_packet_loss_perc(perc: i32) -> Result<()>;
        fn packet_loss_perc() -> Result<i32>;
        fn set_dtx(enabled: bool) -> Result<()>;
        fn dtx() -> Result<bool>;
        fn in_dtx() -> Result<bool>;
        fn set_vbr_constraint(constrained: bool) -> Result<()>;
        fn vbr_constraint() -> Result<bool>;
        fn set_max_bandwidth(bw: Bandwidth) -> Result<()>;
        fn max_bandwidth() -> Result<Bandwidth>;
        fn set_bandwidth(bw: Bandwidth) -> Result<()>;
        fn bandwidth() -> Result<Bandwidth>;
        fn set_force_channels(channels: Option<Channels>) -> Result<()>;
        fn force_channels() -> Result<Option<Channels>>;
        fn set_signal(signal: Signal) -> Result<()>;
        fn signal() -> Result<Signal>;
        fn lookahead() -> Result<i32>;
        fn final_range() -> Result<u32>;
        fn set_lsb_depth(bits: i32) -> Result<()>;
        fn lsb_depth() -> Result<i32>;
        fn set_expert_frame_duration(dur: ExpertFrameDuration) -> Result<()>;
        fn expert_frame_duration() -> Result<ExpertFrameDuration>;
        fn set_prediction_disabled(disabled: bool) -> Result<()>;
        fn prediction_disabled() -> Result<bool>;
        fn set_phase_inversion_disabled(disabled: bool) -> Result<()>;
        fn phase_inversion_disabled() -> Result<bool>;
        #[cfg(feature = "dred")]
        fn set_dred_duration(frames_10ms: i32) -> Result<()>;
        #[cfg(feature = "dred")]
        fn dred_duration() -> Result<i32>;
        fn set_bitrate(bitrate: Bitrate) -> Result<()>;
        fn bitrate() -> Result<Bitrate>;
        fn set_complexity(complexity: Complexity) -> Result<()>;
        fn complexity() -> Result<Complexity>;
        fn set_vbr(enabled: bool) -> Result<()>;
        fn vbr() -> Result<bool>;
    }

    /// Reset the encoder and restore the last successfully registered external DNN model.
    ///
    /// # Errors
    /// Returns a mapped libopus error if the reset or model restoration fails.
    pub fn reset(&mut self) -> Result<()> {
        self.inner.reset()?;
        #[cfg(feature = "dred")]
        if let Some((ptr, len)) = self.active_dnn_blob {
            unsafe { self.inner.apply_dnn_blob(ptr, len)? };
        }
        Ok(())
    }

    #[cfg(feature = "dred")]
    /// Load an external DNN blob into this borrowed encoder state.
    ///
    /// Unlike [`Encoder::set_dnn_blob`], a borrowed wrapper cannot retain storage beyond the
    /// wrapper's lifetime. This method therefore passes the caller's allocation to libopus.
    ///
    /// # Safety
    /// - `ptr` must point to `len` readable bytes containing a complete, correctly formatted
    ///   libopus DNN blob, and must be aligned to at least `align_of::<u32>()`.
    /// - The allocation must remain fixed and readable until the external encoder state is
    ///   destroyed or will never be used again, even if this method returns an error. Dropping
    ///   this Rust wrapper alone does not end that requirement.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] for invalid pointer metadata or alignment, or a mapped libopus
    /// error when loading fails. Embedded-weight libopus builds return [`Error::Unimplemented`].
    pub unsafe fn set_dnn_blob(&mut self, ptr: *const u8, len: i32) -> Result<()> {
        if ptr.is_null() || len <= 0 || !ptr.addr().is_multiple_of(std::mem::align_of::<u32>()) {
            return Err(Error::BadArg);
        }
        unsafe { self.inner.apply_dnn_blob(ptr, len)? };
        self.active_dnn_blob = Some((ptr, len));
        Ok(())
    }
}

impl Deref for EncoderRef<'_> {
    type Target = Encoder;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(all(test, feature = "dred"))]
mod tests {
    use super::*;

    #[test]
    fn dnn_blob_is_copied_into_retained_aligned_storage() {
        let mut encoder =
            Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
        let source = [0u8, 1, 2, 3, 4];
        let unaligned = unsafe { source.as_ptr().add(1) };

        let index = unsafe { encoder.retain_dnn_blob_copy(unaligned, 4) }.unwrap();
        let (retained, len) = encoder.dnn_blobs[index].parts();

        assert_eq!(len, 4);
        assert_eq!((retained as usize) % std::mem::align_of::<u32>(), 0);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(retained, 4) },
            &source[1..]
        );
        assert_eq!(encoder.dnn_blobs.len(), 1);
    }
}
