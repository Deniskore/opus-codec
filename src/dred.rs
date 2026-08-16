//! Safe wrappers for libopus Deep Redundancy (DRED) decoder support.
//! This module is available when the `dred` Cargo feature is enabled.

use crate::bindings::{
    OPUS_SET_DNN_BLOB_REQUEST, OpusDRED, OpusDREDDecoder, opus_decoder_dred_decode,
    opus_decoder_dred_decode_float, opus_dred_alloc, opus_dred_decoder_create,
    opus_dred_decoder_ctl, opus_dred_decoder_destroy, opus_dred_decoder_get_size,
    opus_dred_decoder_init, opus_dred_free, opus_dred_get_size, opus_dred_parse, opus_dred_process,
};
use crate::constants::{is_frame_size_2_5ms_aligned, max_frame_samples_for};
use crate::decoder::Decoder;
use crate::error::{Error, Result};
use crate::types::SampleRate;
use crate::{AlignedBuffer, Ownership, RawHandle};
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;

// libopus computes `100 * max_dred_samples / sampling_rate` in signed 32-bit math.
const MAX_SAFE_DRED_SAMPLES: usize = (i32::MAX as usize) / 100;

/// Managed handle for libopus `OpusDREDDecoder`.
pub struct DredDecoder {
    raw: RawHandle<OpusDREDDecoder>,
    // External-weight builds retain pointers into DNN blobs. Keep each copy,
    // including copies used by failed non-transactional load attempts, until
    // after the C decoder has been destroyed. Field declaration order makes
    // `raw` drop before this storage.
    dnn_blobs: Vec<Box<[u32]>>,
}

unsafe impl Send for DredDecoder {}

/// Borrowed wrapper around an externally allocated DRED decoder.
///
/// The owning handle cannot be moved out of this borrowed wrapper:
///
/// ```compile_fail
/// use opus_codec::dred::{DredDecoder, DredDecoderRef};
/// fn extract<'a>(state: &mut DredDecoderRef<'a>, replacement: DredDecoder) -> DredDecoder {
///     std::mem::replace(&mut **state, replacement)
/// }
/// ```
pub struct DredDecoderRef<'a> {
    inner: DredDecoder,
    _marker: PhantomData<&'a mut OpusDREDDecoder>,
}

unsafe impl Send for DredDecoderRef<'_> {}

impl DredDecoder {
    fn from_raw(ptr: NonNull<OpusDREDDecoder>, ownership: Ownership) -> Self {
        Self {
            raw: RawHandle::new(ptr, ownership, opus_dred_decoder_destroy),
            dnn_blobs: Vec::new(),
        }
    }

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
        let ptr = NonNull::new(ptr).ok_or(Error::AllocFail)?;
        Ok(Self::from_raw(ptr, Ownership::Owned))
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
    pub unsafe fn init_in_place(ptr: *mut OpusDREDDecoder) -> Result<()> {
        if ptr.is_null() {
            return Err(Error::BadArg);
        }
        if !crate::opus_ptr_is_aligned(ptr.cast()) {
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
        self.raw.as_ptr()
    }

    /// Size of a decoder object in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InternalError`] if libopus reports a non-positive size,
    /// indicating an unexpected ABI/runtime mismatch.
    pub fn size() -> Result<usize> {
        let raw = unsafe { opus_dred_decoder_get_size() };
        if raw <= 0 {
            return Err(Error::InternalError);
        }
        usize::try_from(raw).map_err(|_| Error::InternalError)
    }

    /// Load an external DNN model blob into this DRED decoder.
    ///
    /// # Safety
    ///
    /// `data` must contain a complete, correctly formatted libopus DNN weights blob. Some
    /// external-weight libopus builds do not safely handle malformed model records. The bytes are
    /// copied into aligned storage owned by the decoder, so the caller's allocation need not
    /// remain alive after this returns.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadArg`] for an empty or overlong blob,
    /// [`Error::Unimplemented`] when the linked libopus was not built for external model
    /// weights, or another mapped libopus error when loading fails.
    pub unsafe fn set_dnn_blob(&mut self, data: &[u8]) -> Result<()> {
        let (owned_ptr, len) = self.retain_dnn_blob_copy(data)?;
        let r = unsafe {
            opus_dred_decoder_ctl(
                self.raw.as_ptr(),
                OPUS_SET_DNN_BLOB_REQUEST as i32,
                owned_ptr,
                len,
            )
        };
        if r != 0 {
            let error = Error::from_code(r);
            if error == Error::Unimplemented {
                // An unsupported CTL never inspected or retained the pointer.
                // Other failures may leave model fields pointing into the blob.
                let removed = self.dnn_blobs.pop();
                debug_assert!(removed.is_some());
            }
            return Err(error);
        }
        Ok(())
    }

    fn retain_dnn_blob_copy(&mut self, data: &[u8]) -> Result<(*const u8, i32)> {
        if data.is_empty() {
            return Err(Error::BadArg);
        }
        let len = i32::try_from(data.len()).map_err(|_| Error::BadArg)?;
        let word_len = data.len().div_ceil(std::mem::size_of::<u32>());
        let mut blob = vec![0u32; word_len].into_boxed_slice();
        let blob_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                blob.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(&*blob),
            )
        };
        blob_bytes[..data.len()].copy_from_slice(data);
        let owned_ptr = blob.as_ptr().cast::<u8>();
        self.dnn_blobs.push(blob);
        Ok((owned_ptr, len))
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
        let len = i32::try_from(data.len()).map_err(|_| Error::BadArg)?;
        let max_samples = checked_max_dred_samples(max_dred_samples)?;
        let result = unsafe {
            opus_dred_parse(
                self.raw.as_ptr(),
                state.raw.as_ptr(),
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
        let r = unsafe { opus_dred_process(self.raw.as_ptr(), src.raw.as_ptr(), dst.raw.as_ptr()) };
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
        let channel_count = decoder.channels().as_usize();
        let frame_size = validate_pcm_frame_len(pcm, channel_count, decoder.sample_rate())?;
        let result = unsafe {
            opus_decoder_dred_decode(
                decoder.as_mut_ptr(),
                state.raw.as_ptr(),
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
        let channel_count = decoder.channels().as_usize();
        let frame_size = validate_pcm_frame_len(pcm, channel_count, decoder.sample_rate())?;
        let result = unsafe {
            opus_decoder_dred_decode_float(
                decoder.as_mut_ptr(),
                state.raw.as_ptr(),
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

fn checked_max_dred_samples(max_dred_samples: usize) -> Result<i32> {
    if max_dred_samples > MAX_SAFE_DRED_SAMPLES {
        return Err(Error::BadArg);
    }
    i32::try_from(max_dred_samples).map_err(|_| Error::BadArg)
}

impl<'a> DredDecoderRef<'a> {
    /// Wrap an externally-initialized DRED decoder without taking ownership.
    ///
    /// # Safety
    /// - `ptr` must point to valid, initialized memory of at least [`DredDecoder::size()`] bytes
    /// - The memory must remain valid for the lifetime `'a`
    /// - Caller is responsible for freeing the memory after this wrapper is dropped
    ///
    /// Use [`DredDecoder::init_in_place`] to initialize the memory before calling this.
    ///
    /// # Panics
    /// Panics if `ptr` is null or not pointer-aligned.
    #[must_use]
    pub unsafe fn from_raw(ptr: *mut OpusDREDDecoder) -> Self {
        let decoder = DredDecoder::from_raw(
            crate::checked_non_null(ptr, "DredDecoderRef::from_raw"),
            Ownership::Borrowed,
        );
        Self {
            inner: decoder,
            _marker: PhantomData,
        }
    }

    /// Initialize and wrap an externally allocated buffer.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] if the buffer is too small, or a mapped libopus error.
    pub fn init_in(buf: &'a mut AlignedBuffer) -> Result<Self> {
        let required = DredDecoder::size()?;
        if buf.capacity_bytes() < required {
            return Err(Error::BadArg);
        }
        let ptr = buf.as_mut_ptr::<OpusDREDDecoder>();
        unsafe { DredDecoder::init_in_place(ptr)? };
        Ok(unsafe { Self::from_raw(ptr) })
    }

    /// Borrow the raw external decoder pointer.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut OpusDREDDecoder {
        self.inner.as_mut_ptr()
    }

    delegate_ref_mut_methods! {
        fn parse(state: &mut DredState, data: &[u8], max_dred_samples: usize, sampling_rate: SampleRate, dred_end: &mut i32, defer_processing: bool) -> Result<usize>;
        fn process(src: &DredState, dst: &mut DredState) -> Result<()>;
        fn decode_into_i16(decoder: &mut Decoder, state: &DredState, dred_offset: i32, pcm: &mut [i16]) -> Result<usize>;
        fn decode_into_f32(decoder: &mut Decoder, state: &DredState, dred_offset: i32, pcm: &mut [f32]) -> Result<usize>;
    }

    /// Load an external DNN model blob into the borrowed DRED decoder state.
    ///
    /// # Safety
    /// - `data` must contain a complete, correctly formatted libopus DNN weights blob and its
    ///   backing allocation must be aligned to at least `align_of::<u32>()`.
    /// - The allocation must remain fixed and readable until the external decoder state is
    ///   destroyed or will never be used again, even if this method returns an error. Dropping
    ///   this Rust wrapper alone does not end that requirement.
    ///
    /// # Errors
    /// Returns [`Error::BadArg`] for an empty, misaligned, or overlong blob, or a mapped libopus
    /// error when loading fails.
    pub unsafe fn set_dnn_blob(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() || !(data.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u32>())
        {
            return Err(Error::BadArg);
        }
        let len = i32::try_from(data.len()).map_err(|_| Error::BadArg)?;
        let r = unsafe {
            opus_dred_decoder_ctl(
                self.inner.raw.as_ptr(),
                OPUS_SET_DNN_BLOB_REQUEST as i32,
                data.as_ptr(),
                len,
            )
        };
        if r != 0 {
            return Err(Error::from_code(r));
        }
        Ok(())
    }
}

impl Deref for DredDecoderRef<'_> {
    type Target = DredDecoder;

    fn deref(&self) -> &Self::Target {
        &self.inner
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
    if !pcm.len().is_multiple_of(channel_count) {
        return Err(Error::BadArg);
    }
    let frame_size_per_ch = pcm.len() / channel_count;
    if frame_size_per_ch == 0 || frame_size_per_ch > max_frame_samples_for(sample_rate) {
        return Err(Error::BadArg);
    }
    // libopus requires DRED decode frame sizes to be multiples of 2.5 ms.
    if !is_frame_size_2_5ms_aligned(frame_size_per_ch, sample_rate) {
        return Err(Error::BadArg);
    }
    i32::try_from(frame_size_per_ch).map_err(|_| Error::BadArg)
}

/// Managed handle for libopus `OpusDRED` state.
pub struct DredState {
    raw: NonNull<OpusDRED>,
}

unsafe impl Send for DredState {}

impl DredState {
    /// Allocate a new DRED state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AllocFail`] if allocation fails or a mapped libopus error when
    /// creation does not succeed.
    pub fn new() -> Result<Self> {
        let size = Self::size()?;
        let mut err = 0;
        let ptr = unsafe { opus_dred_alloc(std::ptr::addr_of_mut!(err)) };
        if err != 0 {
            return Err(Error::from_code(err));
        }
        let ptr = NonNull::new(ptr).ok_or(Error::AllocFail)?;
        // opus_dred_alloc() is malloc-like and does not initialize OpusDRED.
        unsafe { std::ptr::write_bytes(ptr.as_ptr().cast::<u8>(), 0, size) };
        Ok(Self { raw: ptr })
    }

    /// Size of a DRED state in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unimplemented`] if DRED is disabled in the linked
    /// libopus, or [`Error::InternalError`] if libopus reports an invalid size.
    pub fn size() -> Result<usize> {
        let raw = unsafe { opus_dred_get_size() };
        if raw == 0 {
            return Err(Error::Unimplemented);
        }
        if raw < 0 {
            return Err(Error::InternalError);
        }
        usize::try_from(raw).map_err(|_| Error::InternalError)
    }

    /// Borrow the raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut OpusDRED {
        self.raw.as_ptr()
    }
}

impl Drop for DredState {
    fn drop(&mut self) {
        unsafe { opus_dred_free(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_pcm_frame_len_checks_arguments() {
        // 2.5 ms at 48 kHz = 120 samples/ch, so 240 total for stereo.
        let pcm = vec![0i16; 240];
        assert!(validate_pcm_frame_len(&pcm, 2, SampleRate::Hz48000).is_ok());

        let err = validate_pcm_frame_len(&pcm, 0, SampleRate::Hz48000).unwrap_err();
        assert_eq!(err, Error::InvalidState);

        let err = validate_pcm_frame_len(&pcm[..3], 2, SampleRate::Hz48000).unwrap_err();
        assert_eq!(err, Error::BadArg);

        let err = validate_pcm_frame_len(&[] as &[i16], 2, SampleRate::Hz48000).unwrap_err();
        assert_eq!(err, Error::BadArg);

        // Non-2.5ms-aligned frame size must be rejected.
        let bad_pcm = vec![0i16; 4];
        let err = validate_pcm_frame_len(&bad_pcm, 2, SampleRate::Hz48000).unwrap_err();
        assert_eq!(err, Error::BadArg);
    }

    #[test]
    fn checked_max_dred_samples_blocks_overflow_inputs() {
        assert_eq!(
            checked_max_dred_samples(MAX_SAFE_DRED_SAMPLES),
            Ok(i32::MAX / 100)
        );
        assert_eq!(
            checked_max_dred_samples(MAX_SAFE_DRED_SAMPLES + 1),
            Err(Error::BadArg)
        );
    }

    #[test]
    fn fresh_dred_state_is_inactive() {
        let mut decoder = match DredDecoder::new() {
            Ok(decoder) => decoder,
            Err(Error::Unimplemented) => return,
            Err(err) => panic!("unexpected DRED decoder error: {err:?}"),
        };
        let state = match DredState::new() {
            Ok(state) => state,
            Err(Error::Unimplemented) => return,
            Err(err) => panic!("unexpected DRED state error: {err:?}"),
        };
        let mut dst = DredState::new().unwrap();

        assert_eq!(decoder.process(&state, &mut dst), Err(Error::BadArg));
    }

    #[cfg(not(opus_codec_system_lib))]
    #[test]
    fn dnn_blob_is_copied_into_retained_aligned_storage() {
        let mut decoder = DredDecoder::new().expect("create bundled DRED decoder");
        let source = [0u8, 1, 2, 3, 4];

        let (retained, len) = decoder.retain_dnn_blob_copy(&source[1..]).unwrap();

        assert_eq!(len, 4);
        assert_eq!((retained as usize) % std::mem::align_of::<u32>(), 0);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(retained, 4) },
            &source[1..]
        );
        assert_eq!(decoder.dnn_blobs.len(), 1);
    }

    #[cfg(all(not(opus_codec_system_lib), not(feature = "external-weights")))]
    #[test]
    fn typed_dnn_blob_ctl_has_checked_input_and_exact_abi() {
        let model_word = 0u32;
        let mut decoder = DredDecoder::new().expect("create bundled DRED decoder");
        assert_eq!(unsafe { decoder.set_dnn_blob(&[]) }, Err(Error::BadArg));

        let aligned_blob = unsafe {
            std::slice::from_raw_parts(
                std::ptr::addr_of!(model_word).cast::<u8>(),
                std::mem::size_of_val(&model_word),
            )
        };
        assert_eq!(
            unsafe { decoder.set_dnn_blob(aligned_blob) },
            Err(Error::Unimplemented)
        );
        assert!(decoder.dnn_blobs.is_empty());
    }
}
