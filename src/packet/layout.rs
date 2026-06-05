#![cfg_attr(not(opus_codec_rust_packet_ops), allow(dead_code))]

use crate::bindings::opus_packet_get_samples_per_frame;
use crate::error::{Error, Result};

mod extensions;

use extensions::{generate_extensions, parse_existing_extensions};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FrameSpan {
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl FrameSpan {
    pub(crate) fn slice(self, packet: &[u8]) -> Result<&[u8]> {
        let end = self
            .start
            .checked_add(self.len)
            .ok_or(Error::InternalError)?;
        packet.get(self.start..end).ok_or(Error::InvalidPacket)
    }
}

#[derive(Debug)]
pub(crate) struct PacketRepacketizerLayout {
    pub(crate) toc: u8,
    frames: [FrameSpan; MAX_FRAMES_PER_PACKET],
    frame_count: usize,
    pub(crate) padding_start: usize,
    pub(crate) packet_offset: usize,
    extension_spans: Option<Result<Vec<PacketExtensionSpan>>>,
}

#[derive(Debug)]
struct PacketExtension<'a> {
    id: u8,
    frame: u8,
    data: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
struct PacketExtensionSpan {
    id: u8,
    frame: u8,
    data_start: usize,
    data_len: usize,
}

impl PacketExtensionSpan {
    fn from_extension(extension: &PacketExtension<'_>, padding: &[u8]) -> Result<Self> {
        let data_address = extension.data.as_ptr() as usize;
        let padding_address = padding.as_ptr() as usize;
        let data_start = data_address
            .checked_sub(padding_address)
            .ok_or(Error::InternalError)?;
        let data_end = data_start
            .checked_add(extension.data.len())
            .ok_or(Error::InternalError)?;
        if data_end > padding.len() {
            return Err(Error::InternalError);
        }
        Ok(Self {
            id: extension.id,
            frame: extension.frame,
            data_start,
            data_len: extension.data.len(),
        })
    }

    fn extension(self, padding: &[u8]) -> Result<PacketExtension<'_>> {
        let data_end = self
            .data_start
            .checked_add(self.data_len)
            .ok_or(Error::InternalError)?;
        let data = padding
            .get(self.data_start..data_end)
            .ok_or(Error::InternalError)?;
        Ok(PacketExtension {
            id: self.id,
            frame: self.frame,
            data,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PacketPadding<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) frame_count: usize,
    cached_extensions: Option<&'a Result<Vec<PacketExtensionSpan>>>,
}

impl PacketRepacketizerLayout {
    pub(crate) fn frames(&self) -> &[FrameSpan] {
        &self.frames[..self.frame_count]
    }

    pub(crate) fn packet_padding<'a>(&'a self, packet: &'a [u8]) -> PacketPadding<'a> {
        PacketPadding {
            data: &packet[self.padding_start..self.packet_offset],
            frame_count: self.frame_count,
            cached_extensions: self.extension_spans.as_ref(),
        }
    }
}

impl PacketPadding<'static> {
    pub(crate) const EMPTY: Self = Self {
        data: &[],
        frame_count: 0,
        cached_extensions: None,
    };
}

// The packet writer/parser below mirrors the libopus packet layout helpers:
// opus.c::opus_packet_parse_impl(), repacketizer.c::opus_repacketizer_out_range_impl(),
// and extensions.c::opus_packet_extensions_*().
const OPUS_REFERENCE_SAMPLE_RATE: i32 = 48_000;
pub(crate) const MAX_FRAMES_PER_PACKET: usize = 48;
const MAX_FRAME_PAYLOAD_BYTES: usize = 1275;
const MAX_PACKET_SAMPLES_48KHZ: usize = 5760;

const TOC_CODE_MASK: u8 = 0x03;
const TOC_CONFIG_MASK: u8 = 0xFC;
const PACKET_CODE_1: u8 = 0x01;
const PACKET_CODE_2: u8 = 0x02;
const PACKET_CODE_3: u8 = 0x03;

const CODE_3_FRAME_COUNT_MASK: u8 = 0x3F;
const CODE_3_PADDING_FLAG: u8 = 0x40;
const CODE_3_VBR_FLAG: u8 = 0x80;

const SIZE_TWO_BYTE_THRESHOLD: usize = 252;
const SIZE_LOW_BITS_MASK: usize = 0x03;

const PADDING_CONTINUATION_BYTE: u8 = 255;
const PADDING_CONTINUATION_AMOUNT: usize = 254;

const EXTENSION_PADDING_ID: u8 = 0;
const EXTENSION_FRAME_SEPARATOR_ID: u8 = 1;
#[cfg(not(opus_codec_frame_bounded_extensions))]
const EXTENSION_MIN_DATA_ID: u8 = 2;
#[cfg(opus_codec_frame_bounded_extensions)]
const EXTENSION_REPEAT_ID: u8 = 2;
#[cfg(opus_codec_frame_bounded_extensions)]
const EXTENSION_FRAME_BOUNDED_MIN_DATA_ID: u8 = 3;
const EXTENSION_SHORT_ID_MAX: u8 = 32;
const EXTENSION_MAX_ID: u8 = 127;
const EXTENSION_PADDING_BYTE: u8 = 0x01;
const EXTENSION_ONE_FRAME_SEPARATOR: u8 = 0x02;
const EXTENSION_MULTI_FRAME_SEPARATOR: u8 = 0x03;
const EXTENSION_LENGTH_CHUNK: usize = 255;
const EXTENSION_LENGTH_CHUNK_BYTE: u8 = 255;

#[derive(Debug, Default)]
struct PacketWritePlan {
    total_size: usize,
    cursor: usize,
    extension_padding_begin: usize,
    extension_padding_end: usize,
    extensions_begin: usize,
    extension_bytes: Vec<u8>,
}

#[derive(Debug)]
struct PacketLayoutParser<'a> {
    data: &'a [u8],
    self_delimited: bool,
    sizes: [usize; MAX_FRAMES_PER_PACKET],
    cursor: usize,
    remaining: usize,
    count: usize,
    cbr: bool,
    last_size: usize,
    padding_len: usize,
}

fn encoded_size_len(size: usize) -> usize {
    if size < SIZE_TWO_BYTE_THRESHOLD { 1 } else { 2 }
}

fn encode_size(size: usize, out: &mut [u8]) -> Result<usize> {
    if out.is_empty() {
        return Err(Error::BufferTooSmall);
    }
    if size < SIZE_TWO_BYTE_THRESHOLD {
        out[0] = u8::try_from(size).map_err(|_| Error::InternalError)?;
        return Ok(1);
    }
    if out.len() < 2 {
        return Err(Error::BufferTooSmall);
    }
    let low = SIZE_TWO_BYTE_THRESHOLD + (size & SIZE_LOW_BITS_MASK);
    out[0] = u8::try_from(low).map_err(|_| Error::InternalError)?;
    out[1] = u8::try_from((size - low) >> 2).map_err(|_| Error::InternalError)?;
    Ok(2)
}

fn parse_size(data: &[u8]) -> Option<(usize, usize)> {
    if data.is_empty() {
        None
    } else if usize::from(data[0]) < SIZE_TWO_BYTE_THRESHOLD {
        Some((1, usize::from(data[0])))
    } else if data.len() < 2 {
        None
    } else {
        Some((2, 4 * usize::from(data[1]) + usize::from(data[0])))
    }
}

impl<'a> PacketLayoutParser<'a> {
    fn new(data: &'a [u8], self_delimited: bool) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidPacket);
        }
        Ok(Self {
            data,
            self_delimited,
            sizes: [0usize; MAX_FRAMES_PER_PACKET],
            cursor: 1,
            remaining: data.len() - 1,
            count: 0,
            cbr: false,
            last_size: data.len() - 1,
            padding_len: 0,
        })
    }

    fn parse(mut self) -> Result<PacketRepacketizerLayout> {
        let toc = self.data[0];
        let frame_samples = usize::try_from(unsafe {
            opus_packet_get_samples_per_frame(self.data.as_ptr(), OPUS_REFERENCE_SAMPLE_RATE)
        })
        .map_err(|_| Error::InvalidPacket)?;
        self.parse_header(toc, frame_samples)?;
        self.finish_last_frame_size()?;
        let (frames, padding_start) = self.collect_frames()?;
        let packet_offset = padding_start
            .checked_add(self.padding_len)
            .ok_or(Error::InternalError)?;
        if packet_offset > self.data.len() {
            return Err(Error::InvalidPacket);
        }

        Ok(PacketRepacketizerLayout {
            toc,
            frames,
            frame_count: self.count,
            padding_start,
            packet_offset,
            extension_spans: None,
        })
    }

    fn parse_header(&mut self, toc: u8, frame_samples: usize) -> Result<()> {
        match toc & TOC_CODE_MASK {
            0 => self.parse_single_frame(),
            PACKET_CODE_1 => self.parse_two_frame_cbr()?,
            PACKET_CODE_2 => self.parse_two_frame_vbr()?,
            _ => self.parse_many_frame_packet(frame_samples)?,
        }
        Ok(())
    }

    fn parse_single_frame(&mut self) {
        self.count = 1;
    }

    fn parse_two_frame_cbr(&mut self) -> Result<()> {
        self.count = 2;
        self.cbr = true;
        if !self.self_delimited {
            if !self.remaining.is_multiple_of(2) {
                return Err(Error::InvalidPacket);
            }
            self.last_size = self.remaining / 2;
            self.sizes[0] = self.last_size;
        }
        Ok(())
    }

    fn parse_two_frame_vbr(&mut self) -> Result<()> {
        self.count = 2;
        let (_, size) = self.read_size_field()?;
        self.sizes[0] = size;
        self.last_size = self.remaining - size;
        Ok(())
    }

    fn parse_many_frame_packet(&mut self, frame_samples: usize) -> Result<()> {
        let ch = self.read_byte()?;
        self.count = usize::from(ch & CODE_3_FRAME_COUNT_MASK);
        self.validate_frame_count(frame_samples)?;
        if ch & CODE_3_PADDING_FLAG != 0 {
            self.parse_padding_length()?;
        }

        self.cbr = ch & CODE_3_VBR_FLAG == 0;
        if self.cbr {
            if !self.self_delimited {
                self.parse_many_frame_cbr()?;
            }
        } else {
            self.parse_many_frame_vbr()?;
        }
        Ok(())
    }

    fn validate_frame_count(&self, frame_samples: usize) -> Result<()> {
        let total_duration = frame_samples
            .checked_mul(self.count)
            .ok_or(Error::InvalidPacket)?;
        if self.count == 0
            || self.count > MAX_FRAMES_PER_PACKET
            || total_duration > MAX_PACKET_SAMPLES_48KHZ
        {
            return Err(Error::InvalidPacket);
        }
        Ok(())
    }

    fn parse_padding_length(&mut self) -> Result<()> {
        loop {
            let byte = self.read_byte()?;
            let chunk = if byte == PADDING_CONTINUATION_BYTE {
                PADDING_CONTINUATION_AMOUNT
            } else {
                usize::from(byte)
            };
            if chunk > self.remaining {
                return Err(Error::InvalidPacket);
            }
            self.remaining -= chunk;
            self.padding_len = self
                .padding_len
                .checked_add(chunk)
                .ok_or(Error::InternalError)?;
            if byte != PADDING_CONTINUATION_BYTE {
                return Ok(());
            }
        }
    }

    fn parse_many_frame_vbr(&mut self) -> Result<()> {
        self.last_size = self.remaining;
        for index in 0..(self.count - 1) {
            let (bytes, size) = self.read_size_field()?;
            self.sizes[index] = size;
            let consumed = bytes.checked_add(size).ok_or(Error::InternalError)?;
            if consumed > self.last_size {
                return Err(Error::InvalidPacket);
            }
            self.last_size -= consumed;
        }
        Ok(())
    }

    fn parse_many_frame_cbr(&mut self) -> Result<()> {
        self.last_size = self.remaining / self.count;
        if self
            .last_size
            .checked_mul(self.count)
            .ok_or(Error::InternalError)?
            != self.remaining
        {
            return Err(Error::InvalidPacket);
        }
        for size in self.sizes.iter_mut().take(self.count - 1) {
            *size = self.last_size;
        }
        Ok(())
    }

    fn finish_last_frame_size(&mut self) -> Result<()> {
        if self.self_delimited {
            let (bytes, last) = self.read_size_field()?;
            if self.cbr {
                if last.checked_mul(self.count).ok_or(Error::InternalError)? > self.remaining {
                    return Err(Error::InvalidPacket);
                }
                for size in self.sizes.iter_mut().take(self.count - 1) {
                    *size = last;
                }
            } else if bytes.checked_add(last).ok_or(Error::InternalError)? > self.last_size {
                return Err(Error::InvalidPacket);
            }
            self.sizes[self.count - 1] = last;
            return Ok(());
        }

        if self.last_size > MAX_FRAME_PAYLOAD_BYTES {
            return Err(Error::InvalidPacket);
        }
        self.sizes[self.count - 1] = self.last_size;
        Ok(())
    }

    fn read_byte(&mut self) -> Result<u8> {
        let byte = *self.data.get(self.cursor).ok_or(Error::InvalidPacket)?;
        self.consume_header_bytes(1)?;
        Ok(byte)
    }

    fn read_size_field(&mut self) -> Result<(usize, usize)> {
        let (bytes, size) = parse_size(&self.data[self.cursor..]).ok_or(Error::InvalidPacket)?;
        self.consume_header_bytes(bytes)?;
        if size > self.remaining {
            return Err(Error::InvalidPacket);
        }
        Ok((bytes, size))
    }

    fn consume_header_bytes(&mut self, bytes: usize) -> Result<()> {
        if bytes > self.remaining {
            return Err(Error::InvalidPacket);
        }
        self.cursor = self.cursor.checked_add(bytes).ok_or(Error::InternalError)?;
        self.remaining -= bytes;
        Ok(())
    }

    fn collect_frames(&self) -> Result<([FrameSpan; MAX_FRAMES_PER_PACKET], usize)> {
        let mut frame_cursor = self.cursor;
        let mut frames = [FrameSpan::default(); MAX_FRAMES_PER_PACKET];
        for (index, &len) in self.sizes.iter().take(self.count).enumerate() {
            let end = frame_cursor.checked_add(len).ok_or(Error::InternalError)?;
            if end > self.data.len() {
                return Err(Error::InvalidPacket);
            }
            frames[index] = FrameSpan {
                start: frame_cursor,
                len,
            };
            frame_cursor = end;
        }
        Ok((frames, frame_cursor))
    }
}

fn parse_packet_layout(data: &[u8], self_delimited: bool) -> Result<PacketRepacketizerLayout> {
    PacketLayoutParser::new(data, self_delimited)?.parse()
}

fn ensure_output_capacity(required: usize, out: &[u8]) -> Result<()> {
    if required > out.len() {
        return Err(Error::BufferTooSmall);
    }
    Ok(())
}

fn write_compact_packet_header(
    toc: u8,
    frame_lengths: &[usize],
    out: &mut [u8],
) -> Result<PacketWritePlan> {
    match frame_lengths {
        [] => Err(Error::InvalidPacket),
        [first] => {
            let total_size = first.checked_add(1).ok_or(Error::InternalError)?;
            ensure_output_capacity(total_size, out)?;
            out[0] = toc & TOC_CONFIG_MASK;
            Ok(PacketWritePlan {
                total_size,
                cursor: 1,
                ..PacketWritePlan::default()
            })
        }
        [first, second] if second == first => {
            let total_size = first
                .checked_mul(2)
                .and_then(|size| size.checked_add(1))
                .ok_or(Error::InternalError)?;
            ensure_output_capacity(total_size, out)?;
            out[0] = (toc & TOC_CONFIG_MASK) | PACKET_CODE_1;
            Ok(PacketWritePlan {
                total_size,
                cursor: 1,
                ..PacketWritePlan::default()
            })
        }
        [first, second] => {
            let total_size = first
                .checked_add(*second)
                .and_then(|size| size.checked_add(1 + encoded_size_len(*first)))
                .ok_or(Error::InternalError)?;
            ensure_output_capacity(total_size, out)?;
            out[0] = (toc & TOC_CONFIG_MASK) | PACKET_CODE_2;
            let mut plan = PacketWritePlan {
                total_size,
                cursor: 1,
                ..PacketWritePlan::default()
            };
            plan.cursor += encode_size(*first, &mut out[plan.cursor..])?;
            Ok(plan)
        }
        _ => Ok(PacketWritePlan::default()),
    }
}

fn frame_lengths_are_vbr(frame_lengths: &[usize]) -> bool {
    frame_lengths[1..]
        .iter()
        .any(|&len| len != frame_lengths[0])
}

fn write_general_packet_header(
    toc: u8,
    frame_lengths: &[usize],
    extension_bytes: Vec<u8>,
    out: &mut [u8],
) -> Result<PacketWritePlan> {
    let count = frame_lengths.len();
    let vbr = frame_lengths_are_vbr(frame_lengths);
    let mut plan = PacketWritePlan::default();

    if vbr {
        plan.total_size = 2;
        for &len in frame_lengths.iter().take(count - 1) {
            plan.total_size = plan
                .total_size
                .checked_add(encoded_size_len(len))
                .and_then(|size| size.checked_add(len))
                .ok_or(Error::InternalError)?;
        }
        plan.total_size = plan
            .total_size
            .checked_add(frame_lengths[count - 1])
            .ok_or(Error::InternalError)?;
        ensure_output_capacity(plan.total_size, out)?;
        out[0] = (toc & TOC_CONFIG_MASK) | PACKET_CODE_3;
        out[1] = u8::try_from(count).map_err(|_| Error::InternalError)? | CODE_3_VBR_FLAG;
    } else {
        plan.total_size = frame_lengths[0]
            .checked_mul(count)
            .and_then(|size| size.checked_add(2))
            .ok_or(Error::InternalError)?;
        ensure_output_capacity(plan.total_size, out)?;
        out[0] = (toc & TOC_CONFIG_MASK) | PACKET_CODE_3;
        out[1] = u8::try_from(count).map_err(|_| Error::InternalError)?;
    }
    plan.cursor = 2;
    plan.extension_bytes = extension_bytes;

    let pad_amount = out.len() - plan.total_size;
    if pad_amount != 0 {
        let nb_255s = (pad_amount - 1) / usize::from(PADDING_CONTINUATION_BYTE);
        let overhead = plan
            .total_size
            .checked_add(plan.extension_bytes.len())
            .and_then(|size| size.checked_add(nb_255s + 1))
            .ok_or(Error::InternalError)?;
        ensure_output_capacity(overhead, out)?;
        out[1] |= CODE_3_PADDING_FLAG;
        plan.extensions_begin = plan.total_size + pad_amount - plan.extension_bytes.len();
        plan.extension_padding_begin = plan.total_size + nb_255s + 1;
        plan.extension_padding_end = out.len() - plan.extension_bytes.len();
        out[plan.cursor..plan.cursor + nb_255s].fill(PADDING_CONTINUATION_BYTE);
        plan.cursor += nb_255s;
        out[plan.cursor] =
            u8::try_from(pad_amount - usize::from(PADDING_CONTINUATION_BYTE) * nb_255s - 1)
                .map_err(|_| Error::InternalError)?;
        plan.cursor += 1;
        plan.total_size += pad_amount;
    }

    if vbr {
        for &len in frame_lengths.iter().take(count - 1) {
            plan.cursor += encode_size(len, &mut out[plan.cursor..])?;
        }
    }
    Ok(plan)
}

fn general_packet_frame_offset(
    frame_lengths: &[usize],
    extension_len: usize,
    out_len: usize,
) -> Result<usize> {
    let base_len = general_packet_base_len(frame_lengths)?;
    if base_len > out_len {
        return Err(Error::BufferTooSmall);
    }
    let pad_amount = out_len - base_len;
    let mut cursor = 2usize;
    if pad_amount != 0 {
        let padding_header_len = (pad_amount - 1) / usize::from(PADDING_CONTINUATION_BYTE) + 1;
        let overhead = base_len
            .checked_add(extension_len)
            .and_then(|size| size.checked_add(padding_header_len))
            .ok_or(Error::InternalError)?;
        if overhead > out_len {
            return Err(Error::BufferTooSmall);
        }
        cursor = cursor
            .checked_add(padding_header_len)
            .ok_or(Error::InternalError)?;
    } else if extension_len != 0 {
        return Err(Error::BufferTooSmall);
    }
    if frame_lengths_are_vbr(frame_lengths) {
        for &len in frame_lengths.iter().take(frame_lengths.len() - 1) {
            cursor = cursor
                .checked_add(encoded_size_len(len))
                .ok_or(Error::InternalError)?;
        }
    }
    Ok(cursor)
}

fn compact_packet_len(frame_lengths: &[usize]) -> Result<Option<usize>> {
    match frame_lengths {
        [] => Err(Error::InvalidPacket),
        [first] => first.checked_add(1).map(Some).ok_or(Error::InternalError),
        [first, second] if second == first => first
            .checked_mul(2)
            .and_then(|size| size.checked_add(1))
            .map(Some)
            .ok_or(Error::InternalError),
        [first, second] => first
            .checked_add(*second)
            .and_then(|size| size.checked_add(1 + encoded_size_len(*first)))
            .map(Some)
            .ok_or(Error::InternalError),
        _ => Ok(None),
    }
}

fn general_packet_base_len(frame_lengths: &[usize]) -> Result<usize> {
    let count = frame_lengths.len();
    if count == 0 {
        return Err(Error::InvalidPacket);
    }

    if frame_lengths_are_vbr(frame_lengths) {
        let mut total_size = 2usize;
        for &len in frame_lengths.iter().take(count - 1) {
            total_size = total_size
                .checked_add(encoded_size_len(len))
                .and_then(|size| size.checked_add(len))
                .ok_or(Error::InternalError)?;
        }
        total_size
            .checked_add(frame_lengths[count - 1])
            .ok_or(Error::InternalError)
    } else {
        frame_lengths[0]
            .checked_mul(count)
            .and_then(|size| size.checked_add(2))
            .ok_or(Error::InternalError)
    }
}

fn copy_frame_slices(frames: &[&[u8]], cursor: &mut usize, out: &mut [u8]) -> Result<()> {
    for frame in frames {
        let dst_end = cursor
            .checked_add(frame.len())
            .ok_or(Error::InternalError)?;
        if dst_end > out.len() {
            return Err(Error::BufferTooSmall);
        }
        out[*cursor..dst_end].copy_from_slice(frame);
        *cursor = dst_end;
    }
    Ok(())
}

fn finish_packet_write(plan: &PacketWritePlan, cursor: usize, out: &mut [u8]) -> Result<()> {
    if !plan.extension_bytes.is_empty() {
        let ext_end = plan
            .extensions_begin
            .checked_add(plan.extension_bytes.len())
            .ok_or(Error::InternalError)?;
        if ext_end > out.len() {
            return Err(Error::BufferTooSmall);
        }
        out[plan.extensions_begin..ext_end].copy_from_slice(&plan.extension_bytes);
    }
    if plan.extension_padding_begin < plan.extension_padding_end {
        out[plan.extension_padding_begin..plan.extension_padding_end].fill(EXTENSION_PADDING_BYTE);
    }
    if plan.extension_bytes.is_empty() && cursor < out.len() {
        out[cursor..].fill(0);
    }
    Ok(())
}

fn append_repacketizer_extension_in_range<'a>(
    mut extension: PacketExtension<'a>,
    owner_frame: usize,
    begin: usize,
    end: usize,
    extensions: &mut Vec<PacketExtension<'a>>,
) -> Result<()> {
    let frame = usize::from(extension.frame)
        .checked_add(owner_frame)
        .ok_or(Error::InternalError)?;
    if (begin..end).contains(&frame) {
        extension.frame = u8::try_from(frame - begin).map_err(|_| Error::InvalidPacket)?;
        extensions.push(extension);
    }
    Ok(())
}

fn collect_repacketizer_extensions_in_range<'a>(
    paddings: &[PacketPadding<'a>],
    begin: usize,
    end: usize,
) -> Result<Vec<PacketExtension<'a>>> {
    let mut extensions = Vec::new();
    for (owner_frame, padding) in paddings.iter().enumerate().take(end) {
        if let Some(cached_extensions) = padding.cached_extensions {
            match cached_extensions {
                Ok(spans) => {
                    for span in spans {
                        append_repacketizer_extension_in_range(
                            span.extension(padding.data)?,
                            owner_frame,
                            begin,
                            end,
                            &mut extensions,
                        )?;
                    }
                }
                Err(err) if (begin..end).contains(&owner_frame) => return Err(err.clone()),
                Err(_) => {}
            }
        } else {
            let frame_extensions =
                match parse_existing_extensions(padding.data, padding.frame_count) {
                    Ok(extensions) => extensions,
                    Err(err) if (begin..end).contains(&owner_frame) => return Err(err),
                    Err(_) => continue,
                };
            for extension in frame_extensions {
                append_repacketizer_extension_in_range(
                    extension,
                    owner_frame,
                    begin,
                    end,
                    &mut extensions,
                )?;
            }
        }
    }
    Ok(extensions)
}

pub(crate) fn packet_repacketizer_layout(packet: &[u8]) -> Result<PacketRepacketizerLayout> {
    let mut parsed = parse_packet_layout(packet, false)?;
    if parsed.packet_offset != packet.len() {
        return Err(Error::InvalidPacket);
    }
    let padding = &packet[parsed.padding_start..parsed.packet_offset];
    parsed.extension_spans = Some(
        parse_existing_extensions(padding, parsed.frame_count).and_then(|extensions| {
            extensions
                .iter()
                .map(|extension| PacketExtensionSpan::from_extension(extension, padding))
                .collect()
        }),
    );
    Ok(parsed)
}

pub(crate) fn repacketize_frames(
    toc: u8,
    frames: &[&[u8]],
    paddings: &[PacketPadding<'_>],
    out: &mut [u8],
) -> Result<usize> {
    repacketize_frames_range(toc, frames, paddings, 0, frames.len(), out)
}

pub(crate) fn repacketize_frames_range(
    toc: u8,
    frames: &[&[u8]],
    paddings: &[PacketPadding<'_>],
    begin: usize,
    end: usize,
    out: &mut [u8],
) -> Result<usize> {
    if frames.len() != paddings.len() {
        return Err(Error::InternalError);
    }
    if begin > end || end > frames.len() {
        return Err(Error::BadArg);
    }

    let frames = &frames[begin..end];
    if frames.len() > MAX_FRAMES_PER_PACKET {
        return Err(Error::InvalidPacket);
    }
    let mut frame_lengths = [0usize; MAX_FRAMES_PER_PACKET];
    for (length, frame) in frame_lengths.iter_mut().zip(frames) {
        *length = frame.len();
    }
    let frame_lengths = &frame_lengths[..frames.len()];
    let extensions = collect_repacketizer_extensions_in_range(paddings, begin, end)?;

    let compact_len = if extensions.is_empty() {
        compact_packet_len(frame_lengths)?
    } else {
        None
    };
    let (total_len, extension_bytes) = if let Some(total_len) = compact_len {
        (total_len, Vec::new())
    } else {
        let base_len = general_packet_base_len(frame_lengths)?;
        let extension_bytes = generate_extensions(&extensions, usize::MAX, frames.len())?;
        let ext_len = extension_bytes.len();
        let ext_padding_len = if ext_len == 0 {
            0
        } else {
            ext_len
                .checked_add(ext_len / PADDING_CONTINUATION_AMOUNT)
                .and_then(|len| len.checked_add(1))
                .ok_or(Error::InternalError)?
        };
        let total_len = base_len
            .checked_add(ext_padding_len)
            .ok_or(Error::InternalError)?;
        (total_len, extension_bytes)
    };

    ensure_output_capacity(total_len, out)?;
    let out = &mut out[..total_len];
    let mut plan = if compact_len.is_some() {
        write_compact_packet_header(toc, frame_lengths, out)?
    } else {
        write_general_packet_header(toc, frame_lengths, extension_bytes, out)?
    };

    copy_frame_slices(frames, &mut plan.cursor, out)?;
    finish_packet_write(&plan, plan.cursor, out)?;
    debug_assert_eq!(plan.total_size, total_len);
    Ok(total_len)
}

#[cfg(opus_codec_rust_packet_ops)]
pub(super) fn pad_single_packet(packet: &mut [u8], len: usize, new_len: usize) -> Result<()> {
    let parsed = parse_packet_layout(&packet[..len], false)?;
    if parsed.packet_offset != len {
        return Err(Error::InvalidPacket);
    }
    let frame_count = parsed.frames().len();
    let mut frame_lengths = [0usize; MAX_FRAMES_PER_PACKET];
    for (length, frame) in frame_lengths.iter_mut().zip(parsed.frames()) {
        *length = frame.len;
    }
    let frame_lengths = &frame_lengths[..frame_count];
    let base_len = general_packet_base_len(frame_lengths)?;
    if base_len > new_len {
        return Err(Error::BufferTooSmall);
    }
    let extension_bytes = {
        let padding = &packet[parsed.padding_start..parsed.packet_offset];
        let extensions = parse_existing_extensions(padding, frame_count)?;
        generate_extensions(&extensions, new_len - base_len, frame_count)?
    };
    let frame_offset = general_packet_frame_offset(frame_lengths, extension_bytes.len(), new_len)?;

    let frame_bytes = frame_lengths.iter().try_fold(0usize, |total, &length| {
        total.checked_add(length).ok_or(Error::InternalError)
    })?;
    let frames_end = frame_offset
        .checked_add(frame_bytes)
        .ok_or(Error::InternalError)?;

    // Growing a packet must never move a frame before its source position.
    // Validate every move before mutating the packet so a future layout change
    // cannot turn this overlap assumption into partial or corrupt output.
    let mut frame_destinations = [0usize; MAX_FRAMES_PER_PACKET];
    let mut source_ends = [0usize; MAX_FRAMES_PER_PACKET];
    let mut destination_end = frames_end;
    for index in (0..frame_count).rev() {
        let frame = parsed.frames()[index];
        let destination_start = destination_end
            .checked_sub(frame.len)
            .ok_or(Error::InternalError)?;
        let source_end = frame
            .start
            .checked_add(frame.len)
            .ok_or(Error::InternalError)?;
        if source_end > len || destination_start < frame.start {
            return Err(Error::InternalError);
        }
        frame_destinations[index] = destination_start;
        source_ends[index] = source_end;
        destination_end = destination_start;
    }
    if destination_end != frame_offset {
        return Err(Error::InternalError);
    }

    for index in (0..frame_count).rev() {
        let frame = parsed.frames()[index];
        packet.copy_within(frame.start..source_ends[index], frame_destinations[index]);
    }

    let out = &mut packet[..new_len];
    let plan = write_general_packet_header(parsed.toc, frame_lengths, extension_bytes, out)?;
    debug_assert_eq!(plan.cursor, frame_offset);
    finish_packet_write(&plan, frames_end, out)?;
    debug_assert_eq!(plan.total_size, out.len());
    Ok(())
}

#[cfg(opus_codec_rust_packet_ops)]
pub(super) fn multistream_last_stream_offset(packet: &[u8], nb_streams: usize) -> Result<usize> {
    let mut offset = 0usize;
    for _ in 0..nb_streams.saturating_sub(1) {
        let parsed = parse_packet_layout(&packet[offset..], true)?;
        offset = offset
            .checked_add(parsed.packet_offset)
            .ok_or(Error::InternalError)?;
        if offset > packet.len() {
            return Err(Error::InvalidPacket);
        }
    }
    Ok(offset)
}
