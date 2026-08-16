#![cfg(feature = "external-weights")]

use opus_codec::decoder::DecoderRef;
use opus_codec::dred::{DredDecoder, DredState};
use opus_codec::encoder::EncoderRef;
use opus_codec::{
    AlignedBuffer, Application, Bitrate, Channels, Decoder, Encoder, Result, SampleRate,
};

const SAMPLE_RATE: SampleRate = SampleRate::Hz48000;
const FRAME_SAMPLES: usize = 960;

fn aligned_blob_copy(source: &[u8]) -> Box<[u32]> {
    let word_len = source.len().div_ceil(std::mem::size_of::<u32>());
    let mut words = vec![0u32; word_len].into_boxed_slice();
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            words.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(&*words),
        )
    };
    bytes[..source.len()].copy_from_slice(source);
    words
}

fn encode_until_dred(
    mut encode: impl FnMut(&[i16], &mut [u8]) -> Result<usize>,
    dred_decoder: &mut DredDecoder,
    dred_state: &mut DredState,
    noise: &mut u32,
) -> Vec<u8> {
    let mut packet_storage = vec![0u8; 4_000];
    for _ in 0..300 {
        let mut pcm = vec![0i16; FRAME_SAMPLES];
        for sample in &mut pcm {
            *noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = ((*noise >> 17) as i16).wrapping_sub(16_384);
        }

        let packet_len = encode(&pcm, &mut packet_storage).expect("encode packet");
        let packet = &packet_storage[..packet_len];
        let mut dred_end = 0;
        let dred_amount = dred_decoder
            .parse(
                dred_state,
                packet,
                48_000,
                SAMPLE_RATE,
                &mut dred_end,
                false,
            )
            .expect("parse DRED packet");
        if dred_amount >= FRAME_SAMPLES {
            return packet.to_vec();
        }
    }
    panic!("encoder produced no decodable DRED payload");
}

fn decode_packet_and_loss(
    mut decode: impl FnMut(&[u8], &mut [i16], bool) -> Result<usize>,
    packet: &[u8],
) {
    let mut pcm = vec![0i16; FRAME_SAMPLES];
    assert_eq!(
        decode(packet, &mut pcm, false).expect("decode packet"),
        FRAME_SAMPLES
    );
    assert_eq!(
        decode(&[], &mut pcm, false).expect("decode packet loss"),
        FRAME_SAMPLES
    );
}

fn configure_encoder(encoder: &mut Encoder) {
    encoder
        .set_bitrate(Bitrate::Custom(32_000))
        .expect("set bitrate");
    encoder
        .set_packet_loss_perc(20)
        .expect("set expected packet loss");
    encoder.set_dred_duration(100).expect("enable DRED");
}

fn configure_borrowed_encoder(encoder: &mut EncoderRef<'_>) {
    encoder
        .set_bitrate(Bitrate::Custom(32_000))
        .expect("set bitrate");
    encoder
        .set_packet_loss_perc(20)
        .expect("set expected packet loss");
    encoder.set_dred_duration(100).expect("enable DRED");
}

#[test]
#[ignore = "requires libopus built with USE_WEIGHTS_FILE and OPUS_CODEC_DNN_BLOB"]
fn external_weight_dred_round_trip() {
    let blob_path = std::env::var_os("OPUS_CODEC_DNN_BLOB")
        .expect("OPUS_CODEC_DNN_BLOB must name a valid weights blob");
    let source = std::fs::read(blob_path).expect("read weights blob");
    let blob_len = i32::try_from(source.len()).expect("blob length fits i32");
    let borrowed_blob = aligned_blob_copy(&source);
    let borrowed_blob_ptr = borrowed_blob.as_ptr().cast::<u8>();

    let mut dred_decoder = DredDecoder::new().expect("create DRED decoder");
    unsafe {
        dred_decoder
            .set_dnn_blob(&source)
            .expect("load DRED decoder weights");
    }

    let mut noise = 0x1234_5678u32;
    {
        let mut encoder =
            Encoder::new(SAMPLE_RATE, Channels::Mono, Application::Voip).expect("create encoder");
        let mut decoder = Decoder::new(SAMPLE_RATE, Channels::Mono).expect("create decoder");
        unsafe {
            encoder
                .set_dnn_blob(source.as_ptr(), blob_len)
                .expect("load encoder weights");
            decoder
                .set_dnn_blob(source.as_ptr(), blob_len)
                .expect("load decoder weights");
        }

        // All owning handles retain aligned copies.
        drop(source);

        configure_encoder(&mut encoder);
        let mut dred_state = DredState::new().expect("create DRED state");
        let _packet = encode_until_dred(
            |pcm, out| encoder.encode(pcm, out),
            &mut dred_decoder,
            &mut dred_state,
            &mut noise,
        );
        let mut dred_pcm = vec![0i16; FRAME_SAMPLES];
        assert_eq!(
            dred_decoder
                .decode_into_i16(
                    &mut decoder,
                    &dred_state,
                    FRAME_SAMPLES as i32,
                    &mut dred_pcm,
                )
                .expect("decode DRED audio"),
            FRAME_SAMPLES
        );

        encoder.reset().expect("reset encoder and restore weights");
        decoder.reset().expect("reset decoder and restore weights");
        // OPUS_RESET_STATE resets this runtime setting even though it preserves bitrate settings.
        encoder.set_dred_duration(100).expect("re-enable DRED");

        let packet = encode_until_dred(
            |pcm, out| encoder.encode(pcm, out),
            &mut dred_decoder,
            &mut dred_state,
            &mut noise,
        );
        assert!(packet[0] >> 3 < 16, "deep-PLC test requires SILK/hybrid");
        assert_eq!(
            dred_decoder
                .decode_into_i16(
                    &mut decoder,
                    &dred_state,
                    FRAME_SAMPLES as i32,
                    &mut dred_pcm,
                )
                .expect("decode DRED audio after reset"),
            FRAME_SAMPLES
        );
        decode_packet_and_loss(|data, out, fec| decoder.decode(data, out, fec), &packet);
    }

    // Borrowed wrappers keep caller-owned pointer metadata and reapply it after reset.
    let mut encoder_storage =
        AlignedBuffer::with_capacity_bytes(Encoder::size(Channels::Mono).expect("encoder size"));
    let mut decoder_storage =
        AlignedBuffer::with_capacity_bytes(Decoder::size(Channels::Mono).expect("decoder size"));
    let mut encoder = EncoderRef::init_in(
        &mut encoder_storage,
        SAMPLE_RATE,
        Channels::Mono,
        Application::Voip,
    )
    .expect("create borrowed encoder");
    let mut decoder = DecoderRef::init_in(&mut decoder_storage, SAMPLE_RATE, Channels::Mono)
        .expect("create borrowed decoder");
    unsafe {
        encoder
            .set_dnn_blob(borrowed_blob_ptr, blob_len)
            .expect("load borrowed encoder weights");
        decoder
            .set_dnn_blob(borrowed_blob_ptr, blob_len)
            .expect("load borrowed decoder weights");
    }
    encoder
        .reset()
        .expect("reset borrowed encoder and restore weights");
    decoder
        .reset()
        .expect("reset borrowed decoder and restore weights");
    configure_borrowed_encoder(&mut encoder);

    let mut dred_state = DredState::new().expect("create borrowed-path DRED state");
    let packet = encode_until_dred(
        |pcm, out| encoder.encode(pcm, out),
        &mut dred_decoder,
        &mut dred_state,
        &mut noise,
    );
    assert!(packet[0] >> 3 < 16, "deep-PLC test requires SILK/hybrid");
    decode_packet_and_loss(|data, out, fec| decoder.decode(data, out, fec), &packet);
}
