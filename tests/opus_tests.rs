use opus_codec::decoder::Decoder;
use opus_codec::encoder::Encoder;
use opus_codec::error::Error;
use opus_codec::multistream::{MSDecoder, MSEncoder, Mapping};
use opus_codec::packet::{
    packet_bandwidth, packet_channels, packet_nb_frames, packet_nb_samples, packet_parse, soft_clip,
};
use opus_codec::repacketizer::Repacketizer;
use opus_codec::types::{Application, Bandwidth, Channels, SampleRate};

#[test]
fn test_packet_analysis() {
    // Create a silent packet
    let mut encoder =
        Encoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Audio).unwrap();
    let pcm = vec![0i16; 960 * 2]; // 20ms stereo
    let mut output = [0u8; 100];
    let len = encoder.encode(&pcm, &mut output).unwrap();
    let packet = &output[..len];

    // Analyze
    assert!(packet_nb_frames(packet).unwrap() > 0);
    assert_eq!(packet_nb_samples(packet, SampleRate::Hz48000).unwrap(), 960);
    assert_eq!(packet_channels(packet).unwrap(), Channels::Stereo);
    assert!(packet_bandwidth(packet).unwrap() != Bandwidth::Narrowband); // Likely Fullband for Audio app

    // Parse
    let (_toc, _offset, frames) = packet_parse(packet).unwrap();
    assert!(!frames.is_empty());
}

#[test]
fn test_encode_decode() {
    let mut encoder = Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip).unwrap();
    let mut decoder = Decoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();

    let frame_size = 960; // 20ms
    let pcm_in = vec![0i16; frame_size];
    let mut packet = [0u8; 500];
    let mut pcm_out = vec![0i16; frame_size];

    let len = encoder.encode(&pcm_in, &mut packet).unwrap();
    assert!(len > 0);

    let decoded_len = decoder.decode(&packet[..len], &mut pcm_out, false).unwrap();
    assert_eq!(decoded_len, frame_size);
}

#[test]
fn test_float_api() {
    let mut encoder =
        Encoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Audio).unwrap();
    let mut decoder = Decoder::new(SampleRate::Hz48000, Channels::Stereo).unwrap();

    let frame_size = 480; // 10ms
    let pcm_in = vec![0.0f32; frame_size * 2];
    let mut packet = [0u8; 500];
    let mut pcm_out = vec![0.0f32; frame_size * 2];

    let len = encoder.encode_float(&pcm_in, &mut packet).unwrap();
    assert!(len > 0);

    let decoded_len = decoder
        .decode_float(&packet[..len], &mut pcm_out, false)
        .unwrap();
    assert_eq!(decoded_len, frame_size);
}

#[test]
fn test_multistream_surround() {
    // 5.1 Surround: 6 channels
    let channels = 6;
    let mapping_family = 1; // Family 1 is for surround
    let (mut encoder, _) = MSEncoder::new_surround(
        SampleRate::Hz48000,
        channels,
        mapping_family,
        Application::Audio,
    )
    .unwrap();

    let streams = encoder.streams();
    let coupled = encoder.coupled_streams();
    let mapping_table = [0, 1, 2, 3, 4, 5]; // Standard identity mapping for the streams

    let mapping = Mapping {
        channels,
        streams,
        coupled_streams: coupled,
        mapping: &mapping_table,
    };

    let mut decoder = MSDecoder::new(SampleRate::Hz48000, mapping).unwrap();

    let frame_size = 960;
    let pcm_in = vec![0i16; frame_size * channels as usize];
    let mut packet = [0u8; 1500];
    let mut pcm_out = vec![0i16; frame_size * channels as usize];

    let len = encoder.encode(&pcm_in, frame_size, &mut packet).unwrap();
    assert!(len > 0);

    let decoded_len = decoder
        .decode(&packet[..len], &mut pcm_out, frame_size, false)
        .unwrap();
    assert_eq!(decoded_len, frame_size);
}

#[test]
fn test_repacketizer() {
    let mut rp = Repacketizer::new().unwrap();
    let mut encoder = Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip).unwrap();

    // Create two 20ms frames
    let frame_size = 960;
    let pcm = vec![0i16; frame_size];
    let mut packet1 = [0u8; 200];
    let mut packet2 = [0u8; 200];

    let len1 = encoder.encode(&pcm, &mut packet1).unwrap();
    let len2 = encoder.encode(&pcm, &mut packet2).unwrap();

    // Add them to repacketizer
    rp.push(&packet1[..len1]).unwrap();
    rp.push(&packet2[..len2]).unwrap();

    // Verify we have 2 frames
    assert_eq!(rp.frames(), 2);

    // Merge into one packet
    let mut merged = [0u8; 500];
    let merged_len = rp.out(&mut merged).unwrap();
    assert!(merged_len > 0);

    // Verify the merged packet has 2 frames
    assert_eq!(packet_nb_frames(&merged[..merged_len]).unwrap(), 2);
}

#[test]
fn test_buffer_empty() {
    let mut encoder = Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip).unwrap();
    let pcm = vec![0i16; 960];
    let mut empty_buf = [0u8; 0];

    // The wrapper should catch this and return BadArg before calling libopus
    let result = encoder.encode(&pcm, &mut empty_buf);
    assert_eq!(result, Err(Error::BadArg));
}

#[test]
fn test_projection_ambisonics() {
    use opus_codec::projection::{ProjectionDecoder, ProjectionEncoder};

    // First Order Ambisonics (4 channels) with Family 3 (Ambisonics)
    let channels = 4;
    let mapping_family = 3;
    let mut encoder = ProjectionEncoder::new(
        SampleRate::Hz48000,
        channels,
        mapping_family,
        Application::Audio,
    )
    .unwrap();

    let demixing_matrix = encoder.demixing_matrix_bytes().unwrap();
    assert!(!demixing_matrix.is_empty());

    let mut decoder = ProjectionDecoder::new(
        SampleRate::Hz48000,
        channels,
        encoder.streams(),
        encoder.coupled_streams(),
        &demixing_matrix,
    )
    .unwrap();

    let frame_size = 960;
    let pcm_in = vec![0i16; frame_size * channels as usize];
    let mut packet = [0u8; 1500];
    let mut pcm_out = vec![0i16; frame_size * channels as usize];

    let len = encoder.encode(&pcm_in, frame_size, &mut packet).unwrap();
    assert!(len > 0);

    let decoded_len = decoder
        .decode(&packet[..len], &mut pcm_out, frame_size, false)
        .unwrap();
    assert_eq!(decoded_len, frame_size);
}

#[test]
fn test_soft_clip_validations() {
    let mut pcm = vec![1.5f32; 4];
    let mut state = vec![0f32; 2];
    assert!(soft_clip(&mut pcm, 2, 2, &mut state).is_ok());

    let mut short_pcm = vec![1.5f32; 3];
    assert_eq!(
        soft_clip(&mut short_pcm, 2, 2, &mut state),
        Err(Error::BadArg)
    );

    let mut pcm = vec![1.5f32; 4];
    let mut too_small_state = vec![0f32; 1];
    assert_eq!(
        soft_clip(&mut pcm, 2, 2, &mut too_small_state),
        Err(Error::BadArg)
    );

    let mut pcm = vec![1.5f32; 4];
    assert_eq!(soft_clip(&mut pcm, 2, -1, &mut state), Err(Error::BadArg));
}
