use opus_codec::AlignedBuffer;
use opus_codec::encoder::Encoder;
use opus_codec::error::Error;
use opus_codec::packet::packet_frame_count;
use opus_codec::repacketizer::{Repacketizer, RepacketizerRef};
use opus_codec::types::{Application, Channels, SampleRate};

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
    assert_eq!(rp.len(), 2);

    // Merge into one packet
    let mut merged = [0u8; 500];
    let merged_len = rp.emit(&mut merged).unwrap();
    assert!(merged_len > 0);

    // Verify the merged packet has 2 frames
    assert_eq!(packet_frame_count(&merged[..merged_len]).unwrap(), 2);
}

#[test]
fn test_init_in_place_null_repacketizer() {
    let err = unsafe { Repacketizer::init_in_place(std::ptr::null_mut()) }.unwrap_err();
    assert_eq!(err, Error::BadArg);
}

#[test]
fn test_init_in_place_unowned_repacketizer() {
    let mut encoder = Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip).unwrap();
    let frame_size = 960;
    let pcm = vec![0i16; frame_size];
    let mut packet = [0u8; 500];
    let len = encoder.encode(&pcm, &mut packet).unwrap();

    let rp_size = Repacketizer::size().unwrap();
    let mut rp_buf = AlignedBuffer::with_capacity_bytes(rp_size);
    let rp_ptr = rp_buf.as_mut_ptr();
    unsafe {
        Repacketizer::init_in_place(rp_ptr).unwrap();
    }
    let mut rp = unsafe { RepacketizerRef::from_raw(rp_ptr) };

    rp.push(&packet[..len]).unwrap();
    let mut out = [0u8; 500];
    let out_len = rp.emit(&mut out).unwrap();
    assert_eq!(&out[..out_len], &packet[..len]);
}

#[test]
fn test_repacketizer_owns_packet_data() {
    let mut encoder = Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip).unwrap();
    let mut rp = Repacketizer::new().unwrap();

    let frame_size = 960;
    let pcm = vec![0i16; frame_size];
    let mut packet = [0u8; 200];
    let len = encoder.encode(&pcm, &mut packet).unwrap();
    let original = packet[..len].to_vec();

    rp.push(&packet[..len]).unwrap();
    packet[..len].fill(0);

    let mut out = [0u8; 200];
    let out_len = rp.emit(&mut out).unwrap();
    assert_eq!(&out[..out_len], original.as_slice());
}
