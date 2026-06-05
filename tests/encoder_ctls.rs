use opus_codec::{
    Application, Bandwidth, Bitrate, Channels, Complexity, Encoder, SampleRate, Signal,
};

#[test]
fn encoder_control_roundtrip() {
    let sr = SampleRate::Hz48000;
    let mut encoder =
        Encoder::new(sr, Channels::Stereo, Application::Audio).expect("create encoder");

    encoder
        .set_bitrate(Bitrate::Custom(96_000))
        .expect("set bitrate");
    match encoder.bitrate().expect("get bitrate") {
        Bitrate::Custom(bps) => assert_eq!(bps, 96_000),
        other => panic!("unexpected bitrate variant: {other:?}"),
    }

    encoder
        .set_complexity(Complexity::new(4))
        .expect("set complexity");
    assert_eq!(encoder.complexity().expect("get complexity").value(), 4);

    encoder.set_vbr(false).expect("disable vbr");
    assert!(!encoder.vbr().expect("get vbr"));

    encoder
        .set_vbr_constraint(true)
        .expect("set vbr constraint");
    assert!(encoder.vbr_constraint().expect("get vbr constraint"));

    encoder.set_inband_fec(true).expect("enable fec");
    assert!(encoder.inband_fec().expect("get fec"));

    encoder.set_packet_loss_perc(15).expect("set packet loss");
    assert_eq!(encoder.packet_loss_perc().expect("get packet loss"), 15);

    encoder.set_signal(Signal::Music).expect("set signal");
    assert_eq!(encoder.signal().expect("get signal"), Signal::Music);

    encoder
        .set_max_bandwidth(Bandwidth::Wideband)
        .expect("set max bandwidth");
    assert_eq!(
        encoder.max_bandwidth().expect("get max bandwidth"),
        Bandwidth::Wideband
    );

    encoder
        .set_force_channels(Some(Channels::Mono))
        .expect("force mono");
    assert_eq!(
        encoder.force_channels().expect("get forced channels"),
        Some(Channels::Mono)
    );

    encoder
        .set_force_channels(None)
        .expect("clear force channels");
    assert_eq!(encoder.force_channels().expect("get forced channels"), None);
}

#[test]
fn lookahead_uses_the_configured_sample_rate() {
    let mut encoder_8k = Encoder::new(SampleRate::Hz8000, Channels::Mono, Application::Audio)
        .expect("create 8 kHz encoder");
    let mut encoder_48k = Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio)
        .expect("create 48 kHz encoder");

    let lookahead_8k = encoder_8k.lookahead().expect("8 kHz lookahead");
    let lookahead_48k = encoder_48k.lookahead().expect("48 kHz lookahead");
    assert_eq!(lookahead_48k, lookahead_8k * 6);
}

#[cfg(all(feature = "dred", not(opus_codec_system_lib)))]
#[test]
fn dred_duration_is_an_encoder_ctl_in_ten_ms_frames() {
    let mut encoder = Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio)
        .expect("create DRED encoder");

    encoder
        .set_dred_duration(3)
        .expect("set three 10-ms DRED frames");
    assert_eq!(encoder.dred_duration().expect("get DRED duration"), 3);
    assert_eq!(
        encoder.set_dred_duration(-1),
        Err(opus_codec::Error::BadArg)
    );
}
