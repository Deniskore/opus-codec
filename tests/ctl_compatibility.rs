#![cfg(not(opus_codec_system_lib))]

use opus_codec::multistream::{Mapping, MultistreamDecoder, MultistreamEncoder};
use opus_codec::projection::ProjectionEncoder;
use opus_codec::{
    Application, Bandwidth, Bitrate, Channels, Complexity, Decoder, Encoder, Error,
    ExpertFrameDuration, Result, SampleRate, Signal,
};

type CtlCase<T> = (&'static str, fn(&mut T) -> Result<()>);

fn run_ctl_cases<T>(target: &mut T, cases: &[CtlCase<T>]) {
    for (name, case) in cases {
        case(target).unwrap_or_else(|err| panic!("bundled Opus CTL {name} failed: {err:?}"));
    }
}

#[test]
fn every_single_stream_ctl_is_supported_by_bundled_opus() {
    let mut encoder =
        Encoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Audio).unwrap();
    let encoder_cases: &[CtlCase<Encoder>] = &[
        ("bitrate", |encoder| {
            encoder.set_bitrate(Bitrate::Custom(96_000))?;
            let _ = encoder.bitrate()?;
            Ok(())
        }),
        ("complexity", |encoder| {
            encoder.set_complexity(Complexity::new(5))?;
            let _ = encoder.complexity()?;
            Ok(())
        }),
        ("vbr", |encoder| {
            encoder.set_vbr(true)?;
            let _ = encoder.vbr()?;
            Ok(())
        }),
        ("vbr_constraint", |encoder| {
            encoder.set_vbr_constraint(true)?;
            let _ = encoder.vbr_constraint()?;
            Ok(())
        }),
        ("inband_fec", |encoder| {
            encoder.set_inband_fec(true)?;
            let _ = encoder.inband_fec()?;
            Ok(())
        }),
        ("packet_loss", |encoder| {
            encoder.set_packet_loss_perc(10)?;
            let _ = encoder.packet_loss_perc()?;
            Ok(())
        }),
        ("dtx", |encoder| {
            encoder.set_dtx(true)?;
            let _ = encoder.dtx()?;
            let _ = encoder.in_dtx()?;
            Ok(())
        }),
        ("max_bandwidth", |encoder| {
            encoder.set_max_bandwidth(Bandwidth::Wideband)?;
            let _ = encoder.max_bandwidth()?;
            Ok(())
        }),
        ("bandwidth", |encoder| {
            encoder.set_bandwidth(Bandwidth::Wideband)?;
            let _ = encoder.bandwidth()?;
            Ok(())
        }),
        ("force_channels", |encoder| {
            encoder.set_force_channels(Some(Channels::Mono))?;
            let _ = encoder.force_channels()?;
            Ok(())
        }),
        ("signal", |encoder| {
            encoder.set_signal(Signal::Music)?;
            let _ = encoder.signal()?;
            Ok(())
        }),
        ("lookahead", |encoder| {
            let _ = encoder.lookahead()?;
            Ok(())
        }),
        ("final_range", |encoder| {
            let _ = encoder.final_range()?;
            Ok(())
        }),
        ("lsb_depth", |encoder| {
            assert_eq!(encoder.set_lsb_depth(7), Err(Error::BadArg));
            encoder.set_lsb_depth(16)?;
            assert_eq!(encoder.lsb_depth()?, 16);
            Ok(())
        }),
        ("expert_frame_duration", |encoder| {
            encoder.set_expert_frame_duration(ExpertFrameDuration::Ms20)?;
            let _ = encoder.expert_frame_duration()?;
            Ok(())
        }),
        ("prediction_disabled", |encoder| {
            encoder.set_prediction_disabled(true)?;
            let _ = encoder.prediction_disabled()?;
            Ok(())
        }),
        ("phase_inversion", |encoder| {
            encoder.set_phase_inversion_disabled(true)?;
            let _ = encoder.phase_inversion_disabled()?;
            Ok(())
        }),
        ("reset", Encoder::reset),
    ];
    run_ctl_cases(&mut encoder, encoder_cases);

    let mut decoder = Decoder::new(SampleRate::Hz48000, Channels::Stereo).unwrap();
    let decoder_cases: &[CtlCase<Decoder>] = &[
        ("sample_rate", |decoder| {
            let _ = decoder.get_sample_rate()?;
            Ok(())
        }),
        ("pitch", |decoder| {
            let _ = decoder.get_pitch()?;
            Ok(())
        }),
        ("last_packet_duration", |decoder| {
            let _ = decoder.get_last_packet_duration()?;
            Ok(())
        }),
        ("final_range", |decoder| {
            let _ = decoder.final_range()?;
            Ok(())
        }),
        ("gain", |decoder| {
            decoder.set_gain(128)?;
            let _ = decoder.gain()?;
            Ok(())
        }),
        ("phase_inversion", |decoder| {
            decoder.set_phase_inversion_disabled(true)?;
            let _ = decoder.phase_inversion_disabled()?;
            Ok(())
        }),
        ("reset", Decoder::reset),
    ];
    run_ctl_cases(&mut decoder, decoder_cases);
}

#[test]
fn every_multistream_ctl_is_supported_by_bundled_opus() {
    let mapping_table = [0u8, 1u8];
    let mapping = Mapping {
        channels: 2,
        streams: 1,
        coupled_streams: 1,
        mapping: &mapping_table,
    };
    let mut encoder =
        MultistreamEncoder::new(SampleRate::Hz48000, Application::Audio, mapping).unwrap();
    let encoder_cases: &[CtlCase<MultistreamEncoder>] = &[
        ("final_range", |encoder| {
            let _ = encoder.final_range()?;
            Ok(())
        }),
        ("bitrate", |encoder| {
            encoder.set_bitrate(Bitrate::Custom(96_000))?;
            let _ = encoder.bitrate()?;
            Ok(())
        }),
        ("complexity", |encoder| {
            encoder.set_complexity(Complexity::new(5))?;
            let _ = encoder.complexity()?;
            Ok(())
        }),
        ("lsb_depth", |encoder| {
            encoder.set_lsb_depth(16)?;
            let _ = encoder.lsb_depth()?;
            Ok(())
        }),
        ("dtx", |encoder| {
            encoder.set_dtx(true)?;
            let _ = encoder.dtx()?;
            let _ = encoder.in_dtx()?;
            Ok(())
        }),
        ("inband_fec", |encoder| {
            encoder.set_inband_fec(true)?;
            let _ = encoder.inband_fec()?;
            Ok(())
        }),
        ("packet_loss", |encoder| {
            encoder.set_packet_loss_perc(10)?;
            let _ = encoder.packet_loss_perc()?;
            Ok(())
        }),
        ("vbr", |encoder| {
            encoder.set_vbr(true)?;
            let _ = encoder.vbr()?;
            Ok(())
        }),
        ("vbr_constraint", |encoder| {
            encoder.set_vbr_constraint(true)?;
            let _ = encoder.vbr_constraint()?;
            Ok(())
        }),
        ("max_bandwidth", |encoder| {
            encoder.set_max_bandwidth(Bandwidth::Wideband)?;
            let _ = encoder.max_bandwidth()?;
            Ok(())
        }),
        ("bandwidth", |encoder| {
            encoder.set_bandwidth(Bandwidth::Wideband)?;
            let _ = encoder.bandwidth()?;
            Ok(())
        }),
        ("force_channels", |encoder| {
            encoder.set_force_channels(Some(Channels::Mono))?;
            let _ = encoder.force_channels()?;
            Ok(())
        }),
        ("signal", |encoder| {
            encoder.set_signal(Signal::Music)?;
            let _ = encoder.signal()?;
            Ok(())
        }),
        ("lookahead", |encoder| {
            let _ = encoder.lookahead()?;
            Ok(())
        }),
        ("encoder_state", |encoder| {
            let ptr = unsafe { encoder.encoder_state_ptr(0)? };
            if ptr.is_null() {
                Err(Error::InternalError)
            } else {
                Ok(())
            }
        }),
        ("reset", MultistreamEncoder::reset),
    ];
    run_ctl_cases(&mut encoder, encoder_cases);

    let mut decoder = MultistreamDecoder::new(SampleRate::Hz48000, mapping).unwrap();
    let decoder_cases: &[CtlCase<MultistreamDecoder>] = &[
        ("final_range", |decoder| {
            let _ = decoder.final_range()?;
            Ok(())
        }),
        ("gain", |decoder| {
            decoder.set_gain(128)?;
            let _ = decoder.gain()?;
            Ok(())
        }),
        ("phase_inversion", |decoder| {
            decoder.set_phase_inversion_disabled(true)?;
            let _ = decoder.phase_inversion_disabled()?;
            Ok(())
        }),
        ("sample_rate", |decoder| {
            let _ = decoder.get_sample_rate()?;
            Ok(())
        }),
        ("pitch", |decoder| {
            let _ = decoder.get_pitch()?;
            Ok(())
        }),
        ("last_packet_duration", |decoder| {
            let _ = decoder.get_last_packet_duration()?;
            Ok(())
        }),
        ("decoder_state", |decoder| {
            let ptr = unsafe { decoder.decoder_state_ptr(0)? };
            if ptr.is_null() {
                Err(Error::InternalError)
            } else {
                Ok(())
            }
        }),
        ("reset", MultistreamDecoder::reset),
    ];
    run_ctl_cases(&mut decoder, decoder_cases);
}

#[test]
fn every_projection_encoder_ctl_is_supported_by_bundled_opus() {
    let mut encoder =
        ProjectionEncoder::new(SampleRate::Hz48000, 4, 3, Application::Audio).unwrap();
    let cases: &[CtlCase<ProjectionEncoder>] = &[
        ("bitrate", |encoder| {
            encoder.set_bitrate(Bitrate::Custom(128_000))?;
            let _ = encoder.bitrate()?;
            Ok(())
        }),
        ("demixing_matrix_size", |encoder| {
            let _ = encoder.demixing_matrix_size()?;
            Ok(())
        }),
        ("demixing_matrix_gain", |encoder| {
            let _ = encoder.demixing_matrix_gain()?;
            Ok(())
        }),
        ("demixing_matrix", |encoder| {
            let _ = encoder.demixing_matrix_bytes()?;
            Ok(())
        }),
    ];
    run_ctl_cases(&mut encoder, cases);
}

#[cfg(feature = "dred")]
#[test]
fn every_dred_ctl_uses_the_bundled_opus_signature() {
    use opus_codec::DredDecoder;

    let mut encoder =
        Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
    encoder.set_dred_duration(2).unwrap();
    assert_eq!(encoder.dred_duration().unwrap(), 2);
    assert_eq!(
        unsafe { encoder.set_dnn_blob(std::ptr::null(), 0) },
        Err(Error::BadArg)
    );

    let mut decoder = Decoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
    assert_eq!(
        unsafe { decoder.set_dnn_blob(std::ptr::null(), 0) },
        Err(Error::BadArg)
    );

    let mut dred_decoder = DredDecoder::new().unwrap();
    assert_eq!(
        unsafe { dred_decoder.set_dnn_blob(&[]) },
        Err(Error::BadArg)
    );
}
