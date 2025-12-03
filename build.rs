use std::env;

fn main() {
    emit_rerun_directives();
    let opts = BuildOptions::from_env();

    if opts.use_system_lib {
        handle_system_lib(&opts);
    } else {
        build_bundled_and_link(&opts);
    }

    generate_bindings();
}

struct BuildOptions {
    use_system_lib: bool,
    dred_enabled: bool,
    presume_avx: bool,
    target_arch: String,
    avx_allowed: bool,
}

impl BuildOptions {
    fn from_env() -> Self {
        let use_system_lib = env::var("CARGO_FEATURE_SYSTEM_LIB").is_ok();
        let dred_enabled = env::var("CARGO_FEATURE_DRED").is_ok();
        let presume_avx = env::var("CARGO_FEATURE_PRESUME_AVX2").is_ok();
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let avx_allowed = presume_avx && matches!(target_arch.as_str(), "x86" | "x86_64");

        Self {
            use_system_lib,
            dred_enabled,
            presume_avx,
            target_arch,
            avx_allowed,
        }
    }
}

fn emit_rerun_directives() {
    println!("cargo:rerun-if-changed=opus/include/opus.h");
    println!("cargo:rerun-if-changed=opus/include/opus_defines.h");
    println!("cargo:rerun-if-changed=opus/include/opus_types.h");
    println!("cargo:rerun-if-changed=opus/include/opus_multistream.h");
    println!("cargo:rerun-if-changed=opus/include/opus_projection.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=opus/dnn/download_model.sh");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SYSTEM_LIB");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PRESUME_AVX2");
}

fn handle_system_lib(opts: &BuildOptions) {
    if opts.dred_enabled {
        println!(
            "cargo:warning=system-lib feature enabled; ensure the system libopus includes DRED support"
        );
    }
    if opts.presume_avx {
        println!(
            "cargo:warning=presume-avx2 feature enabled; ensure the system libopus was built with OPUS_X86_PRESUME_AVX2"
        );
    }
    link_system_lib();
}

fn build_bundled_and_link(opts: &BuildOptions) {
    if opts.dred_enabled {
        ensure_dred_assets();
    }
    if opts.presume_avx && !opts.avx_allowed {
        println!(
            "cargo:warning=presume-avx2 feature only applies to x86/x86_64 targets; ignoring for {}",
            opts.target_arch
        );
    }

    let dst = build_bundled(opts.dred_enabled, opts.avx_allowed);
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=opus");
}

fn build_bundled(dred_enabled: bool, presume_avx: bool) -> std::path::PathBuf {
    let mut config = cmake::Config::new("opus");

    config.profile("Release");

    if should_use_msvc_crt_flag() {
        let profile = env::var("PROFILE").unwrap_or_default();
        let crt_flag = if profile.eq_ignore_ascii_case("debug") {
            "/MDd"
        } else {
            "/MD"
        };
        config.cflag(crt_flag);
    }

    config
        .define("OPUS_BUILD_SHARED_LIBRARY", "OFF")
        .define("OPUS_BUILD_TESTING", "OFF")
        .define("OPUS_BUILD_PROGRAMS", "OFF")
        .define("OPUS_DRED", if dred_enabled { "ON" } else { "OFF" })
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("OPUS_DISABLE_INTRINSICS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");

    if presume_avx {
        config
            .define("OPUS_X86_PRESUME_AVX2", "ON")
            .define("OPUS_X86_MAY_HAVE_AVX2", "ON");
    }

    config.build()
}

fn link_system_lib() {
    pkg_config::Config::new()
        .atleast_version("1.5.2")
        .probe("opus")
        .expect("system-lib feature requested but pkg-config couldn't find libopus");
}

fn generate_bindings() {
    let bindings_path = std::path::Path::new("src/bindings.rs");

    if bindings_path.exists() {
        println!(
            "cargo:warning=Using existing src/bindings.rs. Delete this file to force regeneration."
        );
        return;
    }

    let bindings = bindgen::Builder::default()
        .header("opus/include/opus.h")
        .header("opus/include/opus_defines.h")
        .header("opus/include/opus_types.h")
        .header("opus/include/opus_multistream.h")
        .header("opus/include/opus_projection.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(bindings_path)
        .expect("Couldn't write bindings!");
}

fn should_use_msvc_crt_flag() -> bool {
    matches!(
        env::var("CARGO_CFG_TARGET_FAMILY").as_deref(),
        Ok("windows")
    ) && matches!(env::var("CARGO_CFG_TARGET_ENV").as_deref(), Ok("msvc"))
}

fn ensure_dred_assets() {
    use std::path::Path;
    use std::process::Command;

    const REQUIRED_FILE: &str = "opus/dnn/fargan_data.h";
    if Path::new(REQUIRED_FILE).exists() {
        return;
    }

    let script = Path::new("opus/dnn/download_model.sh");
    if !script.exists() {
        panic!("DRED feature requires {script:?}, but it was not found");
    }

    let status = Command::new("sh")
        .arg("dnn/download_model.sh")
        .arg("735117b")
        .current_dir("opus")
        .status()
        .expect("failed to spawn DRED model download script");

    if !status.success() {
        panic!("downloading DRED model assets failed (exit status: {status})");
    }

    if !Path::new(REQUIRED_FILE).exists() {
        panic!("DRED model download completed but {REQUIRED_FILE} is still missing");
    }
}
