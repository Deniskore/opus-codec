use std::env;

fn main() {
    println!("cargo:rerun-if-changed=opus/include/opus.h");
    println!("cargo:rerun-if-changed=opus/include/opus_defines.h");
    println!("cargo:rerun-if-changed=opus/include/opus_types.h");
    println!("cargo:rerun-if-changed=opus/include/opus_multistream.h");
    println!("cargo:rerun-if-changed=opus/include/opus_projection.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=opus/dnn/download_model.sh");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SYSTEM_LIB");

    let use_system_lib = env::var("CARGO_FEATURE_SYSTEM_LIB").is_ok();
    let dred_enabled = env::var("CARGO_FEATURE_DRED").is_ok();

    if use_system_lib {
        if dred_enabled {
            println!(
                "cargo:warning=system-lib feature enabled; ensure the system libopus includes DRED support"
            );
        }
        link_system_lib();
    } else {
        if dred_enabled {
            ensure_dred_assets();
        }
        let dst = build_bundled(dred_enabled);
        println!("cargo:rustc-link-search=native={}/lib", dst.display());
        println!("cargo:rustc-link-lib=static=opus");
    }

    generate_bindings();
}

fn build_bundled(dred_enabled: bool) -> std::path::PathBuf {
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
