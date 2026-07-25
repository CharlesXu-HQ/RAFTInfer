use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=BRT_ENABLE_CUDA");
    println!("cargo:rerun-if-env-changed=BRT_NATIVE_LIBRARY_DIRS");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let cuda = std::env::var("BRT_ENABLE_CUDA").unwrap_or_else(|_| "OFF".into());
    if !parse_cmake_bool(&cuda, "BRT_ENABLE_CUDA") {
        return;
    }
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if !matches!(target_os.as_str(), "linux" | "macos") {
        return;
    }
    for directory in native_library_dirs() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", directory.display());
    }
}

fn parse_cmake_bool(value: &str, name: &str) -> bool {
    match value.to_ascii_uppercase().as_str() {
        "1" | "ON" | "TRUE" | "YES" | "Y" => true,
        "0" | "OFF" | "FALSE" | "NO" | "N" | "" => false,
        _ => panic!("{name} must be a CMake boolean, got: {value}"),
    }
}

fn native_library_dirs() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    if let Some(paths) = std::env::var_os("BRT_NATIVE_LIBRARY_DIRS") {
        for path in std::env::split_paths(&paths) {
            add_existing_directory(&mut directories, path);
        }
    }
    if let Some(prefix) = std::env::var_os("CONDA_PREFIX") {
        let prefix = PathBuf::from(prefix);
        add_existing_directory(&mut directories, prefix.join("lib"));
        if let Ok(arch) = std::env::var("CARGO_CFG_TARGET_ARCH") {
            add_existing_directory(
                &mut directories,
                prefix.join(format!("targets/{arch}-linux/lib")),
            );
        }
    }
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(prefix) = std::env::var_os(variable) {
            add_existing_directory(&mut directories, PathBuf::from(prefix).join("lib64"));
        }
    }
    add_existing_directory(&mut directories, PathBuf::from("/usr/local/cuda/lib64"));
    directories
}

fn add_existing_directory(directories: &mut Vec<PathBuf>, path: PathBuf) {
    if Path::new(&path).is_dir() && !directories.contains(&path) {
        directories.push(path);
    }
}
