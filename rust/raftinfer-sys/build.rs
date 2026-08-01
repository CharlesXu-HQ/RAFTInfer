use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=RAFTINFER_ENABLE_CUDA");
    println!("cargo:rerun-if-env-changed=RAFTINFER_NATIVE_LIBRARY_TYPE");
    println!("cargo:rerun-if-env-changed=RAFTINFER_NATIVE_LIBRARY_DIRS");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let cuda = std::env::var("RAFTINFER_ENABLE_CUDA").unwrap_or_else(|_| "OFF".into());
    let cuda_enabled = parse_cmake_bool(&cuda, "RAFTINFER_ENABLE_CUDA");
    let library_type = std::env::var("RAFTINFER_NATIVE_LIBRARY_TYPE")
        .unwrap_or_else(|_| "STATIC".into())
        .to_ascii_uppercase();
    if !matches!(library_type.as_str(), "STATIC" | "SHARED") {
        panic!("RAFTINFER_NATIVE_LIBRARY_TYPE must be STATIC or SHARED, got: {library_type}");
    }
    let dst = cmake::Config::new("../..")
        .generator("Ninja")
        .define("RAFTINFER_ENABLE_CUDA", cuda)
        .define("RAFTINFER_NATIVE_LIBRARY_TYPE", &library_type)
        .define("RAFTINFER_BUILD_TESTS", "OFF")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    let link_kind = if library_type == "SHARED" {
        "dylib"
    } else {
        "static"
    };
    println!("cargo:rustc-link-lib={link_kind}=raftinfer_cpp");
    if cuda_enabled {
        for directory in native_library_dirs() {
            println!("cargo:rustc-link-search=native={}", directory.display());
        }
        println!("cargo:rustc-link-lib=dylib=rmm");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublasLt");
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
    println!("cargo:rerun-if-changed=../../cpp");
    println!("cargo:rerun-if-changed=../../CMakeLists.txt");
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
    if let Some(paths) = std::env::var_os("RAFTINFER_NATIVE_LIBRARY_DIRS") {
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
