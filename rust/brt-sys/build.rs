fn main() {
    println!("cargo:rerun-if-env-changed=BRT_ENABLE_CUDA");
    println!("cargo:rerun-if-env-changed=BRT_NATIVE_LIBRARY_TYPE");
    let cuda = std::env::var("BRT_ENABLE_CUDA").unwrap_or_else(|_| "OFF".into());
    let cuda_enabled = parse_cmake_bool(&cuda, "BRT_ENABLE_CUDA");
    let library_type = std::env::var("BRT_NATIVE_LIBRARY_TYPE")
        .unwrap_or_else(|_| if cuda_enabled { "SHARED" } else { "STATIC" }.into())
        .to_ascii_uppercase();
    if !matches!(library_type.as_str(), "STATIC" | "SHARED") {
        panic!("BRT_NATIVE_LIBRARY_TYPE must be STATIC or SHARED, got: {library_type}");
    }
    let dst = cmake::Config::new("../..")
        .define("BRT_ENABLE_CUDA", cuda)
        .define("BRT_NATIVE_LIBRARY_TYPE", &library_type)
        .define("BRT_BUILD_TESTS", "OFF")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    let link_kind = if library_type == "SHARED" {
        "dylib"
    } else {
        "static"
    };
    println!("cargo:rustc-link-lib={link_kind}=brt_cpp");
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
