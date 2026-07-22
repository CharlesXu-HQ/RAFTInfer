fn main() {
    let cuda = std::env::var("BRT_ENABLE_CUDA").unwrap_or_else(|_| "OFF".into());
    let dst = cmake::Config::new("../..")
        .define("BRT_ENABLE_CUDA", cuda)
        .define("BRT_BUILD_TESTS", "OFF")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=brt_cpp");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
    println!("cargo:rerun-if-changed=../../cpp");
    println!("cargo:rerun-if-changed=../../CMakeLists.txt");
}
