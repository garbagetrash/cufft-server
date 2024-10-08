use std::env;
use std::path::PathBuf;

fn main() {
    // Build the CUDA C library
    let dst = cmake::build("libcufft_server");

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");

    // Tell cargo to tell rustc to link the library
    println!("cargo:rustc-link-lib=static=cufft_server");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cufft");

    // The bindgen::Builder is the main entrypoint to bindgen, and lets you
    // build up options for the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header("wrapper.h")
        // Tell cargo to invalidate the built crate whenever any  of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
