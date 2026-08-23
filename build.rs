use std::env;
use std::path::PathBuf;

fn main() {
    // Homebrew installs openblas keg-only, so it's not on the default link path.
    let openblas_dir = env::var("OPENBLAS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/opt/homebrew/opt/openblas"));

    println!(
        "cargo:rustc-link-search=native={}",
        openblas_dir.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rerun-if-env-changed=OPENBLAS_DIR");
}
