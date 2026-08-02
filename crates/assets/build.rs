fn main() {
    // `rust-embed` expands files outside this crate's source directory. Cargo's
    // normal dependency tracking does not reliably notice newly added assets,
    // so explicitly invalidate the crate whenever the asset tree changes.
    println!("cargo::rerun-if-changed=../../assets");
}
