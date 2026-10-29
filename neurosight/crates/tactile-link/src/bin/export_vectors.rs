//! Writes the golden vectors and the pattern ROM image for the RTL testbenches.
//!
//! Usage: `cargo run -p neurosight-tactile-link --bin export-vectors -- [output_dir]`
//! (default output directory: `fpga/vectors`).

use std::fs;
use std::path::PathBuf;

use neurosight_encoder::SafetyLimits;
use neurosight_tactile_link::export::{
    packet_vectors_json, pattern_rom_hex, pattern_vectors_json, scheduler_vectors_json,
};

fn main() -> std::io::Result<()> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fpga/vectors".to_string())
        .into();
    fs::create_dir_all(&dir)?;
    let limits = SafetyLimits::default();

    let files = [
        ("pattern_vectors.json", pattern_vectors_json(&limits)),
        ("pattern_rom.hex", pattern_rom_hex(&limits)),
        ("packet_vectors.json", packet_vectors_json()),
        ("scheduler_vectors.json", scheduler_vectors_json(&limits)),
    ];
    for (name, content) in files.iter() {
        let path = dir.join(name);
        fs::write(&path, content)?;
        println!("wrote {} ({} bytes)", path.display(), content.len());
    }
    Ok(())
}
