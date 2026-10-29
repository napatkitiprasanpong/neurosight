//! Link between the Rust software and the FPGA tactile output.
//!
//! * [`wire`]  - the 6-byte UART event packet and a byte-at-a-time parser.
//! * [`rom`]   - event -> 5 x 5 vibration pattern (motor index + 8-bit duty), the content of
//!               the FPGA pattern ROM.
//! * [`sched`] - integer-millisecond model of the safety scheduler (what the RTL implements).
//! * [`export`] - golden vectors (JSON) and the ROM image (hex) for the RTL testbenches.
//!
//! The Rust encoder (`neurosight-encoder`) stays the single source of truth: everything here is
//! derived from it, so the RTL never has to be kept in sync by hand.
//! Full description: `docs/FPGA_SPEC.md`.

pub mod export;
pub mod rom;
pub mod sched;
pub mod wire;

#[cfg(feature = "serial")]
pub mod link;

use std::io;

use wire::Packet;

/// Anything that can receive event packets (a serial port, a test recorder, a printer).
pub trait TactileSink {
    fn send(&mut self, packet: &Packet) -> io::Result<()>;
}

/// Prints every packet as hex; useful before the board exists.
pub struct PrintSink;

impl TactileSink for PrintSink {
    fn send(&mut self, packet: &Packet) -> io::Result<()> {
        println!("tactile packet: {}", wire::hex(&packet.encode()));
        Ok(())
    }
}
