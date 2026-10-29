//! Serial sender (feature `serial`). Build with `--features serial`.

use std::io::{self, Write};
use std::time::Duration;

use crate::wire::Packet;
use crate::TactileSink;

pub struct SerialLink {
    port: Box<dyn serialport::SerialPort>,
}

impl SerialLink {
    /// Open e.g. `COM5` (Windows) or `/dev/ttyUSB0` (Linux) at the given baud rate.
    pub fn open(path: &str, baud: u32) -> Result<Self, serialport::Error> {
        let port = serialport::new(path, baud)
            .timeout(Duration::from_millis(50))
            .open()?;
        Ok(Self { port })
    }
}

impl TactileSink for SerialLink {
    fn send(&mut self, packet: &Packet) -> io::Result<()> {
        self.port.write_all(&packet.encode())?;
        self.port.flush()
    }
}
