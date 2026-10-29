//! UART event packet (PC -> FPGA).
//!
//! ```text
//! byte 0  start      0xA5
//! byte 1  action id  1..=8   (stable ids, see EVENTS; NOT the order of the Rust enum)
//! byte 2  zone       0 left, 1 center, 2 right
//! byte 3  distance   0 near, 1 medium, 2 far   (later replaced by the ToF reading)
//! byte 4  confidence 0..=100
//! byte 5  checksum   XOR of bytes 1..=4
//! ```
//! Check order when decoding (the RTL must use the same order so error codes match):
//! length, start byte, checksum, action, zone, distance, confidence.

use neurosight_common::{Action, Distance, Event, Zone};

pub const START_BYTE: u8 = 0xA5;
pub const PACKET_LEN: usize = 6;

/// The events the tactile output supports, with their stable wire ids.
/// Change this one table to change the event set (then re-export the vectors).
pub const EVENTS: [(u8, Action, &str); 8] = [
    (1, Action::Appeared, "appeared"),
    (2, Action::Left, "left"),
    (3, Action::Waving, "waving"),
    (4, Action::ReachingOut, "reaching_out"),
    (5, Action::Nodding, "nodding"),
    (6, Action::ShakingHead, "shaking_head"),
    (7, Action::LookedLeft, "looked_left"),
    (8, Action::LookedRight, "looked_right"),
];

pub const ZONES: [Zone; 3] = [Zone::Left, Zone::Center, Zone::Right];
pub const DISTANCES: [Distance; 3] = [Distance::Near, Distance::Medium, Distance::Far];

pub fn action_id(action: Action) -> Option<u8> {
    EVENTS.iter().find(|e| e.1 == action).map(|e| e.0)
}

pub fn action_from_id(id: u8) -> Option<Action> {
    EVENTS.iter().find(|e| e.0 == id).map(|e| e.1)
}

pub fn action_name(action: Action) -> &'static str {
    EVENTS
        .iter()
        .find(|e| e.1 == action)
        .map(|e| e.2)
        .unwrap_or("unsupported")
}

pub fn zone_code(zone: Zone) -> u8 {
    match zone {
        Zone::Left => 0,
        Zone::Center => 1,
        Zone::Right => 2,
    }
}

pub fn zone_from_code(code: u8) -> Option<Zone> {
    ZONES.get(code as usize).copied()
}

pub fn distance_code(distance: Distance) -> u8 {
    match distance {
        Distance::Near => 0,
        Distance::Medium => 1,
        Distance::Far => 2,
    }
}

pub fn distance_from_code(code: u8) -> Option<Distance> {
    DISTANCES.get(code as usize).copied()
}

/// What selects a pattern in the ROM (confidence does not).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    pub action: Action,
    pub zone: Zone,
    pub distance: Distance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packet {
    pub key: Key,
    pub confidence: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketError {
    WrongLength,
    BadStart,
    BadChecksum,
    UnknownAction,
    BadZone,
    BadDistance,
    BadConfidence,
}

impl PacketError {
    /// Numeric code the RTL error register should use.
    pub fn code(&self) -> u8 {
        match self {
            PacketError::WrongLength => 1,
            PacketError::BadStart => 2,
            PacketError::BadChecksum => 3,
            PacketError::UnknownAction => 4,
            PacketError::BadZone => 5,
            PacketError::BadDistance => 6,
            PacketError::BadConfidence => 7,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            PacketError::WrongLength => "wrong_length",
            PacketError::BadStart => "bad_start",
            PacketError::BadChecksum => "bad_checksum",
            PacketError::UnknownAction => "unknown_action",
            PacketError::BadZone => "bad_zone",
            PacketError::BadDistance => "bad_distance",
            PacketError::BadConfidence => "bad_confidence",
        }
    }
}

pub fn checksum(body: &[u8]) -> u8 {
    body.iter().fold(0, |acc, b| acc ^ b)
}

/// Build a frame from raw field values (also used to build deliberately invalid frames).
pub fn frame(action_id: u8, zone: u8, distance: u8, confidence: u8) -> [u8; PACKET_LEN] {
    let body = [action_id, zone, distance, confidence];
    [START_BYTE, body[0], body[1], body[2], body[3], checksum(&body)]
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

impl Packet {
    pub fn new(key: Key, confidence: u8) -> Option<Packet> {
        action_id(key.action)?;
        Some(Packet {
            key,
            confidence: confidence.min(100),
        })
    }

    /// Build a packet from a software event. Returns None for actions the tactile output does
    /// not support (still, moving, present, looked up / down). A missing zone becomes center,
    /// a missing distance becomes medium (the same defaults the encoder uses).
    pub fn from_event(event: &Event) -> Option<Packet> {
        let key = Key {
            action: event.action,
            zone: event.zone.unwrap_or(Zone::Center),
            distance: event.distance.unwrap_or(Distance::Medium),
        };
        let confidence = (event.confidence.clamp(0.0, 1.0) * 100.0).round() as u8;
        Packet::new(key, confidence)
    }

    /// The event the FPGA reconstructs from the packet (what the encoder is run on).
    pub fn to_event(&self) -> Event {
        Event {
            time_s: 0.0,
            action: self.key.action,
            zone: Some(self.key.zone),
            distance: Some(self.key.distance),
            facing_camera: false,
            confidence: self.confidence as f32 / 100.0,
        }
    }

    pub fn encode(&self) -> [u8; PACKET_LEN] {
        frame(
            action_id(self.key.action).expect("supported action"),
            zone_code(self.key.zone),
            distance_code(self.key.distance),
            self.confidence,
        )
    }

    pub fn decode(bytes: &[u8]) -> Result<Packet, PacketError> {
        if bytes.len() != PACKET_LEN {
            return Err(PacketError::WrongLength);
        }
        if bytes[0] != START_BYTE {
            return Err(PacketError::BadStart);
        }
        if checksum(&bytes[1..5]) != bytes[5] {
            return Err(PacketError::BadChecksum);
        }
        let action = action_from_id(bytes[1]).ok_or(PacketError::UnknownAction)?;
        let zone = zone_from_code(bytes[2]).ok_or(PacketError::BadZone)?;
        let distance = distance_from_code(bytes[3]).ok_or(PacketError::BadDistance)?;
        if bytes[4] > 100 {
            return Err(PacketError::BadConfidence);
        }
        Ok(Packet {
            key: Key {
                action,
                zone,
                distance,
            },
            confidence: bytes[4],
        })
    }
}

/// Result of feeding one byte to the parser.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parsed {
    Nothing,
    Packet(Packet),
    Error(PacketError),
}

/// Byte-at-a-time parser; the RTL UART packet receiver has the same behaviour:
/// bytes that are not a start byte are ignored while idle (no error); after a start byte it
/// collects 6 bytes in total, then decodes. A bad frame is dropped and the parser goes idle
/// (it does not re-scan the dropped bytes).
#[derive(Default)]
pub struct PacketParser {
    buf: [u8; PACKET_LEN],
    len: usize,
}

impl PacketParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, byte: u8) -> Parsed {
        if self.len == 0 {
            if byte != START_BYTE {
                return Parsed::Nothing;
            }
            self.buf[0] = byte;
            self.len = 1;
            return Parsed::Nothing;
        }
        self.buf[self.len] = byte;
        self.len += 1;
        if self.len < PACKET_LEN {
            return Parsed::Nothing;
        }
        self.len = 0;
        match Packet::decode(&self.buf) {
            Ok(p) => Parsed::Packet(p),
            Err(e) => Parsed::Error(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave_packet() -> Packet {
        Packet::new(
            Key {
                action: Action::Waving,
                zone: Zone::Left,
                distance: Distance::Near,
            },
            83,
        )
        .unwrap()
    }

    #[test]
    fn encode_layout_and_checksum() {
        let bytes = wave_packet().encode();
        assert_eq!(bytes, [0xA5, 3, 0, 0, 83, 3 ^ 0 ^ 0 ^ 83]);
    }

    #[test]
    fn round_trip_every_supported_packet() {
        for (_, action, _) in EVENTS.iter() {
            for zone in ZONES.iter() {
                for distance in DISTANCES.iter() {
                    let p = Packet::new(
                        Key {
                            action: *action,
                            zone: *zone,
                            distance: *distance,
                        },
                        100,
                    )
                    .unwrap();
                    assert_eq!(Packet::decode(&p.encode()), Ok(p));
                }
            }
        }
    }

    #[test]
    fn invalid_frames_are_rejected_in_documented_order() {
        let good = wave_packet().encode();
        assert_eq!(Packet::decode(&good[..5]), Err(PacketError::WrongLength));
        let mut bad_start = good;
        bad_start[0] = 0x5A;
        assert_eq!(Packet::decode(&bad_start), Err(PacketError::BadStart));
        let mut bad_sum = good;
        bad_sum[5] ^= 0x01;
        assert_eq!(Packet::decode(&bad_sum), Err(PacketError::BadChecksum));
        assert_eq!(
            Packet::decode(&frame(0x7F, 0, 0, 50)),
            Err(PacketError::UnknownAction)
        );
        assert_eq!(Packet::decode(&frame(3, 3, 0, 50)), Err(PacketError::BadZone));
        assert_eq!(
            Packet::decode(&frame(3, 0, 3, 50)),
            Err(PacketError::BadDistance)
        );
        assert_eq!(
            Packet::decode(&frame(3, 0, 0, 101)),
            Err(PacketError::BadConfidence)
        );
    }

    #[test]
    fn parser_skips_garbage_and_resumes_after_errors() {
        let good = wave_packet().encode();
        let mut bad = good;
        bad[5] ^= 0xFF;
        let mut parser = PacketParser::new();
        let mut out = Vec::new();
        for b in [0x00u8, 0x12].iter().chain(bad.iter()).chain(good.iter()) {
            match parser.feed(*b) {
                Parsed::Nothing => {}
                other => out.push(other),
            }
        }
        assert_eq!(
            out,
            vec![
                Parsed::Error(PacketError::BadChecksum),
                Parsed::Packet(wave_packet())
            ]
        );
    }

    #[test]
    fn unsupported_actions_have_no_packet() {
        let event = Event {
            time_s: 0.0,
            action: Action::Moving,
            zone: None,
            distance: None,
            facing_camera: false,
            confidence: 1.0,
        };
        assert!(Packet::from_event(&event).is_none());
    }
}
