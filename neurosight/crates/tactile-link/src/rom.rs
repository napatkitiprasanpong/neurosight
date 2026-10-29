//! Event -> tactile pattern: the content of the FPGA pattern ROM.
//!
//! The patterns come straight from `neurosight_encoder::encode`, so the encoder stays the single
//! source of truth. Two conversions happen here:
//!
//! * **Motor index.** The encoder produces *coil* positions for the TMS simulation, which are
//!   mirrored relative to the visual field. For vibration motors on the forehead we want the
//!   visual-field cell instead (col 0 = wearer's left, row 0 = top), so the coil position is
//!   converted back with `retinotopy::phosphene_for_coil`. Motor index = `row * 5 + col`.
//! * **Duty.** Relative intensity (0 ..= max_intensity) -> 8-bit PWM duty (0 ..= 255).
//!   The RTL never computes this: it just reads the already-quantized ROM.

use neurosight_common::{retinotopy, CoilPos, GRID};
use neurosight_encoder::{encode, SafetyLimits};

use crate::wire::{
    action_id, distance_code, zone_code, Key, Packet, DISTANCES, EVENTS, ZONES,
};

/// ROM slots reserved per pattern (the safety limit is 8 pulses).
pub const MAX_PULSES: usize = 8;
/// Number of keys: events x zones x distances.
pub const KEY_COUNT: usize = EVENTS.len() * 3 * 3;
/// ROM words per key: one header word plus `MAX_PULSES` pulse words.
pub const WORDS_PER_KEY: usize = 1 + MAX_PULSES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TactilePulse {
    pub offset_ms: u16,
    pub motor: u8,
    pub duty: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TactilePattern {
    pub pulses: Vec<TactilePulse>,
}

impl TactilePattern {
    /// Offset of the last pulse (the scheduler's "duration").
    pub fn duration_ms(&self) -> u16 {
        self.pulses.iter().map(|p| p.offset_ms).max().unwrap_or(0)
    }
}

/// Motor (0 ..= 24) for a coil position, via the visual-field cell.
pub fn motor_index(coil: CoilPos) -> u8 {
    let (col, row) = retinotopy::phosphene_for_coil(coil);
    row * GRID + col
}

/// Relative intensity -> 8-bit duty. `max_intensity` maps to 255.
pub fn duty_for_intensity(intensity: f32, max_intensity: f32) -> u8 {
    let scaled = intensity.clamp(0.0, max_intensity) / max_intensity * 255.0;
    scaled.round() as u8
}

/// The tactile pattern for a key. Supported actions always give a pattern.
pub fn pattern_for(key: &Key, limits: &SafetyLimits) -> TactilePattern {
    let event = Packet {
        key: *key,
        confidence: 100,
    }
    .to_event();
    let pattern = encode(&event, limits).expect("supported actions always produce a pattern");
    let pulses = pattern
        .pulses
        .iter()
        .map(|p| TactilePulse {
            offset_ms: p.offset_ms as u16,
            motor: motor_index(p.coil),
            duty: duty_for_intensity(p.intensity, limits.max_intensity),
        })
        .collect();
    TactilePattern { pulses }
}

/// Every key in ROM order (action id ascending, then zone, then distance).
pub fn all_keys() -> Vec<Key> {
    let mut keys = Vec::with_capacity(KEY_COUNT);
    for (_, action, _) in EVENTS.iter() {
        for zone in ZONES.iter() {
            for distance in DISTANCES.iter() {
                keys.push(Key {
                    action: *action,
                    zone: *zone,
                    distance: *distance,
                });
            }
        }
    }
    keys
}

/// Slot number of a key; ROM word address = `slot * WORDS_PER_KEY`.
pub fn rom_slot(key: &Key) -> usize {
    let id = action_id(key.action).expect("supported action") as usize;
    ((id - 1) * 3 + zone_code(key.zone) as usize) * 3 + distance_code(key.distance) as usize
}

/// ROM words for one key: `[header, pulse0 .. pulse7]`.
/// header: bits 31:16 duration_ms, bits 7:0 pulse count.
/// pulse:  bits 31:16 offset_ms, bits 15:8 motor, bits 7:0 duty.
pub fn rom_words(pattern: &TactilePattern) -> [u32; WORDS_PER_KEY] {
    let mut words = [0u32; WORDS_PER_KEY];
    words[0] = ((pattern.duration_ms() as u32) << 16) | pattern.pulses.len() as u32;
    for (i, p) in pattern.pulses.iter().take(MAX_PULSES).enumerate() {
        words[1 + i] = ((p.offset_ms as u32) << 16) | ((p.motor as u32) << 8) | p.duty as u32;
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use neurosight_common::{Action, Distance, Zone};

    #[test]
    fn motor_index_is_visual_field_cell_not_mirrored() {
        // Phosphene cell (col 1, row 3) -> encoder coil (3, 1) -> motor 3 * 5 + 1 = 16.
        let coil = retinotopy::coil_for_phosphene(1, 3);
        assert_eq!(motor_index(coil), 3 * 5 + 1);
        // Top-left of the visual field is motor 0.
        assert_eq!(motor_index(retinotopy::coil_for_phosphene(0, 0)), 0);
    }

    #[test]
    fn wave_on_the_left_sweeps_left_then_right_of_the_persons_column() {
        let limits = SafetyLimits::default();
        let key = Key {
            action: Action::Waving,
            zone: Zone::Left,
            distance: Distance::Medium,
        };
        let pattern = pattern_for(&key, &limits);
        assert_eq!(pattern.pulses.len(), 6);
        // Zone left -> column 1; the wave alternates columns 0 and 2 on row 3.
        assert_eq!(pattern.pulses[0].motor, 3 * 5 + 0);
        assert_eq!(pattern.pulses[1].motor, 3 * 5 + 2);
        assert_eq!(pattern.duration_ms(), 1000);
    }

    #[test]
    fn every_rom_pattern_obeys_the_safety_limits() {
        let limits = SafetyLimits::default();
        for key in all_keys() {
            let p = pattern_for(&key, &limits);
            assert!(!p.pulses.is_empty() && p.pulses.len() <= MAX_PULSES);
            for pair in p.pulses.windows(2) {
                assert!(
                    pair[1].offset_ms >= pair[0].offset_ms + limits.min_pulse_gap_ms as u16,
                    "pulses too close for {:?}",
                    key
                );
            }
            for pulse in &p.pulses {
                assert!(pulse.motor < GRID * GRID);
            }
        }
    }

    #[test]
    fn rom_slots_are_unique_and_dense() {
        let mut seen = vec![false; KEY_COUNT];
        for key in all_keys() {
            let slot = rom_slot(&key);
            assert!(!seen[slot], "duplicate slot");
            seen[slot] = true;
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn duty_scales_to_full_range() {
        assert_eq!(duty_for_intensity(1.5, 1.5), 255);
        assert_eq!(duty_for_intensity(0.0, 1.5), 0);
        assert_eq!(duty_for_intensity(9.0, 1.5), 255);
    }
}
