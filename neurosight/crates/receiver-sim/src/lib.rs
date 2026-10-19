//! Simulated receiver (SIMULATION ONLY).
//!
//! Takes the pulses of a stimulation pattern and works out what the person would
//! "see": a short flash of light at some spot of the visual field, brighter for stronger
//! pulses. This is a deliberately simple model:
//!
//! * where: the coil position maps to a visual-field cell through the crossed and flipped
//!   retinotopy table in `neurosight-common`;
//! * how bright: an S-shaped curve of the relative strength. At the simulated phosphene
//!   threshold (1.0) a flash is seen about half the time (brightness 50%), clearly above it
//!   (1.3) it is almost fully bright, clearly below it (0.8) it is barely visible.
//!
//! Nothing here talks to a device.

use neurosight_common::{retinotopy, Pattern, Pulse, GRID};

/// Brightness model.
#[derive(Clone, Debug)]
pub struct PhospheneModel {
    /// Relative strength at which the brightness is 50%.
    pub threshold: f32,
    /// Smaller = sharper step around the threshold.
    pub slope: f32,
    /// Degrees of visual field between neighbouring grid cells.
    pub degrees_per_cell: f32,
}

impl Default for PhospheneModel {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            slope: 0.08,
            degrees_per_cell: 5.0,
        }
    }
}

/// One flash of light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Phosphene {
    /// Visual-field cell: col 0 = far left, row 0 = top.
    pub col: u8,
    pub row: u8,
    /// Degrees from the center of the view: positive = right.
    pub azimuth_deg: f32,
    /// Degrees from the center of the view: positive = up.
    pub elevation_deg: f32,
    /// 0.0 (invisible) to 1.0 (fully bright).
    pub brightness: f32,
}

impl PhospheneModel {
    pub fn percept(&self, pulse: &Pulse) -> Phosphene {
        let (col, row) = retinotopy::phosphene_for_coil(pulse.coil);
        let center = (GRID as f32 - 1.0) / 2.0;
        let brightness =
            1.0 / (1.0 + (-(pulse.intensity - self.threshold) / self.slope).exp());
        Phosphene {
            col,
            row,
            azimuth_deg: (col as f32 - center) * self.degrees_per_cell,
            elevation_deg: (center - row as f32) * self.degrees_per_cell,
            brightness,
        }
    }
}

impl Phosphene {
    /// Short plain-English description, e.g. "flash 5 deg up, 10 deg left, brightness 87%".
    pub fn describe(&self) -> String {
        let vertical = if self.elevation_deg > 0.0 {
            format!("{:.0} deg up", self.elevation_deg)
        } else if self.elevation_deg < 0.0 {
            format!("{:.0} deg down", -self.elevation_deg)
        } else {
            "level".to_string()
        };
        let horizontal = if self.azimuth_deg > 0.0 {
            format!("{:.0} deg right", self.azimuth_deg)
        } else if self.azimuth_deg < 0.0 {
            format!("{:.0} deg left", -self.azimuth_deg)
        } else {
            "centered".to_string()
        };
        format!(
            "flash {}, {}, brightness {:.0}%",
            vertical,
            horizontal,
            self.brightness * 100.0
        )
    }

    /// The visual field as five short rows joined with "/", the flash shown by a letter
    /// from faint (`.`) to bright (`@`), for example `...../...../..@../...../.....`.
    pub fn grid_inline(&self) -> String {
        let glyph = if self.brightness < 0.2 {
            '.'
        } else if self.brightness < 0.5 {
            'o'
        } else if self.brightness < 0.8 {
            'O'
        } else {
            '@'
        };
        let mut rows: Vec<String> = Vec::new();
        for r in 0..GRID {
            let row: String = (0..GRID)
                .map(|c| {
                    if c == self.col && r == self.row {
                        if glyph == '.' {
                            ':'
                        } else {
                            glyph
                        }
                    } else {
                        '.'
                    }
                })
                .collect();
            rows.push(row);
        }
        rows.join("/")
    }
}

/// Plays a pattern back in (simulated) real time: call `start` when a pattern begins, then
/// `due` on every frame to get the pulses whose time has come.
#[derive(Default)]
pub struct Player {
    /// (absolute time in seconds, pulse number starting at 1, total pulses, pulse)
    pending: Vec<(f64, usize, usize, Pulse)>,
}

impl Player {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(&mut self, pattern: &Pattern, now_s: f64) {
        let total = pattern.pulses.len();
        for (i, pulse) in pattern.pulses.iter().enumerate() {
            self.pending
                .push((now_s + pulse.offset_ms as f64 / 1000.0, i + 1, total, *pulse));
        }
    }

    /// Pulses due at `now_s` (removed from the player): (pulse number, total, pulse).
    pub fn due(&mut self, now_s: f64) -> Vec<(usize, usize, Pulse)> {
        let mut out = Vec::new();
        let mut rest = Vec::new();
        for item in self.pending.drain(..) {
            if item.0 <= now_s {
                out.push((item.1, item.2, item.3));
            } else {
                rest.push(item);
            }
        }
        self.pending = rest;
        out
    }

    pub fn is_idle(&self) -> bool {
        self.pending.is_empty()
    }
}

/// One line of JSON describing a pattern that started at `t_s` seconds (for log files).
pub fn pattern_json(pattern: &Pattern, t_s: f64) -> String {
    let pulses: Vec<String> = pattern
        .pulses
        .iter()
        .map(|p| {
            format!(
                "{{\"ms\":{},\"coil_col\":{},\"coil_row\":{},\"intensity\":{:.3}}}",
                p.offset_ms, p.coil.col, p.coil.row, p.intensity
            )
        })
        .collect();
    format!(
        "{{\"t\":{:.3},\"action\":\"{}\",\"pattern\":\"{}\",\"pulses\":[{}]}}",
        t_s,
        pattern.action,
        pattern.name,
        pulses.join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use neurosight_common::{Action, CoilPos};

    fn pulse_at(col: u8, row: u8, intensity: f32, offset_ms: u32) -> Pulse {
        Pulse {
            offset_ms,
            coil: CoilPos { col, row },
            intensity,
        }
    }

    #[test]
    fn brightness_grows_with_strength_and_is_half_at_threshold() {
        let m = PhospheneModel::default();
        let at = |i: f32| m.percept(&pulse_at(2, 2, i, 0)).brightness;
        assert!((at(1.0) - 0.5).abs() < 1e-3);
        assert!(at(0.8) < 0.1);
        assert!(at(1.3) > 0.95);
        assert!(at(0.9) < at(1.0) && at(1.0) < at(1.1));
    }

    #[test]
    fn coil_on_the_right_gives_a_flash_in_the_left_visual_field() {
        let m = PhospheneModel::default();
        let right_side = m.percept(&pulse_at(4, 2, 1.2, 0));
        assert!(right_side.azimuth_deg < 0.0);
        let left_side = m.percept(&pulse_at(0, 2, 1.2, 0));
        assert!(left_side.azimuth_deg > 0.0);
    }

    #[test]
    fn low_coil_gives_a_flash_in_the_upper_visual_field() {
        let m = PhospheneModel::default();
        let low = m.percept(&pulse_at(2, 4, 1.2, 0));
        assert!(low.elevation_deg > 0.0);
        let high = m.percept(&pulse_at(2, 0, 1.2, 0));
        assert!(high.elevation_deg < 0.0);
    }

    #[test]
    fn center_cell_is_centered() {
        let m = PhospheneModel::default();
        let c = m.percept(&pulse_at(2, 2, 1.2, 0));
        assert_eq!(c.azimuth_deg, 0.0);
        assert_eq!(c.elevation_deg, 0.0);
        assert!(c.describe().contains("centered"));
    }

    #[test]
    fn grid_inline_marks_the_flash() {
        let m = PhospheneModel::default();
        let c = m.percept(&pulse_at(2, 2, 1.3, 0));
        assert_eq!(c.grid_inline(), "...../...../..@../...../.....");
    }

    #[test]
    fn player_releases_pulses_on_time() {
        let pattern = Pattern {
            name: "t",
            action: Action::Waving,
            pulses: vec![
                pulse_at(0, 0, 1.2, 0),
                pulse_at(1, 0, 1.2, 200),
                pulse_at(2, 0, 1.2, 400),
            ],
        };
        let mut player = Player::new();
        player.start(&pattern, 10.0);
        assert_eq!(player.due(9.9).len(), 0);
        let first = player.due(10.0);
        assert_eq!(first.len(), 1);
        assert_eq!((first[0].0, first[0].1), (1, 3));
        assert_eq!(player.due(10.25).len(), 1);
        assert!(!player.is_idle());
        assert_eq!(player.due(11.0).len(), 1);
        assert!(player.is_idle());
    }

    #[test]
    fn json_line_contains_the_essentials() {
        let pattern = Pattern {
            name: "wave",
            action: Action::Waving,
            pulses: vec![pulse_at(1, 3, 1.15, 0), pulse_at(3, 3, 1.15, 200)],
        };
        let line = pattern_json(&pattern, 1.5);
        assert!(line.starts_with('{') && line.ends_with('}'));
        assert!(line.contains("\"action\":\"waving\""));
        assert!(line.contains("\"pattern\":\"wave\""));
        assert!(line.contains("\"ms\":200"));
    }
}
