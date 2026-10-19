//! Shared types for NeuroSight.
//!
//! Every stage of the pipeline talks to the next one only through the types
//! in this crate, so any stage can be replaced without touching the others.

use std::fmt;

// ---------------------------------------------------------------- images

/// A grayscale image: row-major, one byte per pixel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrayFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl GrayFrame {
    /// Create a frame from raw pixel data. Panics if the length is wrong.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "frame data length must equal width * height"
        );
        Self { width, height, data }
    }

    /// A frame where every pixel has the same value.
    pub fn filled(width: usize, height: usize, value: u8) -> Self {
        Self {
            width,
            height,
            data: vec![value; width * height],
        }
    }

    /// Convert packed RGB8 pixels (R,G,B,R,G,B,...) to grayscale and shrink the
    /// image with a simple box filter. Averaging many source pixels also
    /// reduces camera noise.
    pub fn from_rgb_downscaled(
        rgb: &[u8],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Self {
        assert!(src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0);
        assert!(rgb.len() >= src_w * src_h * 3, "rgb buffer too small");

        let mut data = Vec::with_capacity(dst_w * dst_h);
        for dy in 0..dst_h {
            let y0 = dy * src_h / dst_h;
            let y1 = ((dy + 1) * src_h / dst_h).max(y0 + 1).min(src_h);
            for dx in 0..dst_w {
                let x0 = dx * src_w / dst_w;
                let x1 = ((dx + 1) * src_w / dst_w).max(x0 + 1).min(src_w);

                let mut sum: u32 = 0;
                let mut count: u32 = 0;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let i = (y * src_w + x) * 3;
                        let r = rgb[i] as u32;
                        let g = rgb[i + 1] as u32;
                        let b = rgb[i + 2] as u32;
                        sum += (299 * r + 587 * g + 114 * b) / 1000;
                        count += 1;
                    }
                }
                data.push((sum / count.max(1)) as u8);
            }
        }
        Self::new(dst_w, dst_h, data)
    }
}

/// A color image: packed RGB8 (R,G,B,R,G,B,...), row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RgbFrame {
    /// Create a frame from raw pixel data. Panics if the length is wrong.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            width * height * 3,
            "frame data length must equal width * height * 3"
        );
        Self { width, height, data }
    }

    /// Resize packed RGB8 pixels to `dst_w` x `dst_h` with a box filter.
    /// (The aspect ratio is not preserved: the image is squeezed to fit.)
    pub fn from_rgb_downscaled(
        rgb: &[u8],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Self {
        assert!(src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0);
        assert!(rgb.len() >= src_w * src_h * 3, "rgb buffer too small");

        let mut data = Vec::with_capacity(dst_w * dst_h * 3);
        for dy in 0..dst_h {
            let y0 = dy * src_h / dst_h;
            let y1 = ((dy + 1) * src_h / dst_h).max(y0 + 1).min(src_h);
            for dx in 0..dst_w {
                let x0 = dx * src_w / dst_w;
                let x1 = ((dx + 1) * src_w / dst_w).max(x0 + 1).min(src_w);

                let mut sums = [0u32; 3];
                let mut count: u32 = 0;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let i = (y * src_w + x) * 3;
                        sums[0] += rgb[i] as u32;
                        sums[1] += rgb[i + 1] as u32;
                        sums[2] += rgb[i + 2] as u32;
                        count += 1;
                    }
                }
                let n = count.max(1);
                data.push((sums[0] / n) as u8);
                data.push((sums[1] / n) as u8);
                data.push((sums[2] / n) as u8);
            }
        }
        Self::new(dst_w, dst_h, data)
    }
}

// ------------------------------------------------------------------ pose

/// Number of body keypoints (COCO / MoveNet layout).
pub const KEYPOINT_COUNT: usize = 17;

/// Indices into `Pose::keypoints`. "Left" and "right" are the person's own
/// left and right.
pub mod kp {
    pub const NOSE: usize = 0;
    pub const LEFT_EYE: usize = 1;
    pub const RIGHT_EYE: usize = 2;
    pub const LEFT_EAR: usize = 3;
    pub const RIGHT_EAR: usize = 4;
    pub const LEFT_SHOULDER: usize = 5;
    pub const RIGHT_SHOULDER: usize = 6;
    pub const LEFT_ELBOW: usize = 7;
    pub const RIGHT_ELBOW: usize = 8;
    pub const LEFT_WRIST: usize = 9;
    pub const RIGHT_WRIST: usize = 10;
    pub const LEFT_HIP: usize = 11;
    pub const RIGHT_HIP: usize = 12;
    pub const LEFT_KNEE: usize = 13;
    pub const RIGHT_KNEE: usize = 14;
    pub const LEFT_ANKLE: usize = 15;
    pub const RIGHT_ANKLE: usize = 16;
}

/// One body point. `x` and `y` are 0.0 (left / top) to 1.0 (right / bottom)
/// of the camera image; `score` is the model's confidence, 0.0 to 1.0.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

/// The estimated body pose of one person.
#[derive(Clone, Debug, PartialEq)]
pub struct Pose {
    pub keypoints: [Keypoint; KEYPOINT_COUNT],
}

impl Pose {
    pub fn new(keypoints: [Keypoint; KEYPOINT_COUNT]) -> Self {
        Self { keypoints }
    }
}

// ---------------------------------------------------------------- events

/// Horizontal region of the camera image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zone {
    Left,
    Center,
    Right,
}

impl Zone {
    pub fn from_x(x: f32) -> Zone {
        if x < 1.0 / 3.0 {
            Zone::Left
        } else if x > 2.0 / 3.0 {
            Zone::Right
        } else {
            Zone::Center
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Zone::Left => "left",
            Zone::Center => "center",
            Zone::Right => "right",
        }
    }
}

/// Rough distance of a person from the camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distance {
    Near,
    Medium,
    Far,
}

impl Distance {
    pub fn as_str(&self) -> &'static str {
        match self {
            Distance::Near => "close by",
            Distance::Medium => "at a medium distance",
            Distance::Far => "far away",
        }
    }
}

/// What the system thinks is happening.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    // Motion-only mode
    Still,
    Moving,
    // Pose mode
    Appeared,
    Present,
    Left,
    ReachingOut,
    LookedLeft,
    LookedRight,
    LookedUp,
    LookedDown,
    Nodding,
    ShakingHead,
    // Both modes
    Waving,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Action::Still => "still",
            Action::Moving => "moving",
            Action::Appeared => "appeared",
            Action::Present => "present",
            Action::Left => "left",
            Action::ReachingOut => "reaching out",
            Action::LookedLeft => "looked left",
            Action::LookedRight => "looked right",
            Action::LookedUp => "looked up",
            Action::LookedDown => "looked down",
            Action::Nodding => "nodding",
            Action::ShakingHead => "shaking head",
            Action::Waving => "waving",
        };
        write!(f, "{}", s)
    }
}

/// A change in what is happening, ready to be turned into text (and later,
/// into a stimulation pattern).
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// Seconds since the program started.
    pub time_s: f64,
    pub action: Action,
    /// Where in the image it happens (None when there is nobody / nothing).
    pub zone: Option<Zone>,
    /// Rough distance (pose mode only).
    pub distance: Option<Distance>,
    /// True if the person seems to face the camera (pose mode only).
    pub facing_camera: bool,
    /// 0.0 to 1.0.
    pub confidence: f32,
}

impl Event {
    /// Plain-English description of the event.
    pub fn to_text(&self) -> String {
        let place = match self.zone {
            Some(z) => format!(" on the {} of the image", z.as_str()),
            None => String::new(),
        };
        let dist = match self.distance {
            Some(d) => format!(", {}", d.as_str()),
            None => String::new(),
        };
        let pct = (self.confidence * 100.0).round() as i32;
        match self.action {
            Action::Still => "Nothing is moving.".to_string(),
            Action::Moving => {
                format!("Someone or something is moving{} (confidence {}%).", place, pct)
            }
            Action::Appeared => format!("Someone appeared{}{}.", place, dist),
            Action::Present => format!("Someone is in view{}{}.", place, dist),
            Action::Left => "The person left.".to_string(),
            Action::ReachingOut => format!(
                "Someone is reaching out a hand (handshake offer){}{}.",
                place, dist
            ),
            Action::LookedLeft => {
                "Someone turned their head toward the left of the image.".to_string()
            }
            Action::LookedRight => {
                "Someone turned their head toward the right of the image.".to_string()
            }
            Action::LookedUp => "Someone is looking up.".to_string(),
            Action::LookedDown => "Someone is looking down.".to_string(),
            Action::Nodding => "Someone is nodding (yes).".to_string(),
            Action::ShakingHead => "Someone is shaking their head (no).".to_string(),
            Action::Waving => {
                let target = if self.facing_camera { " at you" } else { "" };
                format!(
                    "Someone is waving{}{}{} (confidence {}%).",
                    target, place, dist, pct
                )
            }
        }
    }
}

// ---------------------------------------------- stimulation (simulation only)
//
// These types describe an ABSTRACT stimulation pattern for a SIMULATION. They have no
// physical units (no tesla, joules or amperes) and nothing in this project sends them to a
// device. Real brain stimulation needs medical screening and clinical supervision.

/// Side length of the virtual coil grid and of the phosphene grid (GRID x GRID cells).
pub const GRID: u8 = 5;

/// Position of a virtual coil over the back of the head, seen from behind.
/// `col` 0 = the person's left side ... `col` 4 = the person's right side.
/// `row` 0 = highest on the head ... `row` 4 = lowest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoilPos {
    pub col: u8,
    pub row: u8,
}

/// One magnetic pulse of a pattern.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pulse {
    /// Milliseconds after the start of the pattern.
    pub offset_ms: u32,
    pub coil: CoilPos,
    /// Relative strength: 1.0 = the simulated phosphene threshold.
    pub intensity: f32,
}

/// A short sequence of pulses that stands for one event.
#[derive(Clone, Debug, PartialEq)]
pub struct Pattern {
    pub name: &'static str,
    pub action: Action,
    pub pulses: Vec<Pulse>,
}

impl Pattern {
    /// Time of the last pulse, in milliseconds.
    pub fn duration_ms(&self) -> u32 {
        self.pulses.iter().map(|p| p.offset_ms).max().unwrap_or(0)
    }
}

/// How the virtual coil grid relates to the visual field (very simplified).
///
/// Each half of the visual cortex handles the opposite half of the visual field, and the
/// upper visual field is handled by the lower part of the visual cortex. So a phosphene in
/// the LEFT visual field is made by a coil on the RIGHT side of the head, and a phosphene
/// in the UPPER visual field by a coil LOW on the head.
pub mod retinotopy {
    use super::{CoilPos, GRID};

    /// Phosphene cell (col 0 = far left of the visual field, row 0 = top) -> coil position.
    pub fn coil_for_phosphene(col: u8, row: u8) -> CoilPos {
        CoilPos {
            col: GRID - 1 - col.min(GRID - 1),
            row: GRID - 1 - row.min(GRID - 1),
        }
    }

    /// Coil position -> phosphene cell (col, row) in the visual field.
    pub fn phosphene_for_coil(coil: CoilPos) -> (u8, u8) {
        (
            GRID - 1 - coil.col.min(GRID - 1),
            GRID - 1 - coil.row.min(GRID - 1),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zone_from_x_splits_into_thirds() {
        assert_eq!(Zone::from_x(0.1), Zone::Left);
        assert_eq!(Zone::from_x(0.5), Zone::Center);
        assert_eq!(Zone::from_x(0.9), Zone::Right);
    }

    #[test]
    fn gray_downscale_keeps_uniform_color() {
        let rgb = vec![100u8; 4 * 4 * 3];
        let f = GrayFrame::from_rgb_downscaled(&rgb, 4, 4, 2, 2);
        assert_eq!(f.width, 2);
        assert_eq!(f.height, 2);
        assert!(f.data.iter().all(|&p| p == 100));
    }

    #[test]
    fn gray_downscale_averages_blocks() {
        // 2x1 image: one black pixel, one white pixel -> 1x1 mid gray
        let rgb = vec![0, 0, 0, 255, 255, 255];
        let f = GrayFrame::from_rgb_downscaled(&rgb, 2, 1, 1, 1);
        assert_eq!(f.data, vec![127]);
    }

    #[test]
    fn rgb_downscale_keeps_channels_separate() {
        // 2x1 image: pure red and pure blue -> 1x1 (127, 0, 127)
        let rgb = vec![255, 0, 0, 0, 0, 255];
        let f = RgbFrame::from_rgb_downscaled(&rgb, 2, 1, 1, 1);
        assert_eq!(f.data, vec![127, 0, 127]);
    }

    #[test]
    fn rgb_downscale_output_size() {
        let rgb = vec![10u8; 640 * 480 * 3];
        let f = RgbFrame::from_rgb_downscaled(&rgb, 640, 480, 192, 192);
        assert_eq!(f.width, 192);
        assert_eq!(f.height, 192);
        assert_eq!(f.data.len(), 192 * 192 * 3);
    }

    #[test]
    fn waving_text_mentions_you_when_facing_camera() {
        let e = Event {
            time_s: 1.0,
            action: Action::Waving,
            zone: Some(Zone::Left),
            distance: Some(Distance::Medium),
            facing_camera: true,
            confidence: 0.9,
        };
        let text = e.to_text();
        assert!(text.contains("waving at you"));
        assert!(text.contains("left"));
        assert!(text.contains("medium"));
    }

    #[test]
    fn head_gesture_texts() {
        let mk = |action| Event {
            time_s: 0.0,
            action,
            zone: None,
            distance: None,
            facing_camera: true,
            confidence: 0.8,
        };
        assert!(mk(Action::Nodding).to_text().contains("nodding"));
        assert!(mk(Action::ShakingHead).to_text().contains("shaking"));
        assert!(mk(Action::LookedUp).to_text().contains("up"));
        assert!(mk(Action::LookedDown).to_text().contains("down"));
        assert!(mk(Action::LookedLeft).to_text().contains("left"));
        assert!(mk(Action::LookedRight).to_text().contains("right"));
        assert!(mk(Action::ReachingOut).to_text().contains("handshake"));
    }

    #[test]
    fn retinotopy_is_crossed_and_flipped() {
        // Far left of the visual field -> coil on the right side of the head.
        let c = retinotopy::coil_for_phosphene(0, 2);
        assert_eq!(c.col, GRID - 1);
        // Top of the visual field -> coil low on the head.
        let c = retinotopy::coil_for_phosphene(2, 0);
        assert_eq!(c.row, GRID - 1);
        // And back again.
        for col in 0..GRID {
            for row in 0..GRID {
                let coil = retinotopy::coil_for_phosphene(col, row);
                assert_eq!(retinotopy::phosphene_for_coil(coil), (col, row));
            }
        }
    }

    #[test]
    fn pattern_duration_is_last_pulse_time() {
        let pulse = |ms| Pulse {
            offset_ms: ms,
            coil: CoilPos { col: 0, row: 0 },
            intensity: 1.0,
        };
        let p = Pattern {
            name: "t",
            action: Action::Waving,
            pulses: vec![pulse(0), pulse(200), pulse(400)],
        };
        assert_eq!(p.duration_ms(), 400);
    }

    #[test]
    fn waving_text_without_facing_has_no_you() {
        let e = Event {
            time_s: 1.0,
            action: Action::Waving,
            zone: Some(Zone::Right),
            distance: None,
            facing_camera: false,
            confidence: 0.8,
        };
        assert!(!e.to_text().contains("at you"));
    }
}
