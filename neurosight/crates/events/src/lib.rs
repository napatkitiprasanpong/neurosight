//! Event engine: turns camera data into events.
//!
//! Two analyzers produce the same `Event` type:
//!
//! * `MotionAnalyzer` (this file): grayscale frames in, still / moving / waving out.
//!   No machine learning, works without any model file.
//! * `PoseAnalyzer` (`pose.rs`): body keypoints in, appeared / waving / present / left
//!   out, with side, rough distance and whether the person faces the camera.
//!
//! `MotionAnalyzer` uses simple motion analysis:
//!
//! 1. Compare each frame with the previous one to find the pixels that changed.
//! 2. Track the horizontal center of that motion over the last ~1.5 seconds.
//! 3. If the center swings left and right several times at a hand-wave
//!    frequency (about 1-6 Hz), call it waving.
//!
//! Later versions replace step 1-3 with body-pose models, but they will keep
//! producing the same `Event` type.

use std::collections::VecDeque;

use neurosight_common::{Action, Event, GrayFrame, Zone};

mod head;
mod pose;
mod reach;
pub use head::{HeadAnalyzer, HeadConfig, HeadDebug, HeadGesture};
pub use pose::{PoseAnalyzer, PoseConfig, PoseDebug};
pub use reach::{ReachConfig, ReachDetector};

/// Tunable thresholds. The defaults are a starting point: expect to adjust
/// them for your camera and lighting (run the app with `--debug`).
#[derive(Clone, Debug)]
pub struct Config {
    /// A pixel counts as "changed" if it differs by more than this (0-255).
    pub pixel_threshold: u8,
    /// Fraction of pixels that must change for the frame to count as motion.
    pub min_motion_ratio: f32,
    /// If more than this fraction changes, treat it as a lighting/exposure
    /// change (not a person) and ignore the frame.
    pub max_motion_ratio: f32,
    /// How many seconds of motion history to analyze.
    pub window_s: f64,
    /// Direction changes needed within the window to call it a wave.
    pub min_reversals: usize,
    /// Minimum sideways swing (fraction of image width) for a direction change.
    pub min_swing: f32,
    /// Allowed wave frequency range in Hz.
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    /// Keep the current state this long after the evidence disappears.
    pub hold_s: f64,
    /// Minimum seconds between "same action, different side" announcements.
    pub zone_cooldown_s: f64,
    /// Waving is only evaluated if there was motion this recently.
    pub recent_s: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            pixel_threshold: 25,
            min_motion_ratio: 0.008,
            max_motion_ratio: 0.60,
            window_s: 1.5,
            min_reversals: 4,
            min_swing: 0.03,
            min_freq_hz: 1.0,
            max_freq_hz: 6.0,
            hold_s: 0.8,
            zone_cooldown_s: 1.5,
            recent_s: 0.3,
        }
    }
}

/// Numbers from the latest frame, useful for tuning.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub motion_ratio: f32,
    pub centroid_x: Option<f32>,
    pub reversals: usize,
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    t: f64,
    cx: f32,
}

pub struct MotionAnalyzer {
    cfg: Config,
    prev: Option<GrayFrame>,
    samples: VecDeque<Sample>,
    state: Action,
    last_zone: Option<Zone>,
    last_cx: f32,
    last_motion_t: f64,
    last_wave_t: f64,
    last_emit_t: f64,
    stats: FrameStats,
}

impl MotionAnalyzer {
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg,
            prev: None,
            samples: VecDeque::new(),
            state: Action::Still,
            last_zone: None,
            last_cx: 0.5,
            last_motion_t: f64::NEG_INFINITY,
            last_wave_t: f64::NEG_INFINITY,
            last_emit_t: f64::NEG_INFINITY,
            stats: FrameStats::default(),
        }
    }

    pub fn state(&self) -> Action {
        self.state
    }

    pub fn stats(&self) -> FrameStats {
        self.stats
    }

    /// Feed one frame (with its timestamp in seconds). Returns an event only
    /// when something worth announcing changes.
    pub fn process(&mut self, frame: &GrayFrame, t: f64) -> Option<Event> {
        // Need a previous frame of the same size to measure motion.
        let prev = match self.prev.replace(frame.clone()) {
            Some(p) if p.width == frame.width && p.height == frame.height => p,
            _ => {
                self.samples.clear();
                return None;
            }
        };

        let (ratio, cx) = frame_motion(&prev, frame, self.cfg.pixel_threshold);
        let moving = ratio >= self.cfg.min_motion_ratio && ratio <= self.cfg.max_motion_ratio;

        if moving {
            if let Some(cx) = cx {
                self.samples.push_back(Sample { t, cx });
                self.last_cx = cx;
            }
        }
        let cutoff = t - self.cfg.window_s;
        while self.samples.front().map_or(false, |s| s.t < cutoff) {
            self.samples.pop_front();
        }

        // Is the motion a wave?
        let recent = self
            .samples
            .back()
            .map_or(false, |s| s.t >= t - self.cfg.recent_s);
        let mut waving = false;
        let mut reversals = 0usize;
        if recent && self.samples.len() >= 6 {
            let first_t = self.samples.front().map(|s| s.t).unwrap_or(t);
            let last_t = self.samples.back().map(|s| s.t).unwrap_or(t);
            let span = last_t - first_t;
            if span >= 0.6 {
                let xs: Vec<f32> = self.samples.iter().map(|s| s.cx).collect();
                let xs = smooth3(&xs);
                reversals = count_reversals(&xs, self.cfg.min_swing);
                let freq = reversals as f64 / 2.0 / span;
                waving = reversals >= self.cfg.min_reversals
                    && freq >= self.cfg.min_freq_hz
                    && freq <= self.cfg.max_freq_hz;
            }
        }
        self.stats = FrameStats {
            motion_ratio: ratio,
            centroid_x: cx,
            reversals,
        };

        // Pick the action, with a little hysteresis so it does not flicker.
        let mut action = if waving {
            Action::Waving
        } else if moving {
            Action::Moving
        } else {
            Action::Still
        };
        if waving {
            self.last_wave_t = t;
        }
        if moving {
            self.last_motion_t = t;
        }
        if action != Action::Waving
            && self.state == Action::Waving
            && t - self.last_wave_t < self.cfg.hold_s
        {
            action = Action::Waving;
        }
        if action == Action::Still
            && self.state != Action::Still
            && t - self.last_motion_t < self.cfg.hold_s
        {
            action = Action::Moving;
        }

        // Where is it happening?
        let zone = if action == Action::Still {
            None
        } else {
            let mean_x = if self.samples.is_empty() {
                self.last_cx
            } else {
                self.samples.iter().map(|s| s.cx).sum::<f32>() / self.samples.len() as f32
            };
            Some(Zone::from_x(mean_x))
        };

        let action_changed = action != self.state;
        let zone_changed = action != Action::Still
            && zone != self.last_zone
            && t - self.last_emit_t >= self.cfg.zone_cooldown_s;

        if action_changed || zone_changed {
            self.state = action;
            self.last_zone = zone;
            self.last_emit_t = t;
            let confidence = match action {
                Action::Still => 1.0,
                Action::Moving => (ratio / 0.05).clamp(0.3, 1.0),
                Action::Waving => (reversals as f32 / 8.0).clamp(0.5, 1.0),
                // The motion analyzer only produces the three actions above.
                _ => 1.0,
            };
            return Some(Event {
                time_s: t,
                action,
                zone,
                distance: None,
                facing_camera: false,
                confidence,
            });
        }
        None
    }
}

/// Returns (fraction of pixels that changed, horizontal center of the change
/// as 0.0 = left edge .. 1.0 = right edge).
fn frame_motion(prev: &GrayFrame, cur: &GrayFrame, threshold: u8) -> (f32, Option<f32>) {
    let mut count: usize = 0;
    let mut sum_x: f64 = 0.0;
    for y in 0..cur.height {
        let row = y * cur.width;
        for x in 0..cur.width {
            let a = prev.data[row + x];
            let b = cur.data[row + x];
            if a.abs_diff(b) > threshold {
                count += 1;
                sum_x += x as f64;
            }
        }
    }
    let total = (cur.width * cur.height).max(1);
    let ratio = count as f32 / total as f32;
    let cx = if count > 0 {
        Some(((sum_x / count as f64) + 0.5) as f32 / cur.width.max(1) as f32)
    } else {
        None
    };
    (ratio, cx)
}

/// 3-point moving average (keeps the end points).
pub(crate) fn smooth3(xs: &[f32]) -> Vec<f32> {
    if xs.len() < 3 {
        return xs.to_vec();
    }
    let mut out = Vec::with_capacity(xs.len());
    out.push(xs[0]);
    for i in 1..xs.len() - 1 {
        out.push((xs[i - 1] + xs[i] + xs[i + 1]) / 3.0);
    }
    out.push(xs[xs.len() - 1]);
    out
}

/// Count how many times the signal changes direction by more than `min_swing`.
pub(crate) fn count_reversals(xs: &[f32], min_swing: f32) -> usize {
    if xs.len() < 3 {
        return 0;
    }
    let mut reversals = 0usize;
    let mut extreme = xs[0];
    let mut dir: i8 = 0; // 0 = unknown, 1 = rising, -1 = falling
    for &x in &xs[1..] {
        match dir {
            0 => {
                if x - extreme > min_swing {
                    dir = 1;
                    extreme = x;
                } else if extreme - x > min_swing {
                    dir = -1;
                    extreme = x;
                }
            }
            1 => {
                if x > extreme {
                    extreme = x;
                } else if extreme - x > min_swing {
                    dir = -1;
                    extreme = x;
                    reversals += 1;
                }
            }
            _ => {
                if x < extreme {
                    extreme = x;
                } else if x - extreme > min_swing {
                    dir = 1;
                    extreme = x;
                    reversals += 1;
                }
            }
        }
    }
    reversals
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const W: usize = 80;
    const H: usize = 60;
    const FPS: f64 = 30.0;

    /// Gray background with a bright 8x8 block whose left edge is at `x`.
    fn frame_with_block(x: i32) -> GrayFrame {
        let mut data = vec![100u8; W * H];
        for y in 25..33 {
            for dx in 0..8 {
                let xx = x + dx;
                if xx >= 0 && (xx as usize) < W {
                    data[y * W + xx as usize] = 220;
                }
            }
        }
        GrayFrame::new(W, H, data)
    }

    fn run(frames: &[GrayFrame]) -> Vec<Event> {
        let mut analyzer = MotionAnalyzer::new(Config::default());
        let mut events = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            if let Some(e) = analyzer.process(f, i as f64 / FPS) {
                events.push(e);
            }
        }
        events
    }

    #[test]
    fn identical_frames_produce_no_events() {
        let frames = vec![frame_with_block(30); 60];
        assert!(run(&frames).is_empty());
    }

    #[test]
    fn single_jump_on_the_left_is_moving_then_still() {
        let mut frames = vec![frame_with_block(2); 10];
        frames.extend(vec![frame_with_block(10); 60]);
        let events = run(&frames);
        assert_eq!(events.len(), 2, "events: {:?}", events);
        assert_eq!(events[0].action, Action::Moving);
        assert_eq!(events[0].zone, Some(Zone::Left));
        assert_eq!(events[1].action, Action::Still);
    }

    #[test]
    fn side_to_side_motion_is_waving() {
        let mut frames: Vec<GrayFrame> = (0..90)
            .map(|i| {
                let t = i as f64 / FPS;
                frame_with_block((40.0 + 20.0 * (2.0 * PI * 2.5 * t).sin()) as i32)
            })
            .collect();
        frames.extend(vec![frame_with_block(40); 40]);
        let events = run(&frames);
        assert!(
            events.iter().any(|e| e.action == Action::Waving),
            "events: {:?}",
            events
        );
        assert_eq!(events.last().map(|e| e.action), Some(Action::Still));
    }

    #[test]
    fn one_way_walk_is_not_waving() {
        let mut frames: Vec<GrayFrame> = (0..18).map(|i| frame_with_block(2 + 4 * i)).collect();
        frames.extend(vec![frame_with_block(70); 60]);
        let events = run(&frames);
        assert!(!events.is_empty());
        assert!(events.iter().all(|e| e.action != Action::Waving));
    }

    #[test]
    fn slow_sway_is_ignored() {
        let frames: Vec<GrayFrame> = (0..150)
            .map(|i| {
                let t = i as f64 / FPS;
                frame_with_block((40.0 + 20.0 * (2.0 * PI * 0.3 * t).sin()) as i32)
            })
            .collect();
        assert!(run(&frames).is_empty());
    }

    #[test]
    fn small_sensor_noise_is_ignored() {
        // Deterministic pseudo-random noise of +/- 4 levels (below the threshold).
        let mut seed: u32 = 12345;
        let frames: Vec<GrayFrame> = (0..120)
            .map(|_| {
                let data: Vec<u8> = (0..W * H)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        let n = ((seed >> 24) % 9) as i32 - 4;
                        (100 + n) as u8
                    })
                    .collect();
                GrayFrame::new(W, H, data)
            })
            .collect();
        assert!(run(&frames).is_empty());
    }

    #[test]
    fn whole_image_brightness_flicker_is_ignored() {
        let frames: Vec<GrayFrame> = (0..90)
            .map(|i| GrayFrame::filled(W, H, if i % 7 == 0 { 160 } else { 100 }))
            .collect();
        assert!(run(&frames).is_empty());
    }
}
