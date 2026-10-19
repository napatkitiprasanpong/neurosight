//! "Reaching out a hand" (a handshake offer), from one person's arm keypoints.
//!
//! A hand held out toward a camera in front of the person looks like this in the image:
//! the upper arm hangs down from the shoulder (elbow clearly below the shoulder), but the
//! forearm points at the camera, so it looks very short (the wrist is close to the elbow)
//! and the wrist is at about the same height as the elbow, between chest and waist.
//! The pose must be held fairly still for a moment.
//!
//! This is a heuristic on 2D points: it cannot see an actual handshake between two people,
//! and it only recognizes the "hand held out toward the camera" pose.

use std::collections::VecDeque;

use neurosight_common::Keypoint;

#[derive(Clone, Debug)]
pub struct ReachConfig {
    /// All three arm points must have at least this confidence.
    pub min_score: f32,
    /// The upper arm must look at least this long (shoulder widths).
    pub upper_min: f32,
    /// The forearm must look at most this long (shoulder widths) ...
    pub fore_max: f32,
    /// ... and at most this fraction of the upper arm.
    pub fore_ratio_max: f32,
    /// The elbow must be at least this far below the shoulder (shoulder widths).
    pub elbow_below_min: f32,
    /// The wrist must be this far below the shoulder, at least and at most.
    pub wrist_below_min: f32,
    pub wrist_below_max: f32,
    /// The wrist must be within this vertical distance of the elbow.
    pub wrist_elbow_dy_max: f32,
    /// The wrist may not move more than this (shoulder widths) over the steady window.
    pub steady_range_max: f32,
    pub steady_window_s: f64,
    /// The pose must hold this long before it is reported.
    pub hold_s: f64,
    /// The pose must be gone this long before a new report is possible.
    pub end_s: f64,
}

impl Default for ReachConfig {
    fn default() -> Self {
        Self {
            min_score: 0.3,
            upper_min: 0.35,
            fore_max: 0.35,
            fore_ratio_max: 0.6,
            elbow_below_min: 0.25,
            wrist_below_min: 0.25,
            wrist_below_max: 1.3,
            wrist_elbow_dy_max: 0.25,
            steady_range_max: 0.15,
            steady_window_s: 0.5,
            hold_s: 0.6,
            end_s: 0.4,
        }
    }
}

struct ArmState {
    /// (time, wrist x, wrist y) in shoulder widths, only while the arm geometry fits.
    track: VecDeque<(f64, f32, f32)>,
    ok_since: Option<f64>,
    last_ok: f64,
    active: bool,
}

impl ArmState {
    fn new() -> Self {
        Self {
            track: VecDeque::new(),
            ok_since: None,
            last_ok: f64::NEG_INFINITY,
            active: false,
        }
    }
}

pub struct ReachDetector {
    cfg: ReachConfig,
    arms: [ArmState; 2],
}

impl ReachDetector {
    pub fn new(cfg: ReachConfig) -> Self {
        Self {
            cfg,
            arms: [ArmState::new(), ArmState::new()],
        }
    }

    pub fn reset(&mut self) {
        self.arms = [ArmState::new(), ArmState::new()];
    }

    /// Feed one frame. `aspect` is image width / height and `shoulder_w` is the shoulder
    /// width in image-height units. Returns true when a reach starts (either arm).
    pub fn process(
        &mut self,
        shoulders: [Keypoint; 2],
        elbows: [Keypoint; 2],
        wrists: [Keypoint; 2],
        aspect: f32,
        shoulder_w: f32,
        t: f64,
    ) -> bool {
        let cfg = self.cfg.clone();
        let mut started = false;
        for i in 0..2 {
            let ok = geometry_ok(shoulders[i], elbows[i], wrists[i], aspect, shoulder_w, &cfg);
            let arm = &mut self.arms[i];
            if ok {
                arm.track.push_back((
                    t,
                    wrists[i].x * aspect / shoulder_w,
                    wrists[i].y / shoulder_w,
                ));
            }
            while arm
                .track
                .front()
                .map_or(false, |s| s.0 < t - cfg.steady_window_s)
            {
                arm.track.pop_front();
            }

            let mut steady = true;
            if arm.track.len() >= 3 {
                let (mut x_lo, mut x_hi) = (f32::MAX, f32::MIN);
                let (mut y_lo, mut y_hi) = (f32::MAX, f32::MIN);
                for s in arm.track.iter() {
                    x_lo = x_lo.min(s.1);
                    x_hi = x_hi.max(s.1);
                    y_lo = y_lo.min(s.2);
                    y_hi = y_hi.max(s.2);
                }
                steady = x_hi - x_lo <= cfg.steady_range_max && y_hi - y_lo <= cfg.steady_range_max;
            }

            if ok && steady {
                arm.last_ok = t;
                let since = *arm.ok_since.get_or_insert(t);
                if !arm.active && t - since >= cfg.hold_s {
                    arm.active = true;
                    started = true;
                }
            } else if t - arm.last_ok >= cfg.end_s {
                arm.ok_since = None;
                arm.active = false;
            }
        }
        started
    }
}

/// Length between two keypoints in shoulder widths (x is corrected for the image shape).
fn dist(a: Keypoint, b: Keypoint, aspect: f32, shoulder_w: f32) -> f32 {
    ((a.x - b.x) * aspect).hypot(a.y - b.y) / shoulder_w
}

fn geometry_ok(
    sh: Keypoint,
    el: Keypoint,
    wr: Keypoint,
    aspect: f32,
    shoulder_w: f32,
    cfg: &ReachConfig,
) -> bool {
    if sh.score.min(el.score).min(wr.score) < cfg.min_score {
        return false;
    }
    let upper = dist(sh, el, aspect, shoulder_w);
    let fore = dist(el, wr, aspect, shoulder_w);
    if upper < cfg.upper_min || fore > cfg.fore_max || fore > cfg.fore_ratio_max * upper {
        return false;
    }
    if (el.y - sh.y) / shoulder_w < cfg.elbow_below_min {
        return false;
    }
    let wrist_below = (wr.y - sh.y) / shoulder_w;
    if wrist_below < cfg.wrist_below_min || wrist_below > cfg.wrist_below_max {
        return false;
    }
    (wr.y - el.y).abs() / shoulder_w <= cfg.wrist_elbow_dy_max
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASPECT: f32 = 4.0 / 3.0;
    const FPS: f64 = 30.0;
    /// Shoulder width in image-height units for shoulders 0.2 apart (in image width).
    const SW_H: f32 = 0.2 * ASPECT;

    fn kpt(x: f32, y: f32) -> Keypoint {
        Keypoint { x, y, score: 0.9 }
    }

    /// Shoulder + elbow + wrist for one arm. Offsets are in shoulder widths:
    /// the elbow relative to the shoulder, the wrist relative to the elbow.
    fn arm(sh: (f32, f32), upper: (f32, f32), fore: (f32, f32)) -> [Keypoint; 3] {
        let s = kpt(sh.0, sh.1);
        let e = kpt(s.x + upper.0 * SW_H / ASPECT, s.y + upper.1 * SW_H);
        let w = kpt(e.x + fore.0 * SW_H / ASPECT, e.y + fore.1 * SW_H);
        [s, e, w]
    }

    const LEFT_SH: (f32, f32) = (0.6, 0.4);
    const RIGHT_SH: (f32, f32) = (0.4, 0.4);

    fn hang(sh: (f32, f32)) -> [Keypoint; 3] {
        arm(sh, (0.1, 0.55), (0.0, 0.55))
    }

    fn run(frames: &[([Keypoint; 3], [Keypoint; 3])]) -> usize {
        let mut det = ReachDetector::new(ReachConfig::default());
        let mut count = 0;
        for (i, (l, r)) in frames.iter().enumerate() {
            let started = det.process(
                [l[0], r[0]],
                [l[1], r[1]],
                [l[2], r[2]],
                ASPECT,
                SW_H,
                i as f64 / FPS,
            );
            if started {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn hanging_arms_are_not_a_reach() {
        let frames = vec![(hang(LEFT_SH), hang(RIGHT_SH)); 60];
        assert_eq!(run(&frames), 0);
    }

    #[test]
    fn hand_held_out_is_a_reach() {
        let out = arm(RIGHT_SH, (0.05, 0.6), (-0.02, -0.05));
        let frames = vec![(hang(LEFT_SH), out); 60];
        assert_eq!(run(&frames), 1);
    }

    #[test]
    fn hand_held_out_with_jitter_is_still_a_reach() {
        let mut seed = 5u32;
        let frames: Vec<_> = (0..60)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let j = ((seed >> 16) as f32 / 65535.0 - 0.5) * 0.06;
                (hang(LEFT_SH), arm(RIGHT_SH, (0.05, 0.6), (-0.02 + j, -0.05)))
            })
            .collect();
        assert_eq!(run(&frames), 1);
    }

    #[test]
    fn raised_hand_is_not_a_reach() {
        let raised = arm(RIGHT_SH, (0.1, 0.5), (0.0, -0.55));
        assert_eq!(run(&vec![(hang(LEFT_SH), raised); 60]), 0);
    }

    #[test]
    fn hand_on_chin_is_not_a_reach() {
        let chin = arm(RIGHT_SH, (0.15, 0.5), (-0.3, -0.45));
        assert_eq!(run(&vec![(hang(LEFT_SH), chin); 60]), 0);
    }

    #[test]
    fn crossed_arms_are_not_a_reach() {
        let l = arm(LEFT_SH, (0.2, 0.5), (-0.7, 0.05));
        let r = arm(RIGHT_SH, (-0.2, 0.5), (0.7, 0.05));
        assert_eq!(run(&vec![(l, r); 60]), 0);
    }

    #[test]
    fn hands_on_hips_are_not_a_reach() {
        let l = arm(LEFT_SH, (0.4, 0.45), (-0.3, 0.5));
        let r = arm(RIGHT_SH, (-0.4, 0.45), (0.3, 0.5));
        assert_eq!(run(&vec![(l, r); 60]), 0);
    }

    #[test]
    fn moving_hand_is_not_a_reach() {
        let frames: Vec<_> = (0..60)
            .map(|i| {
                let dx = 0.25 * (2.0 * std::f64::consts::PI * 2.0 * i as f64 / FPS).sin() as f32;
                (hang(LEFT_SH), arm(RIGHT_SH, (0.05, 0.6), (-0.02 + dx, -0.05)))
            })
            .collect();
        assert_eq!(run(&frames), 0);
    }

    #[test]
    fn very_brief_reach_is_ignored() {
        let out = arm(RIGHT_SH, (0.05, 0.6), (-0.02, -0.05));
        let mut frames = vec![(hang(LEFT_SH), out); 9];
        frames.extend(vec![(hang(LEFT_SH), hang(RIGHT_SH)); 30]);
        assert_eq!(run(&frames), 0);
    }
}
