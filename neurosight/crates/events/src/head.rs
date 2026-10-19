//! Head gestures from the position of the nose relative to the shoulders.
//!
//! Input per frame: where the nose is, measured from the middle of the shoulders in
//! units of shoulder width:
//!   x: sideways (positive = toward the right of the image)
//!   y: vertical (positive = down)
//!
//! * Turning the head sideways moves the nose sideways; looking up or down moves it
//!   up or down. Because it is measured from the shoulders, moving the whole body
//!   does not count.
//! * Turns and looks are measured against a slowly adapting "neutral" position
//!   (learned from the first half second and then re-learned while the head is
//!   neutral), so it does not matter if the person sits off-center or the camera
//!   is high or low.
//! * Shaking (no) and nodding (yes) are repeated sideways / vertical swings, found
//!   the same way as hand waving: several direction changes within 1.5 seconds.
//! * While shaking or nodding is going on, single turns and looks are not reported.

use std::collections::VecDeque;

use crate::{count_reversals, smooth3};

#[derive(Clone, Debug)]
pub struct HeadConfig {
    /// Seconds of history used to find shaking / nodding.
    pub window_s: f64,
    /// The history must contain a sample this recent.
    pub recent_s: f64,
    /// The swing must still be going on in this many last seconds.
    pub recent_motion_s: f64,
    /// Direction changes needed in the window.
    pub min_reversals: usize,
    /// Minimum vertical swing (shoulder widths) for a nod direction change.
    pub nod_swing: f32,
    /// Minimum sideways swing (shoulder widths) for a shake direction change.
    pub shake_swing: f32,
    /// The total swing must be at least this many times the minimum swing
    /// (helps ignore camera jitter).
    pub min_amp_factor: f32,
    /// Allowed frequency range of shaking / nodding in Hz.
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    /// Sideways offset from neutral that counts as a turn (shoulder widths).
    pub turn_enter: f32,
    /// ... and the offset below which the head counts as back to neutral.
    pub turn_exit: f32,
    /// Vertical offset from neutral that counts as looking up / down.
    pub pitch_enter: f32,
    pub pitch_exit: f32,
    /// A new head position must last this long before it is reported.
    pub dwell_s: f64,
    /// Time constant of the neutral-position learning, in seconds.
    pub tau_s: f64,
    /// Frames used to learn the first neutral position.
    pub init_frames: usize,
    /// Keep treating the head as shaking / nodding this long after the swing stops.
    pub osc_hold_s: f64,
}

impl Default for HeadConfig {
    fn default() -> Self {
        Self {
            window_s: 1.5,
            recent_s: 0.3,
            recent_motion_s: 0.5,
            min_reversals: 4,
            nod_swing: 0.05,
            shake_swing: 0.06,
            min_amp_factor: 1.5,
            min_freq_hz: 0.8,
            max_freq_hz: 5.0,
            turn_enter: 0.12,
            turn_exit: 0.06,
            pitch_enter: 0.07,
            pitch_exit: 0.035,
            dwell_s: 0.25,
            tau_s: 4.0,
            init_frames: 10,
            osc_hold_s: 0.7,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGesture {
    Nodding,
    ShakingHead,
    /// Toward the left of the image.
    TurnedLeft,
    /// Toward the right of the image.
    TurnedRight,
    LookedUp,
    LookedDown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LookState {
    Neutral,
    Left,
    Right,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Osc {
    Nodding,
    Shaking,
}

/// Numbers from the latest frame, useful for tuning.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadDebug {
    /// Offset from the learned neutral position, in shoulder widths.
    pub dx: f32,
    pub dy: f32,
    pub shake_reversals: usize,
    pub nod_reversals: usize,
}

pub struct HeadAnalyzer {
    cfg: HeadConfig,
    xs: VecDeque<(f64, f32)>,
    ys: VecDeque<(f64, f32)>,
    base: Option<(f32, f32)>,
    init: Vec<(f32, f32)>,
    state: LookState,
    cand: LookState,
    cand_since: f64,
    osc: Option<Osc>,
    osc_t: f64,
    last_t: Option<f64>,
    debug: HeadDebug,
}

impl HeadAnalyzer {
    pub fn new(cfg: HeadConfig) -> Self {
        Self {
            cfg,
            xs: VecDeque::new(),
            ys: VecDeque::new(),
            base: None,
            init: Vec::new(),
            state: LookState::Neutral,
            cand: LookState::Neutral,
            cand_since: 0.0,
            osc: None,
            osc_t: f64::NEG_INFINITY,
            last_t: None,
            debug: HeadDebug::default(),
        }
    }

    pub fn debug(&self) -> HeadDebug {
        self.debug
    }

    /// Forget everything (for example when the person leaves).
    pub fn reset(&mut self) {
        *self = Self::new(self.cfg.clone());
    }

    /// Feed the nose position for one frame (see the module docs for units).
    /// Returns a gesture when a new one starts.
    pub fn process(&mut self, x: f32, y: f32, t: f64) -> Option<HeadGesture> {
        let dt = match self.last_t {
            None => 0.033,
            Some(last) => (t - last).max(1e-3),
        };
        self.last_t = Some(t);

        let cutoff = t - self.cfg.window_s;
        self.xs.push_back((t, x));
        self.ys.push_back((t, y));
        while self.xs.front().map_or(false, |s| s.0 < cutoff) {
            self.xs.pop_front();
        }
        while self.ys.front().map_or(false, |s| s.0 < cutoff) {
            self.ys.pop_front();
        }

        // Shaking or nodding?
        let (shake_ok, shake_rev, shake_amp) =
            osc_check(&self.xs, self.cfg.shake_swing, t, &self.cfg);
        let (nod_ok, nod_rev, nod_amp) = osc_check(&self.ys, self.cfg.nod_swing, t, &self.cfg);
        self.debug.shake_reversals = shake_rev;
        self.debug.nod_reversals = nod_rev;

        let kind = match (shake_ok, nod_ok) {
            (true, true) => {
                if shake_amp / self.cfg.shake_swing >= nod_amp / self.cfg.nod_swing {
                    Some(Osc::Shaking)
                } else {
                    Some(Osc::Nodding)
                }
            }
            (true, false) => Some(Osc::Shaking),
            (false, true) => Some(Osc::Nodding),
            (false, false) => None,
        };

        let mut event: Option<HeadGesture> = None;
        if let Some(k) = kind {
            self.osc_t = t;
            if self.osc != Some(k) {
                self.osc = Some(k);
                event = Some(match k {
                    Osc::Nodding => HeadGesture::Nodding,
                    Osc::Shaking => HeadGesture::ShakingHead,
                });
            }
        } else if self.osc.is_some() && t - self.osc_t >= self.cfg.osc_hold_s {
            self.osc = None;
        }

        // Learn the first neutral position.
        let (base_x, base_y) = match self.base {
            Some(b) => b,
            None => {
                self.init.push((x, y));
                if self.init.len() >= self.cfg.init_frames {
                    let n = self.init.len() as f32;
                    let bx = self.init.iter().map(|p| p.0).sum::<f32>() / n;
                    let by = self.init.iter().map(|p| p.1).sum::<f32>() / n;
                    self.base = Some((bx, by));
                }
                return event;
            }
        };

        // No single turns or looks while shaking / nodding.
        if self.osc.is_some() {
            self.state = LookState::Neutral;
            self.cand = LookState::Neutral;
            self.cand_since = t;
            return event;
        }

        let dx = x - base_x;
        let dy = y - base_y;
        self.debug.dx = dx;
        self.debug.dy = dy;

        // Where is the head, with a bit of hysteresis and a short dwell time?
        let cand = candidate(self.state, dx, dy, &self.cfg);
        if cand != self.cand {
            self.cand = cand;
            self.cand_since = t;
        }
        if self.cand != self.state && t - self.cand_since >= self.cfg.dwell_s {
            self.state = self.cand;
            if event.is_none() {
                event = match self.state {
                    LookState::Neutral => None,
                    LookState::Left => Some(HeadGesture::TurnedLeft),
                    LookState::Right => Some(HeadGesture::TurnedRight),
                    LookState::Up => Some(HeadGesture::LookedUp),
                    LookState::Down => Some(HeadGesture::LookedDown),
                };
            }
        }

        // Keep learning the neutral position while the head is neutral.
        if self.state == LookState::Neutral
            && dx.abs() < self.cfg.turn_enter * 0.8
            && dy.abs() < self.cfg.pitch_enter * 0.8
        {
            let a = (1.0 - (-dt / self.cfg.tau_s).exp()) as f32;
            self.base = Some((base_x + a * dx, base_y + a * dy));
        }

        event
    }
}

/// Which head position does this offset from neutral look like?
fn candidate(state: LookState, dx: f32, dy: f32, cfg: &HeadConfig) -> LookState {
    let turn_limit = if matches!(state, LookState::Left | LookState::Right) {
        cfg.turn_exit
    } else {
        cfg.turn_enter
    };
    let pitch_limit = if matches!(state, LookState::Up | LookState::Down) {
        cfg.pitch_exit
    } else {
        cfg.pitch_enter
    };
    let sx = dx.abs() / turn_limit;
    let sy = dy.abs() / pitch_limit;
    if sx < 1.0 && sy < 1.0 {
        LookState::Neutral
    } else if sx >= sy {
        if dx > 0.0 {
            LookState::Right
        } else {
            LookState::Left
        }
    } else if dy > 0.0 {
        LookState::Down
    } else {
        LookState::Up
    }
}

/// Is this signal swinging back and forth? Returns (yes, direction changes, total swing).
fn osc_check(
    track: &VecDeque<(f64, f32)>,
    swing: f32,
    t: f64,
    cfg: &HeadConfig,
) -> (bool, usize, f32) {
    let (first, last) = match (track.front(), track.back()) {
        (Some(f), Some(l)) => (*f, *l),
        _ => return (false, 0, 0.0),
    };
    if track.len() < 8 || last.0 < t - cfg.recent_s {
        return (false, 0, 0.0);
    }
    let span_t = last.0 - first.0;
    if span_t < 0.6 {
        return (false, 0, 0.0);
    }

    // The swing must still be going on right now.
    let late: Vec<f32> = track
        .iter()
        .filter(|s| s.0 >= t - cfg.recent_motion_s)
        .map(|s| s.1)
        .collect();
    if late.len() < 3 {
        return (false, 0, 0.0);
    }
    let (lo, hi) = late
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    if hi - lo < 0.5 * swing {
        return (false, 0, 0.0);
    }

    let values: Vec<f32> = track.iter().map(|s| s.1).collect();
    let smooth = smooth3(&values);
    let reversals = count_reversals(&smooth, swing);
    let freq = reversals as f64 / 2.0 / span_t;
    let (mn, mx) = smooth
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let amp = mx - mn;
    let ok = reversals >= cfg.min_reversals
        && freq >= cfg.min_freq_hz
        && freq <= cfg.max_freq_hz
        && amp >= cfg.min_amp_factor * swing;
    (ok, reversals, amp)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::f64::consts::PI;

    const FPS: f64 = 30.0;
    /// Shoulder width in cm, used by the simple head model below.
    const SHOULDER_CM: f64 = 40.0;

    /// Nose position (x sideways, y down), in shoulder widths from the middle of the
    /// shoulders, for a head turned by `yaw_deg` (positive = toward the image right)
    /// and tilted by `pitch_deg` (positive = looking down). Simple model: the nose tip
    /// is 11 cm in front of the head center, the head center is 20 cm above the shoulders.
    pub(crate) fn nose_offset(yaw_deg: f64, pitch_deg: f64) -> (f32, f32) {
        let psi = yaw_deg.to_radians();
        let phi = pitch_deg.to_radians();
        let (nx, ny, nz) = (0.0_f64, -1.0_f64, 11.0_f64);
        let x1 = nx * psi.cos() + nz * psi.sin();
        let z1 = -nx * psi.sin() + nz * psi.cos();
        let y2 = ny * phi.cos() - z1 * phi.sin();
        let x = x1 / SHOULDER_CM;
        let y = -(20.0 + y2) / SHOULDER_CM;
        (x as f32, y as f32)
    }

    /// Small deterministic noise (roughly bell-shaped), `sigma` in shoulder widths.
    fn noise(seed: &mut u32, sigma: f32) -> f32 {
        let mut sum = 0.0_f32;
        for _ in 0..4 {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            sum += (*seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
        }
        sum * sigma * 1.73 // sum of 4 uniforms(-0.5..0.5) has std 0.577
    }

    fn ramp(a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * i as f64 / (n.max(2) - 1) as f64)
            .collect()
    }

    fn run(yaws: &[f64], pitches: &[f64], sigma: f32) -> Vec<(f64, HeadGesture)> {
        let mut head = HeadAnalyzer::new(HeadConfig::default());
        let mut seed = 7u32;
        let mut out = Vec::new();
        for (i, (&yw, &pt)) in yaws.iter().zip(pitches.iter()).enumerate() {
            let (x, y) = nose_offset(yw, pt);
            let (x, y) = (x + noise(&mut seed, sigma), y + noise(&mut seed, sigma));
            let t = i as f64 / FPS;
            if let Some(g) = head.process(x, y, t) {
                out.push((t, g));
            }
        }
        out
    }

    fn gestures(events: &[(f64, HeadGesture)]) -> Vec<HeadGesture> {
        events.iter().map(|e| e.1).collect()
    }

    #[test]
    fn neutral_head_is_quiet_even_with_noise() {
        assert!(run(&vec![0.0; 150], &vec![0.0; 150], 0.0).is_empty());
        assert!(run(&vec![0.0; 150], &vec![0.0; 150], 0.012).is_empty());
    }

    #[test]
    fn turning_right_then_left() {
        let mut yaw = vec![0.0; 30];
        yaw.extend(ramp(0.0, 50.0, 15));
        yaw.extend(vec![50.0; 45]);
        yaw.extend(ramp(50.0, 0.0, 15));
        yaw.extend(vec![0.0; 30]);
        yaw.extend(ramp(0.0, -50.0, 15));
        yaw.extend(vec![-50.0; 45]);
        yaw.extend(ramp(-50.0, 0.0, 15));
        yaw.extend(vec![0.0; 30]);
        let pitch = vec![0.0; yaw.len()];
        for sigma in [0.0, 0.012] {
            let events = run(&yaw, &pitch, sigma);
            assert_eq!(
                gestures(&events),
                vec![HeadGesture::TurnedRight, HeadGesture::TurnedLeft],
                "sigma {}: {:?}",
                sigma,
                events
            );
        }
    }

    #[test]
    fn looking_up_then_down() {
        let mut pitch = vec![0.0; 30];
        pitch.extend(ramp(0.0, -25.0, 12));
        pitch.extend(vec![-25.0; 45]);
        pitch.extend(ramp(-25.0, 0.0, 12));
        pitch.extend(vec![0.0; 30]);
        pitch.extend(ramp(0.0, 25.0, 12));
        pitch.extend(vec![25.0; 45]);
        pitch.extend(ramp(25.0, 0.0, 12));
        pitch.extend(vec![0.0; 30]);
        let yaw = vec![0.0; pitch.len()];
        for sigma in [0.0, 0.012] {
            let events = run(&yaw, &pitch, sigma);
            assert_eq!(
                gestures(&events),
                vec![HeadGesture::LookedUp, HeadGesture::LookedDown],
                "sigma {}: {:?}",
                sigma,
                events
            );
        }
    }

    #[test]
    fn nodding_is_not_reported_as_looking_up_or_down() {
        let n = 120;
        let mut pitch = vec![0.0; 30];
        pitch.extend((0..n).map(|i| 12.0 * (2.0 * PI * 2.0 * i as f64 / FPS).sin()));
        let yaw = vec![0.0; pitch.len()];
        for sigma in [0.0, 0.012] {
            let events = run(&yaw, &pitch, sigma);
            assert_eq!(gestures(&events), vec![HeadGesture::Nodding], "{:?}", events);
        }
    }

    #[test]
    fn shaking_is_not_reported_as_turns() {
        let n = 120;
        let mut yaw = vec![0.0; 30];
        yaw.extend((0..n).map(|i| 30.0 * (2.0 * PI * 2.0 * i as f64 / FPS).sin()));
        let pitch = vec![0.0; yaw.len()];
        for sigma in [0.0, 0.012] {
            let events = run(&yaw, &pitch, sigma);
            assert_eq!(
                gestures(&events),
                vec![HeadGesture::ShakingHead],
                "{:?}",
                events
            );
        }
    }

    #[test]
    fn nod_then_shake() {
        let n = 120;
        let mut yaw = vec![0.0; 30];
        let mut pitch = vec![0.0; 30];
        // nod
        yaw.extend(vec![0.0; n]);
        pitch.extend((0..n).map(|i| 12.0 * (2.0 * PI * 2.0 * i as f64 / FPS).sin()));
        // pause
        yaw.extend(vec![0.0; 30]);
        pitch.extend(vec![0.0; 30]);
        // shake
        yaw.extend((0..n).map(|i| 30.0 * (2.0 * PI * 2.0 * i as f64 / FPS).sin()));
        pitch.extend(vec![0.0; n]);
        yaw.extend(vec![0.0; 30]);
        pitch.extend(vec![0.0; 30]);
        let events = run(&yaw, &pitch, 0.006);
        assert_eq!(
            gestures(&events),
            vec![HeadGesture::Nodding, HeadGesture::ShakingHead],
            "{:?}",
            events
        );
    }

    #[test]
    fn slow_drift_is_not_a_gesture() {
        // Looking down by 15 degrees very slowly (like reading) is absorbed by the neutral position.
        let mut pitch = vec![0.0; 30];
        pitch.extend(ramp(0.0, 15.0, 90));
        let yaw = vec![0.0; pitch.len()];
        assert!(run(&yaw, &pitch, 0.0).is_empty());
    }
}
