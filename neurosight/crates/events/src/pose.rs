//! Pose-based event engine.
//!
//! Input: the body keypoints of one person (or nothing) for each frame.
//! Output: events such as "someone appeared", "someone is waving at you",
//! "the person left".
//!
//! How a wave is recognized:
//!
//! * The hand must be raised: the wrist is above the elbow.
//! * The wrist swings left and right (measured in shoulder widths, so it does
//!   not matter how far away the person is, and walking sideways does not count).
//! * The swing changes direction several times within about 1.5 seconds, at
//!   a hand-wave frequency (about 1-6 Hz).
//! * The swing is still going on in the last half second (so "stopped waving"
//!   is noticed quickly).
//!
//! Other upper-body gestures come from two helpers:
//!
//! * `HeadAnalyzer` (head.rs): turning the head left / right, looking up / down,
//!   nodding and shaking the head.
//! * `ReachDetector` (reach.rs): holding a hand out toward the camera (handshake offer).

use std::collections::VecDeque;

use neurosight_common::{kp, Action, Distance, Event, Pose, Zone};

use crate::head::{HeadAnalyzer, HeadConfig, HeadGesture};
use crate::reach::{ReachConfig, ReachDetector};
use crate::{count_reversals, smooth3};

/// Tunable thresholds. The defaults are a starting point.
#[derive(Clone, Debug)]
pub struct PoseConfig {
    /// A keypoint counts as "seen" if its confidence is at least this.
    pub min_score: f32,
    /// Frames in a row a person must be seen before "appeared".
    pub appear_frames: usize,
    /// Seconds without a person before "left".
    pub leave_s: f64,
    /// Seconds of wrist history to analyze.
    pub window_s: f64,
    /// A wave needs wrist samples this recent.
    pub recent_s: f64,
    /// Direction changes needed in the window to call it a wave.
    pub min_reversals: usize,
    /// Minimum swing for a direction change, in shoulder widths.
    pub min_swing: f32,
    /// Allowed wave frequency range in Hz.
    pub min_freq_hz: f64,
    pub max_freq_hz: f64,
    /// Wrist must be above the elbow by this much (fraction of image height).
    pub raised_margin: f32,
    /// Fraction of the window the hand must be raised.
    pub raised_fraction: f32,
    /// Keep "waving" this long after the evidence disappears.
    pub hold_s: f64,
    /// Seconds over which the swing must still be going on.
    pub recent_motion_s: f64,
    /// Shoulder width (fraction of image width) above which the person is near.
    pub near_span: f32,
    /// ...and below which the person is far.
    pub far_span: f32,
    /// Width divided by height of the camera image (the program updates this from the
    /// real camera). Needed to measure lengths correctly.
    pub image_aspect: f32,
    /// Settings for head gestures.
    pub head: HeadConfig,
    /// Settings for the hand-held-out (handshake offer) pose.
    pub reach: ReachConfig,
}

impl Default for PoseConfig {
    fn default() -> Self {
        Self {
            min_score: 0.3,
            appear_frames: 3,
            leave_s: 1.0,
            window_s: 1.5,
            recent_s: 0.3,
            min_reversals: 3,
            min_swing: 0.25,
            min_freq_hz: 0.8,
            max_freq_hz: 6.0,
            raised_margin: 0.02,
            raised_fraction: 0.7,
            hold_s: 0.5,
            recent_motion_s: 0.5,
            near_span: 0.30,
            far_span: 0.15,
            image_aspect: 4.0 / 3.0,
            head: HeadConfig::default(),
            reach: ReachConfig::default(),
        }
    }
}

/// Numbers from the latest frame, useful for tuning.
#[derive(Clone, Copy, Debug, Default)]
pub struct PoseDebug {
    pub person_visible: bool,
    pub shoulder_span: f32,
    pub left_raised: bool,
    pub right_raised: bool,
    pub left_reversals: usize,
    pub right_reversals: usize,
    /// Head offset from its neutral position, in shoulder widths.
    pub head_dx: f32,
    pub head_dy: f32,
    pub shake_reversals: usize,
    pub nod_reversals: usize,
}

#[derive(Clone, Copy, Debug)]
struct WristSample {
    t: f64,
    rel_x: f32,
    raised: bool,
}

#[derive(Clone, Copy, Debug)]
struct PersonInfo {
    mid_x: f32,
    mid_y: f32,
    span: f32,
    facing: bool,
}

pub struct PoseAnalyzer {
    cfg: PoseConfig,
    present: bool,
    streak: usize,
    last_seen_t: f64,
    waving: bool,
    last_wave_t: f64,
    /// Index 0 = left wrist, 1 = right wrist.
    tracks: [VecDeque<WristSample>; 2],
    head: HeadAnalyzer,
    reach: ReachDetector,
    /// Events found but not yet handed out (one is returned per `process` call).
    pending: VecDeque<Event>,
    aspect: f32,
    debug: PoseDebug,
}

impl PoseAnalyzer {
    pub fn new(cfg: PoseConfig) -> Self {
        let head = HeadAnalyzer::new(cfg.head.clone());
        let reach = ReachDetector::new(cfg.reach.clone());
        let aspect = cfg.image_aspect;
        Self {
            cfg,
            present: false,
            streak: 0,
            last_seen_t: f64::NEG_INFINITY,
            waving: false,
            last_wave_t: f64::NEG_INFINITY,
            tracks: [VecDeque::new(), VecDeque::new()],
            head,
            reach,
            pending: VecDeque::new(),
            aspect,
            debug: PoseDebug::default(),
        }
    }

    pub fn debug(&self) -> PoseDebug {
        self.debug
    }

    /// Tell the analyzer the camera image shape (width / height).
    pub fn set_aspect(&mut self, aspect: f32) {
        if aspect > 0.1 {
            self.aspect = aspect;
        }
    }

    /// Feed the pose for one frame (None if the model found nobody).
    /// Returns an event only when something worth announcing changes. If several
    /// things happen at the same moment, the extra events come out on the next calls.
    pub fn process(&mut self, pose: Option<&Pose>, t: f64) -> Option<Event> {
        if let Some(event) = self.step(pose, t) {
            self.pending.push_back(event);
        }
        self.pending.pop_front()
    }

    fn step(&mut self, pose: Option<&Pose>, t: f64) -> Option<Event> {
        let info = pose.and_then(|p| self.person_info(p));

        if info.is_some() {
            self.streak += 1;
            self.last_seen_t = t;
        } else {
            self.streak = 0;
        }
        self.debug = PoseDebug {
            person_visible: info.is_some(),
            shoulder_span: info.map_or(0.0, |i| i.span),
            ..PoseDebug::default()
        };

        // Someone appears.
        if !self.present && self.streak >= self.cfg.appear_frames {
            if let Some(i) = info {
                self.present = true;
                return Some(self.make_event(t, Action::Appeared, i, 0.9));
            }
        }

        // Someone leaves.
        if self.present && t - self.last_seen_t >= self.cfg.leave_s {
            self.present = false;
            self.waving = false;
            for track in self.tracks.iter_mut() {
                track.clear();
            }
            self.head.reset();
            self.reach.reset();
            return Some(Event {
                time_s: t,
                action: Action::Left,
                zone: None,
                distance: None,
                facing_camera: false,
                confidence: 1.0,
            });
        }

        let (info, pose) = match (self.present, info, pose) {
            (true, Some(i), Some(p)) => (i, p),
            _ => return None,
        };

        // Head gestures (need the nose) and the hand-held-out pose.
        let shoulder_w = info.span * self.aspect;
        let nose = pose.keypoints[kp::NOSE];
        if nose.score >= self.cfg.min_score {
            let hx = (nose.x - info.mid_x) / info.span;
            let hy = (nose.y - info.mid_y) / shoulder_w;
            if let Some(gesture) = self.head.process(hx, hy, t) {
                let action = match gesture {
                    HeadGesture::Nodding => Action::Nodding,
                    HeadGesture::ShakingHead => Action::ShakingHead,
                    HeadGesture::TurnedLeft => Action::LookedLeft,
                    HeadGesture::TurnedRight => Action::LookedRight,
                    HeadGesture::LookedUp => Action::LookedUp,
                    HeadGesture::LookedDown => Action::LookedDown,
                };
                let event = self.make_event(t, action, info, 0.8);
                self.pending.push_back(event);
            }
            let hd = self.head.debug();
            self.debug.head_dx = hd.dx;
            self.debug.head_dy = hd.dy;
            self.debug.shake_reversals = hd.shake_reversals;
            self.debug.nod_reversals = hd.nod_reversals;
        }
        let reached = self.reach.process(
            [
                pose.keypoints[kp::LEFT_SHOULDER],
                pose.keypoints[kp::RIGHT_SHOULDER],
            ],
            [
                pose.keypoints[kp::LEFT_ELBOW],
                pose.keypoints[kp::RIGHT_ELBOW],
            ],
            [
                pose.keypoints[kp::LEFT_WRIST],
                pose.keypoints[kp::RIGHT_WRIST],
            ],
            self.aspect,
            shoulder_w,
            t,
        );
        if reached {
            let event = self.make_event(t, Action::ReachingOut, info, 0.8);
            self.pending.push_back(event);
        }

        // Record where each wrist is (relative to the shoulders).
        let pairs = [
            (kp::LEFT_WRIST, kp::LEFT_ELBOW),
            (kp::RIGHT_WRIST, kp::RIGHT_ELBOW),
        ];
        let min_score = self.cfg.min_score;
        let raised_margin = self.cfg.raised_margin;
        let cutoff = t - self.cfg.window_s;
        for (idx, (wrist_i, elbow_i)) in pairs.iter().enumerate() {
            let w = pose.keypoints[*wrist_i];
            let e = pose.keypoints[*elbow_i];
            let track = &mut self.tracks[idx];
            if w.score >= min_score && e.score >= min_score {
                track.push_back(WristSample {
                    t,
                    rel_x: (w.x - info.mid_x) / info.span,
                    raised: w.y < e.y - raised_margin,
                });
            }
            while track.front().map_or(false, |s| s.t < cutoff) {
                track.pop_front();
            }
        }

        // Is either hand waving?
        let (left_wave, left_rev) = track_waving(&self.tracks[0], t, &self.cfg);
        let (right_wave, right_rev) = track_waving(&self.tracks[1], t, &self.cfg);
        let waving_now = left_wave || right_wave;
        let reversals = left_rev.max(right_rev);
        self.debug.left_reversals = left_rev;
        self.debug.right_reversals = right_rev;
        self.debug.left_raised = self.tracks[0].back().map_or(false, |s| s.raised);
        self.debug.right_raised = self.tracks[1].back().map_or(false, |s| s.raised);

        if waving_now {
            self.last_wave_t = t;
        }
        if !self.waving && waving_now {
            self.waving = true;
            let confidence = (reversals as f32 / 6.0).clamp(0.5, 1.0);
            return Some(self.make_event(t, Action::Waving, info, confidence));
        }
        if self.waving && !waving_now && t - self.last_wave_t >= self.cfg.hold_s {
            self.waving = false;
            return Some(self.make_event(t, Action::Present, info, 0.9));
        }
        None
    }

    fn person_info(&self, pose: &Pose) -> Option<PersonInfo> {
        let ls = pose.keypoints[kp::LEFT_SHOULDER];
        let rs = pose.keypoints[kp::RIGHT_SHOULDER];
        if ls.score < self.cfg.min_score || rs.score < self.cfg.min_score {
            return None;
        }
        let span = (ls.x - rs.x).abs();
        if span < 0.02 {
            return None;
        }
        let nose = pose.keypoints[kp::NOSE];
        let left_eye = pose.keypoints[kp::LEFT_EYE];
        let right_eye = pose.keypoints[kp::RIGHT_EYE];
        let facing = nose.score >= self.cfg.min_score
            && left_eye.score >= self.cfg.min_score
            && right_eye.score >= self.cfg.min_score;
        Some(PersonInfo {
            mid_x: (ls.x + rs.x) / 2.0,
            mid_y: (ls.y + rs.y) / 2.0,
            span,
            facing,
        })
    }

    fn distance_of(&self, span: f32) -> Distance {
        if span > self.cfg.near_span {
            Distance::Near
        } else if span < self.cfg.far_span {
            Distance::Far
        } else {
            Distance::Medium
        }
    }

    fn make_event(&self, t: f64, action: Action, info: PersonInfo, confidence: f32) -> Event {
        Event {
            time_s: t,
            action,
            zone: Some(Zone::from_x(info.mid_x)),
            distance: Some(self.distance_of(info.span)),
            facing_camera: info.facing,
            confidence,
        }
    }
}

/// Decide whether one wrist's recent history looks like waving.
/// Returns (is_waving, number_of_direction_changes).
fn track_waving(track: &VecDeque<WristSample>, t: f64, cfg: &PoseConfig) -> (bool, usize) {
    let (first, last) = match (track.front(), track.back()) {
        (Some(f), Some(l)) => (*f, *l),
        _ => return (false, 0),
    };
    if track.len() < 6 || last.t < t - cfg.recent_s {
        return (false, 0);
    }
    let span_t = last.t - first.t;
    if span_t < 0.6 {
        return (false, 0);
    }

    let raised = track.iter().filter(|s| s.raised).count() as f32 / track.len() as f32;
    if raised < cfg.raised_fraction {
        return (false, 0);
    }

    // The swing must still be going on right now.
    let late: Vec<f32> = track
        .iter()
        .filter(|s| s.t >= t - cfg.recent_motion_s)
        .map(|s| s.rel_x)
        .collect();
    if late.len() < 3 {
        return (false, 0);
    }
    let (lo, hi) = late
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    if hi - lo < 0.5 * cfg.min_swing {
        return (false, 0);
    }

    let xs: Vec<f32> = track.iter().map(|s| s.rel_x).collect();
    let xs = smooth3(&xs);
    let reversals = count_reversals(&xs, cfg.min_swing);
    let freq = reversals as f64 / 2.0 / span_t;
    let waving =
        reversals >= cfg.min_reversals && freq >= cfg.min_freq_hz && freq <= cfg.max_freq_hz;
    (waving, reversals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neurosight_common::Keypoint;
    use std::f64::consts::PI;

    const FPS: f64 = 30.0;

    fn kpt(x: f32, y: f32, score: f32) -> Keypoint {
        Keypoint { x, y, score }
    }

    /// A person standing facing the camera, arms down, in the middle of the image.
    fn base_keypoints() -> [Keypoint; 17] {
        let mut k = [kpt(0.5, 0.5, 0.0); 17];
        k[kp::NOSE] = kpt(0.5, 0.25, 0.9);
        k[kp::LEFT_EYE] = kpt(0.53, 0.22, 0.9);
        k[kp::RIGHT_EYE] = kpt(0.47, 0.22, 0.9);
        k[kp::LEFT_EAR] = kpt(0.56, 0.24, 0.9);
        k[kp::RIGHT_EAR] = kpt(0.44, 0.24, 0.9);
        k[kp::LEFT_SHOULDER] = kpt(0.6, 0.4, 0.9);
        k[kp::RIGHT_SHOULDER] = kpt(0.4, 0.4, 0.9);
        k[kp::LEFT_ELBOW] = kpt(0.65, 0.55, 0.9);
        k[kp::RIGHT_ELBOW] = kpt(0.35, 0.55, 0.9);
        k[kp::LEFT_WRIST] = kpt(0.65, 0.70, 0.9);
        k[kp::RIGHT_WRIST] = kpt(0.35, 0.70, 0.9);
        k[kp::LEFT_HIP] = kpt(0.57, 0.8, 0.9);
        k[kp::RIGHT_HIP] = kpt(0.43, 0.8, 0.9);
        k
    }

    fn standing() -> Option<Pose> {
        Some(Pose::new(base_keypoints()))
    }

    /// Left arm with the wrist swinging sideways at `freq` Hz, `amp` wide.
    fn wave_pose(t: f64, freq: f64, amp: f64, raised: bool) -> Option<Pose> {
        let mut k = base_keypoints();
        k[kp::LEFT_ELBOW] = kpt(0.65, 0.50, 0.9);
        let y = if raised { 0.35 } else { 0.62 };
        let x = 0.65 + amp * (2.0 * PI * freq * t).sin();
        k[kp::LEFT_WRIST] = kpt(x as f32, y, 0.9);
        Some(Pose::new(k))
    }

    fn run(poses: &[Option<Pose>]) -> Vec<Event> {
        let mut analyzer = PoseAnalyzer::new(PoseConfig::default());
        let mut events = Vec::new();
        for (i, p) in poses.iter().enumerate() {
            if let Some(e) = analyzer.process(p.as_ref(), i as f64 / FPS) {
                events.push(e);
            }
        }
        events
    }

    fn actions(events: &[Event]) -> Vec<Action> {
        events.iter().map(|e| e.action).collect()
    }

    #[test]
    fn nobody_produces_no_events() {
        assert!(run(&vec![None; 60]).is_empty());
    }

    #[test]
    fn standing_person_appears_once() {
        let events = run(&vec![standing(); 60]);
        assert_eq!(actions(&events), vec![Action::Appeared]);
        assert_eq!(events[0].zone, Some(Zone::Center));
        assert_eq!(events[0].distance, Some(Distance::Medium));
        assert!(events[0].facing_camera);
    }

    #[test]
    fn appear_wave_stop_leave() {
        let mut poses: Vec<Option<Pose>> = vec![None; 10];
        poses.extend((0..90).map(|i| wave_pose(i as f64 / FPS, 2.5, 0.12, true)));
        poses.extend(vec![wave_pose(0.0, 2.5, 0.0, true); 80]);
        poses.extend(vec![None; 50]);
        let events = run(&poses);
        assert_eq!(
            actions(&events),
            vec![
                Action::Appeared,
                Action::Waving,
                Action::Present,
                Action::Left
            ],
            "events: {:?}",
            events
        );
        assert!(events[1].facing_camera);
    }

    #[test]
    fn arm_swinging_low_is_not_waving() {
        let poses: Vec<Option<Pose>> = (0..90)
            .map(|i| wave_pose(i as f64 / FPS, 2.5, 0.12, false))
            .collect();
        assert_eq!(actions(&run(&poses)), vec![Action::Appeared]);
    }

    #[test]
    fn static_raised_hand_is_not_waving() {
        let poses = vec![wave_pose(0.0, 2.5, 0.0, true); 90];
        assert_eq!(actions(&run(&poses)), vec![Action::Appeared]);
    }

    #[test]
    fn slow_sway_is_not_waving() {
        let poses: Vec<Option<Pose>> = (0..150)
            .map(|i| wave_pose(i as f64 / FPS, 0.3, 0.12, true))
            .collect();
        assert_eq!(actions(&run(&poses)), vec![Action::Appeared]);
    }

    #[test]
    fn tiny_wiggle_is_not_waving() {
        let poses: Vec<Option<Pose>> = (0..90)
            .map(|i| wave_pose(i as f64 / FPS, 2.5, 0.02, true))
            .collect();
        assert_eq!(actions(&run(&poses)), vec![Action::Appeared]);
    }

    #[test]
    fn walking_sideways_is_not_waving() {
        let poses: Vec<Option<Pose>> = (0..150)
            .map(|i| {
                let dx = (0.2 * (2.0 * PI * 0.5 * i as f64 / FPS).sin()) as f32;
                let mut k = base_keypoints();
                for p in k.iter_mut() {
                    p.x += dx;
                }
                Some(Pose::new(k))
            })
            .collect();
        assert_eq!(actions(&run(&poses)), vec![Action::Appeared]);
    }

    #[test]
    fn small_shoulders_mean_far_away() {
        let mut k = base_keypoints();
        for p in k.iter_mut() {
            p.x = 0.5 + (p.x - 0.5) * 0.4;
        }
        let events = run(&vec![Some(Pose::new(k)); 10]);
        assert_eq!(events[0].distance, Some(Distance::Far));
    }

    #[test]
    fn hidden_face_means_not_facing_camera() {
        let mut k = base_keypoints();
        k[kp::NOSE].score = 0.1;
        k[kp::LEFT_EYE].score = 0.1;
        k[kp::RIGHT_EYE].score = 0.1;
        let events = run(&vec![Some(Pose::new(k)); 10]);
        assert!(!events[0].facing_camera);
    }

    /// Standing person whose nose is placed for the given head yaw / pitch (degrees).
    fn pose_with_head(yaw_deg: f64, pitch_deg: f64) -> Option<Pose> {
        let (hx, hy) = crate::head::tests::nose_offset(yaw_deg, pitch_deg);
        let mut k = base_keypoints();
        let span = 0.2_f32; // shoulders at x = 0.4 and 0.6
        let shoulder_w = span * 4.0 / 3.0;
        k[kp::NOSE] = kpt(0.5 + hx * span, 0.4 + hy * shoulder_w, 0.9);
        Some(Pose::new(k))
    }

    #[test]
    fn shaking_head_is_reported_through_the_pose_analyzer() {
        let mut poses: Vec<Option<Pose>> = vec![pose_with_head(0.0, 0.0); 30];
        poses.extend((0..120).map(|i| pose_with_head(30.0 * (2.0 * PI * 2.0 * i as f64 / FPS).sin(), 0.0)));
        let events = run(&poses);
        assert_eq!(
            actions(&events),
            vec![Action::Appeared, Action::ShakingHead],
            "events: {:?}",
            events
        );
    }

    #[test]
    fn nodding_head_is_reported_through_the_pose_analyzer() {
        let mut poses: Vec<Option<Pose>> = vec![pose_with_head(0.0, 0.0); 30];
        poses.extend((0..120).map(|i| pose_with_head(0.0, 12.0 * (2.0 * PI * 2.0 * i as f64 / FPS).sin())));
        let events = run(&poses);
        assert_eq!(
            actions(&events),
            vec![Action::Appeared, Action::Nodding],
            "events: {:?}",
            events
        );
    }

    #[test]
    fn turning_and_looking_are_reported_through_the_pose_analyzer() {
        let mut poses: Vec<Option<Pose>> = vec![pose_with_head(0.0, 0.0); 30];
        poses.extend(vec![pose_with_head(50.0, 0.0); 45]); // turn toward image right
        poses.extend(vec![pose_with_head(0.0, 0.0); 45]);
        poses.extend(vec![pose_with_head(0.0, -25.0); 45]); // look up
        poses.extend(vec![pose_with_head(0.0, 0.0); 45]);
        let events = run(&poses);
        assert_eq!(
            actions(&events),
            vec![Action::Appeared, Action::LookedRight, Action::LookedUp],
            "events: {:?}",
            events
        );
    }

    #[test]
    fn hand_held_out_is_reported_through_the_pose_analyzer() {
        let shoulder_w = 0.2_f32 * 4.0 / 3.0;
        let mut k = base_keypoints();
        // Right arm (person's right = image left): upper arm down, forearm toward the camera.
        k[kp::RIGHT_ELBOW] = kpt(0.4 + 0.05 * 0.2, 0.4 + 0.6 * shoulder_w, 0.9);
        k[kp::RIGHT_WRIST] = kpt(0.4 + 0.03 * 0.2, 0.4 + 0.55 * shoulder_w, 0.9);
        let events = run(&vec![Some(Pose::new(k)); 60]);
        assert_eq!(
            actions(&events),
            vec![Action::Appeared, Action::ReachingOut],
            "events: {:?}",
            events
        );
    }

    #[test]
    fn waving_survives_single_frame_dropouts() {
        let poses: Vec<Option<Pose>> = (0..120)
            .map(|i| {
                if i % 17 == 0 {
                    None
                } else {
                    wave_pose(i as f64 / FPS, 2.5, 0.12, true)
                }
            })
            .collect();
        let events = run(&poses);
        assert!(events.iter().any(|e| e.action == Action::Waving));
        assert!(events.iter().all(|e| e.action != Action::Left));
    }
}
