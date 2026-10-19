//! Encoder: turns an `Event` into a stimulation `Pattern` (SIMULATION ONLY).
//!
//! The patterns are designed in "phosphene space": which spot in the visual field the person
//! should see flash. The retinotopy table in `neurosight-common` then converts each spot to
//! the virtual coil position that would produce it.
//!
//! Layout of the 5 x 5 visual field used for the codes (row 0 = top):
//!
//! ```text
//!   rows 0-2   head events      (looking up / down, turning, nodding, shaking)
//!   row  3     arm events       (waving, reaching out a hand)
//!   row  4     presence events  (appeared, present, left, moving)
//! ```
//!
//! The column of a person's events follows where they are in the camera image
//! (left / center / right), and the strength follows how close they are.
//!
//! The patterns have no physical units and nothing here talks to a device.
//! `SafetyLimits` are abstract limits for the simulation (they keep patterns short and
//! spaced out so they stay realistic); they are NOT medical safety guidance.

use std::collections::VecDeque;

use neurosight_common::{
    retinotopy, Action, CoilPos, Distance, Event, Pattern, Pulse, Zone, GRID,
};

/// Limits applied to every pattern and to the pattern schedule.
#[derive(Clone, Debug)]
pub struct SafetyLimits {
    pub max_pulses_per_pattern: usize,
    /// Minimum time between two pulses (100 ms = at most 10 pulses per second).
    pub min_pulse_gap_ms: u32,
    /// Highest relative strength (1.0 = simulated phosphene threshold).
    pub max_intensity: f32,
    /// Rest between the end of one pattern and the start of the next.
    pub min_pattern_gap_s: f64,
    pub max_patterns_per_minute: usize,
    /// Patterns waiting for their turn; the oldest are dropped first.
    pub max_queue: usize,
    /// A waiting pattern older than this many seconds is dropped: it describes something
    /// that is long over.
    pub max_age_s: f64,
}

impl Default for SafetyLimits {
    fn default() -> Self {
        Self {
            max_pulses_per_pattern: 8,
            min_pulse_gap_ms: 100,
            max_intensity: 1.5,
            min_pattern_gap_s: 2.0,
            max_patterns_per_minute: 12,
            max_queue: 3,
            max_age_s: 2.0,
        }
    }
}

impl SafetyLimits {
    /// Make a pattern obey the limits: cut extra pulses, cap the strength, and space
    /// pulses at least `min_pulse_gap_ms` apart.
    pub fn enforce(&self, mut pattern: Pattern) -> Pattern {
        pattern.pulses.truncate(self.max_pulses_per_pattern);
        let mut previous: Option<u32> = None;
        for pulse in pattern.pulses.iter_mut() {
            pulse.intensity = pulse.intensity.clamp(0.0, self.max_intensity);
            if let Some(prev_ms) = previous {
                let earliest = prev_ms + self.min_pulse_gap_ms;
                if pulse.offset_ms < earliest {
                    pulse.offset_ms = earliest;
                }
            }
            previous = Some(pulse.offset_ms);
        }
        pattern
    }

    /// Check that a pattern obeys the limits.
    pub fn validate(&self, pattern: &Pattern) -> Result<(), String> {
        if pattern.pulses.is_empty() {
            return Err("pattern has no pulses".to_string());
        }
        if pattern.pulses.len() > self.max_pulses_per_pattern {
            return Err(format!("too many pulses: {}", pattern.pulses.len()));
        }
        let mut previous: Option<u32> = None;
        for pulse in &pattern.pulses {
            if pulse.intensity > self.max_intensity + 1e-6 {
                return Err(format!("intensity {} is above the limit", pulse.intensity));
            }
            if pulse.coil.col >= GRID || pulse.coil.row >= GRID {
                return Err("coil position is outside the grid".to_string());
            }
            if let Some(prev_ms) = previous {
                if pulse.offset_ms < prev_ms + self.min_pulse_gap_ms {
                    return Err("pulses are too close together".to_string());
                }
            }
            previous = Some(pulse.offset_ms);
        }
        Ok(())
    }
}

/// Visual-field column for a place in the camera image.
fn zone_col(zone: Option<Zone>) -> u8 {
    match zone {
        Some(Zone::Left) => 1,
        Some(Zone::Right) => 3,
        _ => 2,
    }
}

/// Relative strength for a distance (closer = stronger).
fn base_intensity(distance: Option<Distance>) -> f32 {
    match distance {
        Some(Distance::Near) => 1.30,
        Some(Distance::Far) => 1.05,
        _ => 1.15,
    }
}

/// One pulse at a phosphene cell (visual-field column / row).
fn pulse(offset_ms: u32, col: u8, row: u8, intensity: f32) -> Pulse {
    let coil: CoilPos = retinotopy::coil_for_phosphene(col, row);
    Pulse {
        offset_ms,
        coil,
        intensity,
    }
}

/// Pulses at the given phosphene cells, `gap_ms` apart, all with the same strength.
fn sequence(cells: &[(u8, u8)], gap_ms: u32, intensity: f32) -> Vec<Pulse> {
    cells
        .iter()
        .enumerate()
        .map(|(i, &(col, row))| pulse(i as u32 * gap_ms, col, row, intensity))
        .collect()
}

/// Turn an event into a pattern. Returns None for events that need no stimulation.
pub fn encode(event: &Event, limits: &SafetyLimits) -> Option<Pattern> {
    let col = zone_col(event.zone);
    let base = base_intensity(event.distance);
    let head_strength = 1.2;

    let (name, pulses): (&'static str, Vec<Pulse>) = match event.action {
        Action::Still => return None,
        Action::Moving => ("moving", sequence(&[(col, 4)], 0, base)),
        Action::Present => ("present", sequence(&[(col, 4)], 0, base - 0.1)),
        Action::Appeared => ("appeared", sequence(&[(col, 4), (col, 4)], 150, base)),
        Action::Left => (
            "left",
            vec![
                pulse(0, 2, 4, 1.25),
                pulse(200, 2, 4, 1.12),
                pulse(400, 2, 4, 1.0),
            ],
        ),
        Action::Waving => {
            // Alternate left and right of the person's column, about 2.5 times a second.
            let c = col.clamp(1, 3);
            let cells: Vec<(u8, u8)> = (0..6)
                .map(|i| if i % 2 == 0 { (c - 1, 3) } else { (c + 1, 3) })
                .collect();
            ("wave", sequence(&cells, 200, base))
        }
        Action::ReachingOut => {
            // Three pulses that grow stronger.
            let c = col.clamp(1, 3);
            (
                "reach",
                vec![
                    pulse(0, c, 3, base - 0.15),
                    pulse(250, c, 3, base),
                    pulse(500, c, 3, base + 0.15),
                ],
            )
        }
        Action::LookedLeft => (
            "head-left",
            sequence(&[(3, 1), (2, 1), (1, 1)], 150, head_strength),
        ),
        Action::LookedRight => (
            "head-right",
            sequence(&[(1, 1), (2, 1), (3, 1)], 150, head_strength),
        ),
        Action::LookedUp => (
            "head-up",
            sequence(&[(2, 2), (2, 1), (2, 0)], 150, head_strength),
        ),
        Action::LookedDown => (
            "head-down",
            sequence(&[(2, 0), (2, 1), (2, 2)], 150, head_strength),
        ),
        Action::Nodding => {
            let cells: Vec<(u8, u8)> = (0..6)
                .map(|i| if i % 2 == 0 { (2, 0) } else { (2, 2) })
                .collect();
            ("nod", sequence(&cells, 200, head_strength))
        }
        Action::ShakingHead => {
            let cells: Vec<(u8, u8)> = (0..6)
                .map(|i| if i % 2 == 0 { (1, 1) } else { (3, 1) })
                .collect();
            ("shake", sequence(&cells, 200, head_strength))
        }
    };

    Some(limits.enforce(Pattern {
        name,
        action: event.action,
        pulses,
    }))
}

/// Decides when queued patterns may start, so stimulation stays sparse.
pub struct Scheduler {
    limits: SafetyLimits,
    /// Waiting patterns with the time they were queued.
    queue: VecDeque<(Pattern, f64)>,
    rest_until_s: f64,
    starts: VecDeque<f64>,
}

impl Scheduler {
    pub fn new(limits: SafetyLimits) -> Self {
        Self {
            limits,
            queue: VecDeque::new(),
            rest_until_s: f64::NEG_INFINITY,
            starts: VecDeque::new(),
        }
    }

    pub fn limits(&self) -> &SafetyLimits {
        &self.limits
    }

    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    /// Add a pattern to the queue (`now_s` = current time in seconds).
    /// A newer pattern of the same kind replaces an older waiting one. If the queue is
    /// still full, the oldest waiting pattern is dropped.
    pub fn push(&mut self, pattern: Pattern, now_s: f64) {
        self.queue.retain(|(p, _)| p.action != pattern.action);
        while self.queue.len() >= self.limits.max_queue.max(1) {
            self.queue.pop_front();
        }
        self.queue.push_back((pattern, now_s));
    }

    /// Ask for the next pattern to start now (time in seconds). Returns None while the
    /// system must rest or the per-minute limit is reached.
    pub fn poll(&mut self, now_s: f64) -> Option<Pattern> {
        // Forget patterns that waited too long.
        let max_age = self.limits.max_age_s;
        self.queue.retain(|(_, queued_at)| now_s - *queued_at <= max_age);
        if self.queue.is_empty() || now_s < self.rest_until_s {
            return None;
        }
        while self.starts.front().map_or(false, |&s| now_s - s >= 60.0) {
            self.starts.pop_front();
        }
        if self.starts.len() >= self.limits.max_patterns_per_minute {
            return None;
        }
        let (pattern, _) = self.queue.pop_front()?;
        self.rest_until_s =
            now_s + pattern.duration_ms() as f64 / 1000.0 + self.limits.min_pattern_gap_s;
        self.starts.push_back(now_s);
        Some(pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_ACTIONS: [Action; 12] = [
        Action::Moving,
        Action::Appeared,
        Action::Present,
        Action::Left,
        Action::ReachingOut,
        Action::LookedLeft,
        Action::LookedRight,
        Action::LookedUp,
        Action::LookedDown,
        Action::Nodding,
        Action::ShakingHead,
        Action::Waving,
    ];

    fn event(action: Action, zone: Option<Zone>, distance: Option<Distance>) -> Event {
        Event {
            time_s: 0.0,
            action,
            zone,
            distance,
            facing_camera: true,
            confidence: 0.9,
        }
    }

    #[test]
    fn still_needs_no_stimulation() {
        let limits = SafetyLimits::default();
        assert!(encode(&event(Action::Still, None, None), &limits).is_none());
    }

    #[test]
    fn every_pattern_obeys_the_limits() {
        let limits = SafetyLimits::default();
        let zones = [None, Some(Zone::Left), Some(Zone::Center), Some(Zone::Right)];
        let distances = [
            None,
            Some(Distance::Near),
            Some(Distance::Medium),
            Some(Distance::Far),
        ];
        for action in ALL_ACTIONS {
            for zone in zones {
                for distance in distances {
                    let pattern = encode(&event(action, zone, distance), &limits)
                        .unwrap_or_else(|| panic!("no pattern for {:?}", action));
                    assert_eq!(pattern.action, action);
                    limits
                        .validate(&pattern)
                        .unwrap_or_else(|e| panic!("{:?} {:?} {:?}: {}", action, zone, distance, e));
                }
            }
        }
    }

    /// The part of a pattern a person would have to recognize: where and when, ignoring
    /// small strength differences.
    fn signature(p: &Pattern) -> Vec<(u32, u8, u8)> {
        p.pulses
            .iter()
            .map(|x| (x.offset_ms, x.coil.col, x.coil.row))
            .collect()
    }

    #[test]
    fn different_events_have_different_patterns() {
        let limits = SafetyLimits::default();
        // Presence events (moving / present) intentionally share a spot; they never occur
        // in the same mode, so leave "moving" out of this comparison.
        let actions: Vec<Action> = ALL_ACTIONS
            .iter()
            .copied()
            .filter(|a| *a != Action::Moving)
            .collect();
        let patterns: Vec<(Action, Vec<(u32, u8, u8)>, Vec<u32>)> = actions
            .iter()
            .map(|&a| {
                let p = encode(
                    &event(a, Some(Zone::Center), Some(Distance::Medium)),
                    &limits,
                )
                .unwrap();
                let strengths = p
                    .pulses
                    .iter()
                    .map(|x| (x.intensity * 100.0).round() as u32)
                    .collect();
                (a, signature(&p), strengths)
            })
            .collect();
        for i in 0..patterns.len() {
            for j in (i + 1)..patterns.len() {
                assert!(
                    patterns[i].1 != patterns[j].1 || patterns[i].2 != patterns[j].2,
                    "{:?} and {:?} have the same pattern",
                    patterns[i].0,
                    patterns[j].0
                );
            }
        }
    }

    #[test]
    fn person_on_the_left_is_seen_in_the_left_visual_field() {
        let limits = SafetyLimits::default();
        let e = event(Action::Appeared, Some(Zone::Left), Some(Distance::Medium));
        let pattern = encode(&e, &limits).unwrap();
        for pulse in &pattern.pulses {
            let (col, _row) = retinotopy::phosphene_for_coil(pulse.coil);
            assert_eq!(col, 1, "left zone should use visual-field column 1");
            // ... which a coil on the right side of the head produces.
            assert!(pulse.coil.col > 2);
        }
    }

    #[test]
    fn closer_people_get_stronger_pulses() {
        let limits = SafetyLimits::default();
        let near = encode(
            &event(Action::Appeared, Some(Zone::Center), Some(Distance::Near)),
            &limits,
        )
        .unwrap();
        let far = encode(
            &event(Action::Appeared, Some(Zone::Center), Some(Distance::Far)),
            &limits,
        )
        .unwrap();
        assert!(near.pulses[0].intensity > far.pulses[0].intensity);
    }

    #[test]
    fn wave_alternates_sides_about_2_5_times_a_second() {
        let limits = SafetyLimits::default();
        let e = event(Action::Waving, Some(Zone::Center), Some(Distance::Medium));
        let p = encode(&e, &limits).unwrap();
        assert_eq!(p.pulses.len(), 6);
        assert_eq!(p.pulses[1].offset_ms - p.pulses[0].offset_ms, 200);
        assert_ne!(p.pulses[0].coil, p.pulses[1].coil);
        assert_eq!(p.pulses[0].coil, p.pulses[2].coil);
    }

    #[test]
    fn enforce_caps_strength_count_and_spacing() {
        let limits = SafetyLimits::default();
        let cell = CoilPos { col: 0, row: 0 };
        let pulses: Vec<Pulse> = (0..12)
            .map(|i| Pulse {
                offset_ms: i * 10, // far too close together
                coil: cell,
                intensity: 3.0, // far too strong
            })
            .collect();
        let fixed = limits.enforce(Pattern {
            name: "test",
            action: Action::Waving,
            pulses,
        });
        assert!(limits.validate(&fixed).is_ok(), "{:?}", fixed);
        assert_eq!(fixed.pulses.len(), limits.max_pulses_per_pattern);
    }

    #[test]
    fn validate_rejects_bad_patterns() {
        let limits = SafetyLimits::default();
        let p = Pattern {
            name: "empty",
            action: Action::Waving,
            pulses: vec![],
        };
        assert!(limits.validate(&p).is_err());
    }

    fn some_pattern() -> Pattern {
        encode(
            &event(Action::Waving, Some(Zone::Center), None),
            &SafetyLimits::default(),
        )
        .unwrap()
    }

    #[test]
    fn scheduler_enforces_rest_between_patterns() {
        let mut s = Scheduler::new(SafetyLimits::default());
        let nod = || {
            encode(
                &event(Action::Nodding, Some(Zone::Center), None),
                &SafetyLimits::default(),
            )
            .unwrap()
        };
        s.push(some_pattern(), 0.0);
        assert!(s.poll(0.0).is_some());
        // The wave lasts 1.0 s (last pulse at 1000 ms), then 2 s of rest.
        s.push(nod(), 1.5);
        assert!(s.poll(1.5).is_none());
        assert!(s.poll(2.9).is_none());
        assert_eq!(s.poll(3.1).map(|p| p.action), Some(Action::Nodding));
        assert!(s.poll(100.0).is_none()); // queue is empty
    }

    #[test]
    fn scheduler_drops_the_oldest_when_the_queue_is_full() {
        let limits = SafetyLimits {
            max_queue: 2,
            ..SafetyLimits::default()
        };
        let mut s = Scheduler::new(limits);
        let mk = |a: Action| {
            encode(&event(a, Some(Zone::Center), None), &SafetyLimits::default()).unwrap()
        };
        s.push(mk(Action::Appeared), 0.0);
        s.push(mk(Action::Waving), 0.0);
        s.push(mk(Action::Nodding), 0.0);
        assert_eq!(s.queued(), 2);
        assert_eq!(s.poll(0.0).unwrap().action, Action::Waving);
    }

    #[test]
    fn newer_pattern_of_the_same_kind_replaces_the_waiting_one() {
        let mut s = Scheduler::new(SafetyLimits::default());
        s.push(some_pattern(), 0.0);
        s.push(some_pattern(), 0.5);
        assert_eq!(s.queued(), 1);
    }

    #[test]
    fn stale_patterns_are_dropped() {
        let mut s = Scheduler::new(SafetyLimits::default());
        s.push(some_pattern(), 0.0);
        assert!(s.poll(5.0).is_none());
        assert_eq!(s.queued(), 0);
    }

    #[test]
    fn scheduler_caps_patterns_per_minute() {
        let limits = SafetyLimits {
            max_patterns_per_minute: 3,
            min_pattern_gap_s: 0.0,
            ..SafetyLimits::default()
        };
        let mut s = Scheduler::new(limits);
        let mut started = 0;
        let mut t = 0.0;
        while t < 50.0 {
            s.push(some_pattern(), t);
            if s.poll(t).is_some() {
                started += 1;
            }
            t += 2.0;
        }
        assert_eq!(started, 3);
        // After a minute the cap opens again.
        s.push(some_pattern(), 61.0);
        assert!(s.poll(61.0).is_some());
    }
}
