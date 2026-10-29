//! Integer-millisecond model of the safety scheduler (what the RTL implements).
//!
//! It mirrors `neurosight_encoder::Scheduler` rule for rule, but in whole milliseconds so the
//! RTL (which counts ms ticks) can be compared with it exactly, including on the boundaries:
//!
//! * `push`: a waiting entry with the same action is replaced; if the queue is full
//!   (`max_queue`) the oldest entry is dropped; then the new entry is appended.
//! * `poll(now)`:
//!   1. drop entries with `now - queued_at > max_age`  (exactly `max_age` is still kept)
//!   2. nothing if the queue is empty or `now < rest_until`  (exactly `rest_until` is allowed)
//!   3. forget starts with `now - start >= 60000`  (exactly 60 s ago is forgotten)
//!   4. nothing if `max_patterns_per_minute` starts remain
//!   5. start the oldest entry; `rest_until = now + duration + min_pattern_gap`
//!
//! In a run, pushes at millisecond `t` are applied before the poll at `t`.

use std::collections::VecDeque;

use neurosight_encoder::SafetyLimits;

use crate::rom::pattern_for;
use crate::wire::{Key, Packet};

struct Queued {
    key: Key,
    queued_at: i64,
    duration_ms: i64,
}

pub struct MsScheduler {
    limits: SafetyLimits,
    queue: VecDeque<Queued>,
    rest_until_ms: i64,
    starts: VecDeque<i64>,
}

impl MsScheduler {
    pub fn new(limits: SafetyLimits) -> Self {
        Self {
            limits,
            queue: VecDeque::new(),
            rest_until_ms: i64::MIN,
            starts: VecDeque::new(),
        }
    }

    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    pub fn push(&mut self, packet: &Packet, now_ms: i64) {
        let key = packet.key;
        self.queue.retain(|q| q.key.action != key.action);
        while self.queue.len() >= self.limits.max_queue.max(1) {
            self.queue.pop_front();
        }
        let duration_ms = pattern_for(&key, &self.limits).duration_ms() as i64;
        self.queue.push_back(Queued {
            key,
            queued_at: now_ms,
            duration_ms,
        });
    }

    pub fn poll(&mut self, now_ms: i64) -> Option<Key> {
        let max_age_ms = (self.limits.max_age_s * 1000.0).round() as i64;
        self.queue.retain(|q| now_ms - q.queued_at <= max_age_ms);
        if self.queue.is_empty() || now_ms < self.rest_until_ms {
            return None;
        }
        while self.starts.front().map_or(false, |&s| now_ms - s >= 60_000) {
            self.starts.pop_front();
        }
        if self.starts.len() >= self.limits.max_patterns_per_minute {
            return None;
        }
        let entry = self.queue.pop_front()?;
        let gap_ms = (self.limits.min_pattern_gap_s * 1000.0).round() as i64;
        self.rest_until_ms = now_ms + entry.duration_ms + gap_ms;
        self.starts.push_back(now_ms);
        Some(entry.key)
    }
}

/// Run a scenario: `inputs` are `(time_ms, packet)` sorted by time; the scheduler is polled
/// every millisecond from 0 to `end_ms` inclusive. Returns `(start_time_ms, key)` per started
/// pattern.
pub fn run_schedule(
    inputs: &[(i64, Packet)],
    end_ms: i64,
    limits: &SafetyLimits,
) -> Vec<(i64, Key)> {
    let mut scheduler = MsScheduler::new(limits.clone());
    let mut starts = Vec::new();
    for t in 0..=end_ms {
        for (at, packet) in inputs.iter() {
            if *at == t {
                scheduler.push(packet, t);
            }
        }
        if let Some(key) = scheduler.poll(t) {
            starts.push((t, key));
        }
    }
    starts
}

#[cfg(test)]
mod tests {
    use super::*;
    use neurosight_common::{Action, Distance, Zone};
    use neurosight_encoder::{encode, Scheduler};

    fn packet(action: Action, zone: Zone) -> Packet {
        Packet::new(
            Key {
                action,
                zone,
                distance: Distance::Medium,
            },
            90,
        )
        .unwrap()
    }

    #[test]
    fn rest_boundary_is_inclusive() {
        // A wave lasts 1000 ms, so the next pattern may start at 0 + 1000 + 2000 = 3000.
        let limits = SafetyLimits::default();
        let mut s = MsScheduler::new(limits);
        s.push(&packet(Action::Waving, Zone::Center), 0);
        assert!(s.poll(0).is_some());
        s.push(&packet(Action::Appeared, Zone::Center), 2000);
        assert!(s.poll(2999).is_none());
        let started = s.poll(3000).expect("allowed exactly at the end of the rest");
        assert_eq!(started.action, Action::Appeared);
    }

    #[test]
    fn expiry_boundary_is_inclusive() {
        let mut s = MsScheduler::new(SafetyLimits::default());
        s.push(&packet(Action::Waving, Zone::Center), 0);
        assert!(s.poll(0).is_some());
        s.push(&packet(Action::Left, Zone::Center), 100);
        assert!(s.poll(2100).is_none());
        assert_eq!(s.queued(), 1, "age exactly 2000 ms is kept");
        assert!(s.poll(2101).is_none());
        assert_eq!(s.queued(), 0, "age 2001 ms is dropped");
        assert!(s.poll(3000).is_none());
    }

    #[test]
    fn same_action_replaces_and_full_queue_drops_oldest() {
        let mut s = MsScheduler::new(SafetyLimits::default());
        s.push(&packet(Action::Nodding, Zone::Left), 0);
        s.push(&packet(Action::Nodding, Zone::Right), 1);
        assert_eq!(s.queued(), 1);
        s.push(&packet(Action::Waving, Zone::Center), 2);
        s.push(&packet(Action::Appeared, Zone::Center), 3);
        s.push(&packet(Action::Left, Zone::Center), 4);
        assert_eq!(s.queued(), 3);
        // Oldest (the nod) was dropped, so the first start is the wave.
        assert_eq!(s.poll(5).unwrap().action, Action::Waving);
    }

    #[test]
    fn at_most_twelve_patterns_per_minute() {
        // "appeared" lasts 150 ms, so one can start every 2150 ms; the 13th has to wait until
        // the first start is 60 s old.
        let limits = SafetyLimits::default();
        let inputs: Vec<(i64, Packet)> = (0..=27)
            .map(|k| (k * 2150, packet(Action::Appeared, Zone::Center)))
            .collect();
        let starts = run_schedule(&inputs, 61_000, &limits);
        assert_eq!(starts.len(), 13);
        for (k, (t, _)) in starts.iter().take(12).enumerate() {
            assert_eq!(*t, k as i64 * 2150);
        }
        assert_eq!(starts[12].0, 60_000);
    }

    #[test]
    fn agrees_with_the_float_scheduler_away_from_boundaries() {
        let limits = SafetyLimits::default();
        let wave = packet(Action::Waving, Zone::Center);
        let appeared = packet(Action::Appeared, Zone::Center);

        let mut float = Scheduler::new(limits.clone());
        float.push(encode(&wave.to_event(), &limits).unwrap(), 0.0);
        assert!(float.poll(0.0).is_some());
        float.push(encode(&appeared.to_event(), &limits).unwrap(), 2.0);
        let float_early = float.poll(2.5).is_some();
        let float_late = float.poll(3.5).is_some();

        let mut ms = MsScheduler::new(limits);
        ms.push(&wave, 0);
        assert!(ms.poll(0).is_some());
        ms.push(&appeared, 2000);
        let ms_early = ms.poll(2500).is_some();
        let ms_late = ms.poll(3500).is_some();

        assert_eq!((float_early, float_late), (ms_early, ms_late));
        assert_eq!((ms_early, ms_late), (false, true));
    }
}
