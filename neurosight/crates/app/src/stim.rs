//! Glue between events and the simulated stimulation:
//! event -> encoder -> scheduler (rest rules) -> player -> simulated phosphenes.
//!
//! SIMULATION ONLY: nothing here sends anything to a device.

use std::fs::{File, OpenOptions};
use std::io::Write;

use neurosight_common::Event;
use neurosight_encoder::{encode, SafetyLimits, Scheduler};
use neurosight_receiver_sim::{pattern_json, Phosphene, PhospheneModel, Player};

pub struct Stimulation {
    limits: SafetyLimits,
    scheduler: Scheduler,
    player: Player,
    model: PhospheneModel,
    log: Option<File>,
}

impl Stimulation {
    pub fn new(log_path: Option<&str>) -> Self {
        let limits = SafetyLimits::default();
        let log = log_path.and_then(|path| {
            match OpenOptions::new().create(true).append(true).open(path) {
                Ok(f) => Some(f),
                Err(e) => {
                    eprintln!("Could not open log file '{}': {}", path, e);
                    None
                }
            }
        });
        Self {
            scheduler: Scheduler::new(limits.clone()),
            limits,
            player: Player::new(),
            model: PhospheneModel::default(),
            log,
        }
    }

    /// Call for every event. Encodes it and queues the pattern.
    pub fn on_event(&mut self, event: &Event, t: f64) {
        if let Some(pattern) = encode(event, &self.limits) {
            self.scheduler.push(pattern, t);
        }
    }

    /// Call on every frame: starts queued patterns when allowed and plays due pulses.
    /// Returns the latest flash of light if a pulse was played during this call.
    pub fn tick(&mut self, t: f64) -> Option<Phosphene> {
        if let Some(pattern) = self.scheduler.poll(t) {
            println!(
                "[{:7.2}s]   stimulation pattern '{}' ({} pulses, {} ms)",
                t,
                pattern.name,
                pattern.pulses.len(),
                pattern.duration_ms()
            );
            if let Some(file) = self.log.as_mut() {
                if let Err(e) = writeln!(file, "{}", pattern_json(&pattern, t)) {
                    eprintln!("Could not write to the log file: {}", e);
                    self.log = None;
                }
            }
            self.player.start(&pattern, t);
        }
        let mut latest: Option<Phosphene> = None;
        for (number, total, pulse) in self.player.due(t) {
            let flash = self.model.percept(&pulse);
            latest = Some(flash);
            println!(
                "[{:7.2}s]     pulse {}/{}  coil (col {}, row {})  intensity {:.2}  ->  {}  [{}]",
                t,
                number,
                total,
                pulse.coil.col,
                pulse.coil.row,
                pulse.intensity,
                flash.describe(),
                flash.grid_inline()
            );
        }
        latest
    }
}
