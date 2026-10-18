# NeuroSight

A software-only simulation of a "vision to neural signal" system, written in **Rust**.

A computer webcam watches the world. The software works out what is happening
(for example: "someone on your left is waving at you"), converts that event into a
stimulation pattern, and shows what a person would receive on a **simulated** receiver.

Inspired by the haptic vision aid at https://7sense.ee (SuperBrain), which maps objects,
motion and gestures onto the forehead through touch.

**Status:** planning. See [TIMELINE.md](TIMELINE.md) for the 12-week plan.

---

## Quick start

The program watches the webcam and prints what is happening. It has two modes:

**Pose mode** (needs the model file, see [models/README.md](models/README.md)): a body-pose model
finds the person's shoulders, elbows and wrists. A raised hand swinging left and right is a wave.

```
[   0.40s] Someone appeared on the center of the image, at a medium distance.
[   1.60s] Someone is waving at you on the center of the image, at a medium distance (confidence 83%).
[   4.30s] Someone is in view on the center of the image, at a medium distance.
[   9.10s] The person left.
```

**Motion-only mode** (no model needed, used automatically if the model is missing): compares
frames and treats a fast left-right swing of the motion as a wave.

```
[   3.20s] Someone or something is moving on the left of the image (confidence 62%).
[   4.00s] Someone is waving on the left of the image (confidence 75%).
[   6.90s] Nothing is moving.
```

Requirements: Rust (https://rustup.rs, keep it up to date with `rustup update`) and a webcam.
The first build downloads ONNX Runtime (internet needed). On Linux the camera backend needs
Video4Linux; on macOS and Windows the terminal must be allowed to use the camera.

```
cargo test                                         # logic tests, no camera needed
cargo run --release -p neurosight-app              # pose mode if the model exists, else motion mode
cargo run --release -p neurosight-app -- --motion  # force motion-only mode
cargo run --release -p neurosight-app -- --debug   # also print numbers for tuning
cargo run --release -p neurosight-app -- --camera 1
```

If waving is missed or false alarms appear, run with `--debug` and adjust the thresholds in
`crates/events/src/lib.rs` (`Config::default`, motion mode) or `crates/events/src/pose.rs`
(`PoseConfig::default`, pose mode).

Current crates: `common`, `capture`, `perception`, `events`, `app`. The others in the layout
below (`encoder`, `receiver-sim`) come in later weeks.

Notes on accuracy:
- The camera image is squeezed to a square for the pose model. Left/right and up/down
  relationships are kept, which is all the wave logic uses.
- "Facing the camera" is a rough guess (nose and both eyes are visible).
- Distance is only near / medium / far, from the width of the shoulders in the image.
- Pose mode follows one person (the one the model finds most clearly).

Status: the detection logic was checked against synthetic scenarios, but the Rust code has
not been compiled on a real machine yet, and the pose model has not been run. Expect to fix
small build errors on the first `cargo build`, especially in `crates/capture` (`nokhwa`) and
`crates/perception` (`ort` 2.0 release candidates change their API between versions).

## Goal

Build and test, entirely in simulation, a full chain:

```
webcam -> perception -> event engine -> encoder -> simulated receiver -> human recognition test
```

Central research question: **how many different events can a person tell apart, and how fast
can they learn them, from stimulation patterns alone?**

## Scope and safety

- Simulation only. The "brain side" is a virtual display (tactile grid and/or simulated phosphenes).
- Magnetic stimulation (TMS-style) is modeled in software only: coil position, intensity and
  pulse timing are mapped to where a light patch (phosphene) would appear. Nothing in this
  project drives a real coil.
- Real TMS or any brain stimulation can cause seizures, burns or hearing damage and needs
  medical screening and clinical supervision. Do not build or use a real device from this project.
- Camera data stays on the local machine. No cloud upload.

## How it works

1. **Capture**: webcam frames in real time.
2. **Perception**: people, body keypoints, hands, rough distance, facing direction.
3. **Event engine**: turns perception into structured events:
   `{ type, direction, distance, confidence }`
   Example: `{ wave, left, 2.0 m, directed_at_me, 0.93 }`
4. **Encoder**: each event becomes a stimulation pattern ("haptic icon"):
   - position -> location on the grid / visual field
   - distance -> intensity
   - wave -> 2-3 Hz left-right sweep at the person's location
   - handshake reach -> pulse growing toward the center
5. **Simulated receiver**: virtual forehead grid (about 16x16) and a phosphene view.
   TMS model: coil position -> cortical patch -> phosphene location and brightness.
6. **Human test**: show only the pattern (never the camera feed) and measure recognition.

## Why Rust, and the trade-offs

Rust gives low and predictable latency, safe concurrency, and a clean path to real hardware later.
The vision and neuroscience ecosystem is smaller than Python's:

- MediaPipe, SimNIBS and pulse2percept have no Rust versions.
- Neural models run in Rust through **ONNX Runtime** (the `ort` crate), using ONNX exports of
  pose, hand and depth models.
- Anything that must use Python (training the gesture model, SimNIBS field precomputation) lives
  in `tools/` as **offline tooling only**. The real-time app is 100% Rust and loads the results
  as data files.

### Suggested crates (verify current versions before use)

| Need | Crate |
|---|---|
| Webcam capture | `nokhwa` |
| Neural network inference (ONNX) | `ort` |
| Arrays and images | `ndarray`, `image` |
| Real-time display / simulator UI | `macroquad` or `eframe` (egui) |
| Threads and channels | `crossbeam-channel` |
| Data formats | `serde`, `serde_json` |
| Logging and timing | `tracing` |
| Benchmarks | `criterion` |
| Optional Rust-native ML training | `burn` or `candle` |

## Project layout (Cargo workspace)

```
neurosight/
  Cargo.toml
  README.md
  TIMELINE.md
  crates/
    common/        shared types (Event, Pattern, Frame)
    capture/       webcam input
    perception/    pose, hands, depth via ONNX
    events/        wave / handshake / approach detection
    encoder/       event -> stimulation pattern
    receiver-sim/  virtual grid + phosphene + TMS model
    app/           real-time binary tying it together
  models/          ONNX model files
  data/            recorded clips, labels, precomputed field maps
  tools/           offline Python scripts (training, SimNIBS precompute)
  docs/            notes, results, test protocol
```

Rule: crates talk only through the types in `common`, so any stage can be swapped or rewritten
without touching the others.

## Success targets (end of week 12)

| Metric | Target |
|---|---|
| Wave detection accuracy | >= 90% |
| End-to-end latency (camera to pattern) | < 200 ms |
| Human recognition of events from patterns alone | >= 80% after a short training session |
| Distinct events encoded | 5 to 8 |

## Risks and notes

- Pattern design is the hardest part: detection is largely solved, learnable patterns are not.
  Test with real people early.
- Monocular depth from a webcam is rough: good for near/far, not precise distances.
- TMS reality check: the coil is large, so it can only produce coarse light patches, not
  detailed images. Keep the event code simple.
- Keep a tactile output as a comparison. If touch carries more events accurately than simulated
  phosphenes, that is a result in itself.
- The Rust ML ecosystem is younger than Python's: budget extra time for model export and `ort`.

## Positioning

- Implant-based visual prosthetics (PRIMA, Blindsight and others) are crowded and clinical-grade;
  this project does not compete there.
- 7Sense already claims wave and handshake-reach detection, so the wave itself is not novel.
- The open niche: an open-source, hardware-free simulator plus a tested study of which event
  encodings people can learn fastest. Useful to university labs and assistive-tech teams.
