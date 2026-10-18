# NeuroSight: 12-Week Timeline

Software-only simulation in Rust. Webcam in, simulated stimulation pattern out.
Overview and architecture: [README.md](README.md).

| Month | Weeks | Theme | Milestone |
|---|---|---|---|
| 1 | 1-4 | Perception: webcam to meaning | M1: console prints "Person on left, 2 m, waving at you" |
| 2 | 5-8 | Encoding and the simulated receiver | M2: wave at camera, see wave pattern live on virtual grid |
| 3 | 9-12 | Human testing, iteration, packaging | M3: final evaluation, demo and report |

---

## Month 1: Perception

### Week 1: Setup and scope
- [ ] Create the Cargo workspace (crates: common, capture, perception, events, encoder, receiver-sim, app)
- [ ] Webcam capture with `nokhwa`, show frames live in a window
- [ ] Measure baseline capture-to-display latency
- [ ] Fix the first event list: person appears, person leaves, waving, handshake reach, approaching
- [ ] Write down the metrics (accuracy, latency, human recognition rate)

**Done when:** live webcam window runs with a logged frame time.

### Week 2: Where and how far
- [ ] Export a pose/person model to ONNX and run it through `ort`
- [ ] Draw body keypoints over the live video
- [ ] Classify person position: left / center / right
- [ ] Add a monocular depth ONNX model for near / far

**Done when:** the app reports position and rough distance for one person in frame.

### Week 3: Gesture recognition
- [ ] Rule-based wave detector (wrist oscillating above the elbow)
- [ ] Record 150-200 short clips per gesture (wave, handshake reach, idle, other movement)
- [ ] Train a small keypoint-sequence model offline (in `tools/`), export to ONNX
- [ ] Run the trained model in Rust and compare against the rule-based detector

**Done when:** wave detection reaches at least 85% on held-out clips.

### Week 4: "Is it for me?"
- [ ] Estimate body orientation and head pose toward the camera
- [ ] Combine into the event engine: `{ type, direction, distance, directed_at_me, confidence }`
- [ ] Add event smoothing so events do not flicker
- [ ] Log events to a file for later review

**Milestone 1:** live console prints "Person on left, 2 m, waving at you" with under 200 ms delay.

---

## Month 2: Encoding and the simulated receiver

### Week 5: Simulated receiver and raw baseline
- [ ] Build the virtual forehead grid (about 16x16) in `receiver-sim`
- [ ] Baseline mode: depth and motion map straight to pixel intensity (7Sense-style)
- [ ] Define the `Pattern` type in `common` (frames of intensities over time)

**Done when:** moving in front of the webcam changes the grid in real time.

### Week 6: Semantic encoding
- [ ] Design a distinct pattern for each event:
  - wave: 2-3 Hz left-right sweep at the person's location
  - handshake reach: pulse growing toward the center
  - approach / leave: expanding / shrinking pulse
- [ ] Position -> grid location, distance -> intensity
- [ ] Informal test: 3-5 friends try to tell patterns apart

**Done when:** patterns are implemented and a first rough recognition rate is recorded.

### Week 7: Neural-level simulation
- [ ] Convert patterns to stimulation parameters (amplitude, pulse frequency, pulse width) with safety-style limits (maximum rate, rest between pulse trains)
- [ ] Primary path: tactile array view
- [ ] Stretch: TMS model. Precompute field maps offline (SimNIBS, in `tools/`), load them from `data/`, and map coil position and intensity to simulated phosphene location and brightness
- [ ] Phosphene view in the simulator

**Done when:** the same event can be shown as a tactile pattern and as a simulated phosphene pattern.

### Week 8: Close the loop
- [ ] Wire the full chain: webcam -> perception -> events -> encoder -> receiver
- [ ] Latency budget per stage; optimize the slowest
- [ ] Add benchmarks with `criterion`

**Milestone 2:** wave at the camera and the wave pattern appears live on the virtual grid, under 200 ms end to end.

---

## Month 3: Test, improve, package

### Week 9: Human recognition test
- [ ] Write the test protocol in `docs/`
- [ ] Test mode: subject sees only the pattern, never the camera feed
- [ ] About 5 events x 20 trials per person; record accuracy, reaction time, learning curve
- [ ] Run with at least 5 people

**Done when:** a results table and confusion matrix exist.

### Week 10: Iterate
- [ ] Redesign patterns that were mixed up in the confusion matrix
- [ ] Add events: pointing, thumbs up, walking toward you
- [ ] Handle multiple people and different lighting
- [ ] Re-run the human test on the improved patterns

**Done when:** recognition rate improves over week 9.

### Week 11: Hardware-ready and clean-up
- [ ] Define an output trait in `common` so the simulator can later be swapped for a real tactile or audio device
- [ ] Optimize inference (model size, threads)
- [ ] Document camera-data privacy (local only, no storage unless enabled)
- [ ] Clean the repo, add usage notes

**Done when:** the app runs from a single command and the output stage is swappable.

### Week 12: Final evaluation and demo
- [ ] Final benchmark against the success targets
- [ ] Record a demo video
- [ ] Write the report: what works, how many events people can tell apart, what TMS could realistically convey
- [ ] Draft the hardware-phase plan

**Milestone 3:** final evaluation, demo and report delivered.

---

## Success targets

| Metric | Target |
|---|---|
| Wave detection accuracy | >= 90% |
| End-to-end latency (camera to pattern) | < 200 ms |
| Human recognition of events from patterns alone | >= 80% after a short training session |
| Distinct events encoded | 5 to 8 |

## Buffer and risks

- Weeks 2-3 can slip because of ONNX model export and `ort` integration. If they do, shorten week 10's extra events.
- If the TMS stretch in week 7 takes too long, ship the tactile path first and move TMS to week 10.
- Start human testing in week 6, not week 9; pattern design is the hardest part.
