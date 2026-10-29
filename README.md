# NeuroSight

A software-only simulation of a "vision to neural signal" system, written in **Rust**.

A webcam watches a person. The program works out what they are doing ("someone on the left is
waving at you", "someone is nodding", "someone is holding out a hand"), turns each event into a
short stimulation pattern, and plays that pattern on a **simulated** receiver that shows the
flashes of light a person would see. Nothing is sent to any device.

Inspired by the haptic vision aid at https://7sense.ee (SuperBrain), which maps objects, motion
and gestures onto the forehead through touch.

```
webcam -> perception -> event engine -> encoder -> simulated receiver -> (human recognition test)
```

**Extension in progress (planned, not built yet):** an FPGA-based tactile output and smart-system
integration, so the same events can be felt on a 5 x 5 grid of low-voltage vibration motors. See
[FPGA extension](#fpga-extension-tactile-output-and-smart-system-integration-planned).

Central research question: **how many different events can a person tell apart, and how fast can
they learn them, from stimulation patterns alone?**

See [neurosight/TIMELINE.md](neurosight/TIMELINE.md) for the 12-week software plan.

## Status

| Part | State |
|---|---|
| Webcam capture, pose mode, gestures, text output | built; pose mode has been run on a real machine (Windows) |
| Stimulation simulation (encoder, scheduler, simulated receiver, log) | built; shown in the console output |
| Camera window and control panel (camera / mode / stim / mirror buttons) | newest addition; not yet confirmed on a real machine |
| Gesture thresholds | set against a simulated head and arm, not real recordings: expect to tune them |
| Human recognition test mode | not built yet |
| CAD concept models of TMS equipment | built as non-functional shells, see `tms_wearable_concept.scad`, `make_tms_wearable.py`, `tms_wearable_concept.stl` |
| FPGA tactile output: Rust side (`crates/tactile-link`: packet, pattern ROM, scheduler model, golden-vector export) | written, **not yet compiled or run**: run `cargo test -p neurosight-tactile-link` first |
| FPGA tactile output: RTL, UART link on a real board, vibration grid | **planned**, nothing written yet, see the FPGA section below |
| Hardware | none yet. Nothing in this project drives a coil. The planned FPGA output drives only low-voltage vibration motors |

## Quick start

You need Rust (https://rustup.rs, keep it current with `rustup update`) and a webcam. The first
build downloads libraries, including ONNX Runtime, so it needs internet.

1. **Get the pose model** (one time). Download `model.onnx` (about 9 MB, not the quantized
   versions) from
   https://huggingface.co/Xenova/movenet-singlepose-lightning/tree/main/onnx
   and put it in the `models` folder. Other ways to get it are in [neurosight/models/README.md](neurosight/models/README.md).
   Without it the program still runs, but only in motion-only mode.
2. **Run the tests** (no camera needed): `cargo test`
3. **Start the program:** `cargo run --release -p neurosight-app`

Windows, macOS and Linux all use the same commands. The terminal must be allowed to use the camera;
on Linux the camera backend needs Video4Linux.

### Command-line options

Put them after `--`, for example `cargo run --release -p neurosight-app -- --debug`.

| Option | Meaning |
|---|---|
| `--camera N` | camera number (default 0) |
| `--camera-off` | start with the camera off (turn it on in the window) |
| `--model PATH` | use this ONNX model file |
| `--motion` | start in motion-only mode instead of pose mode |
| `--no-stim` | start with the simulated stimulation off |
| `--log FILE` | append every stimulation pattern to FILE, one line of JSON each |
| `--mirror` | start with the camera picture flipped like a mirror |
| `--no-window` | console only: no window and no buttons |
| `--debug` | print numbers twice a second, for tuning |

## The window

The camera picture is on the left. Detected body points and lines are drawn on it (head cyan, arms
orange, torso green). The mode and state are on top, the latest event is at the bottom, and a small
5 x 5 grid labeled "seen" lights up where the simulated stimulation makes a flash. The control panel
on the right has buttons, which also work as keys:

| Button | Key | What it does |
|---|---|---|
| Camera ON / OFF | C | turns the camera off (a note replaces the picture) or back on |
| Pose mode | P | body-pose model. Loads the model the first time; if it cannot load, the panel says why and the mode stays as it was |
| Motion only | M | no model needed: still / moving / waving |
| Stim ON / OFF | S | simulated stimulation on or off |
| Mirror ON / OFF | R | flips the picture; the text still uses the camera's own left and right |

Esc, or closing the window, stops the program. If the camera cannot be opened at start-up, the
window still opens with the reason in the panel, so you can fix it and press the Camera button.
If the pose model fails while running, the program switches to motion-only mode and says so.

## Modes and gestures

**Pose mode** (needs the model) finds the person's shoulders, elbows, wrists, eyes and nose, and
recognizes:

| Gesture | Event text | How it is detected |
|---|---|---|
| Appearing / leaving | "Someone appeared ...", "The person left." | shoulders seen for 3 frames / missing for 1 s |
| Waving | "Someone is waving at you ..." | raised hand swinging left-right, 1-6 Hz |
| Handshake offer | "Someone is reaching out a hand (handshake offer) ..." | upper arm down, forearm pointing at the camera (looks short), wrist at chest-to-waist height, held still 0.6 s |
| Turning the head | "Someone turned their head toward the left / right of the image." | nose moves sideways relative to the shoulders |
| Looking up / down | "Someone is looking up." / "... looking down." | nose moves up / down relative to the shoulders |
| Nodding (yes) | "Someone is nodding (yes)." | nose swings up and down several times |
| Shaking head (no) | "Someone is shaking their head (no)." | nose swings left and right several times |

```
[   0.40s] Someone appeared on the center of the image, at a medium distance.
[   1.60s] Someone is waving at you on the center of the image, at a medium distance (confidence 83%).
[   4.30s] Someone is in view on the center of the image, at a medium distance.
[   9.10s] The person left.
```

**Motion-only mode** (no model; used automatically if the model is missing) compares frames and
treats a fast left-right swing of the motion as a wave. It reports only still / moving / waving.

```
[   3.20s] Someone or something is moving on the left of the image (confidence 62%).
[   4.00s] Someone is waving on the left of the image (confidence 75%).
[   6.90s] Nothing is moving.
```

**Which mode am I in?** The first lines printed say so. `POSE MODE: model ...` means the model
loaded. A block starting `*** MOTION-ONLY MODE: ...` means it did not, and gives the reason (model
not found, with the folders searched, or the model loader's own message). The debug lines differ
too: `state=... motion=...` is motion-only mode, `person=... head_offset(x/y)=...` is pose mode.
The model is looked for in `models/` as `movenet_lightning.onnx`, `model.onnx` or `movenet.onnx`, then
any `.onnx` file there. It tries four input layouts (NHWC / NCHW, int32 / float32) and prints which
one worked.

Things to know:
- Head turns are described from the camera's point of view ("toward the left of the image"), like
  the side-of-image text for people.
- Head gestures are measured from the nose relative to the shoulders and compared with a neutral
  head position learned in the first half second, so start by looking at the camera. Sit reasonably
  close (shoulders at least a third of the image width): from far away the nose moves only a few
  pixels and camera jitter can look like a nod.
- The handshake detector only sees "hand held out toward the camera". It cannot see a real
  handshake between two people, because it follows one person.
- "Facing the camera" is a rough guess (nose and both eyes visible). Distance is only near / medium
  / far, from the width of the shoulders. Pose mode follows one person.
- The camera image is squeezed to a square for the pose model. Left/right and up/down
  relationships are kept, which is all the gesture logic uses.

## Stimulation (simulation only)

Every event is turned into an abstract magnetic-stimulation pattern and played on a simulated
receiver that shows the flash of light (phosphene) each pulse would cause:

```
[   1.60s] Someone is waving at you on the center of the image, at a medium distance (confidence 83%).
[   1.63s]   stimulation pattern 'wave' (6 pulses, 1000 ms)
[   1.63s]     pulse 1/6  coil (col 3, row 1)  intensity 1.15  ->  flash 5 deg up, 5 deg left, brightness 87%  [...../..@../...../...../.....]
```

- **Encoder** (`crates/encoder`): each event gets its own short pulse pattern, designed in the
  visual field. Head events use the upper rows, arm events (wave, handshake offer) row 3, presence
  events (appeared, left) row 4. The column follows where the person is in the image and the
  strength follows how close they are. A wave alternates left and right about 2.5 times a second; a
  handshake offer is three growing pulses.
- **Retinotopy** (`crates/common`): converts a wanted phosphene spot into the virtual coil position
  that would make it. Each half of the visual cortex handles the opposite half of the visual field
  (a flash on the left needs a coil on the right side of the head), and the upper visual field is
  handled by the lower part of the cortex. This is very simplified.
- **Limits and scheduler:** at most 8 pulses per pattern, at least 100 ms apart, a strength cap, at
  least 2 seconds of rest between patterns, at most 12 patterns a minute, and a small queue.
  Patterns waiting more than 2 seconds are dropped (they describe something that is over), and a
  newer pattern of the same kind replaces an older waiting one. These are abstract limits for the
  simulation, not medical safety guidance.
- **Simulated receiver** (`crates/receiver-sim`): coil position -> visual-field spot, strength ->
  brightness (50% at the simulated threshold of 1.0).
- **Log:** `--log patterns.jsonl` records every pattern, for analysis or for the human recognition test.

## FPGA extension: tactile output and smart-system integration (planned)

Status: **design only.** None of the items below exist in this repository yet. Each is marked
with what is already available and what still has to be written.

**Goal:** turn the simulated stimulation into a real, safe, low-power wearable. The encoder's
5 x 5 patterns map directly onto a 5 x 5 grid of coin vibration motors (3 V), so the user feels
the event instead of seeing a simulated flash. The FPGA takes over the parts that benefit from
parallel, fixed-latency hardware and enforces the output limits in the circuit itself.

```
camera -> [FPGA] frame difference + wave detector --+
PC (Rust: MoveNet, events) --UART--> [FPGA] event FSM -> pattern ROM -> safety scheduler
ToF distance + IMU (I2C) -------------------------------+          -> 25-channel PWM
                                                                      -> 74HC595 + ULN2803
                                                                      -> 5 x 5 vibration grid
power: 18650 -> 3.3 V, INA219 current/power monitor     status: UART log / display
```

### What stays in Rust, what moves to the FPGA

| Function | Today (Rust) | Planned (FPGA) |
|---|---|---|
| Pose estimation (MoveNet) | `crates/perception` | stays on the PC |
| Event engine (wave, nod, handshake offer ...) | `crates/events` | stays on the PC; the FPGA also runs a hardware motion/wave detector so it works without the PC |
| Event -> pulse pattern | `crates/encoder` (`encode`) | pattern ROM generated from the Rust encoder |
| Limits and scheduler | `SafetyLimits`, `Scheduler` | circuit: max 8 pulses, 100 ms gap, intensity cap, 2 s rest, 12 per minute, queue of 3, 2 s expiry |
| Simulated receiver | `crates/receiver-sim` | replaced by the real vibration grid (the simulator stays as the reference) |

### How the existing code maps to hardware

- `Pulse { offset_ms, coil: CoilPos { col, row }, intensity }` -> one timed pulse on motor
  `row * 5 + col` (GRID = 5, so 25 channels). `intensity` (0 to `max_intensity` 1.5) becomes a PWM duty.
- `Action` (13 values), `Zone` (3) and `Distance` (3) fit in a few bits each, so an event is a very
  short packet.
- `--log patterns.jsonl` already writes every pattern; those lines are the **golden vectors** for
  testing the RTL against the Rust behavior.

### Event packet (UART)

Frozen in `docs/FPGA_SPEC.md` (stable ids, checksum, parser behaviour, scheduler rules, ROM format). Summary:

| Byte | Field |
|---|---|
| 0 | start byte |
| 1 | action id 1 to 8 (stable table in `crates/tactile-link/src/wire.rs`, not the enum order) |
| 2 | `Zone` (left / center / right) |
| 3 | `Distance` (near / medium / far); later replaced by the ToF reading |
| 4 | confidence 0-100 |
| 5 | checksum: XOR of bytes 1 to 4 |

### Verification plan

1. Export vectors from the Rust encoder and models: `cargo run -p neurosight-tactile-link --bin export-vectors -- fpga/vectors` (patterns, ROM image, packet streams, scheduler scenarios).
2. Simulate the RTL pattern ROM + scheduler with a testbench and compare against those lines,
   pulse by pulse (time offset, motor index, PWM level).
3. Assert the limits in the testbench (pulse count, 100 ms gap, 2 s rest, 12 per minute).
4. Repeat on the real board with a logic analyzer.

### Planned hardware (low voltage only)

FPGA board (for example Sipeed Tang Primer 20K Dock), OV7670 camera, 25 coin vibration motors,
74HC595 + ULN2803A output drivers, VL53L0X ToF distance sensor, MPU-6050 IMU, INA219 power
monitor, USB-UART, and a 2 x 18650 battery pack with a 3.3 V regulator.

### Target KPIs

| KPI | Target |
|---|---|
| End-to-end latency, camera to vibration | at most 200 ms (median) |
| Latency inside the FPGA, UART in to motor out | at most 1 ms |
| Image pipeline frame rate | at least 25 fps |
| Wave detection accuracy | at least 90% |
| RTL matches the Rust reference | 100% of test vectors |
| Human recognition of 5 tactile patterns | at least 80% |
| Average power | at most 1.5 W |
| Battery runtime | at least 6 h |
| Output limits respected | no violations |

### Roadmap

- [x] Freeze the event list (8 events) and the UART packet (proposal in `docs/FPGA_SPEC.md`; confirm or edit the `EVENTS` table)
- [ ] Rust: `tactile-link` crate written (packet, ROM, scheduler model, vector export, optional serial sender); compile, test, then hook the sender into `app` and add a tactile 5 x 5 view
- [ ] RTL: UART receiver, pattern ROM, scheduler, 25-channel PWM, with testbenches
- [ ] RTL: camera interface, frame differencing, wave detector
- [ ] Board bring-up: camera, motor grid and drivers, ToF, IMU, INA219
- [ ] Integration: PC + FPGA + sensors on battery
- [ ] Measure the KPIs and run the human recognition test on the tactile grid

Safety: the FPGA path drives **vibration motors at 3 V only**. The TMS-style pulses stay a
software simulation and nothing in this project should be used to build real stimulation hardware.

## Scope and safety

- **TMS-style stimulation is simulation only.** The patterns have no physical units, and nothing in
  this project sends them to a coil. The planned FPGA extension drives only low-voltage vibration motors.
- Real brain stimulation, including TMS, needs medical screening and clinical supervision and can
  cause seizures, burns or hearing damage. Do not build or use a real device from this project. This
  project does not design stimulation electronics.
- The CAD models (`tms_wearable_concept.scad`, `.stl`, `make_tms_wearable.py`) are non-functional shells for visualization.
- A magnetic coil stimulates a patch about a centimeter across, so a real device could only make a
  few coarse flashes, not detailed images. That is why the patterns are small and simple.
- Camera pictures stay on the local machine. Nothing is uploaded.

## Tuning and troubleshooting

| Problem | What to try |
|---|---|
| Output starts with `*** MOTION-ONLY MODE` | read the reason on that line; make sure the `.onnx` file is in `models/` |
| Camera does not open | check the camera number (`--camera 1`), that no other program uses it, and the camera permission for your terminal; the window still opens so you can press Camera later |
| Waving missed, or false waves | run with `--debug` and adjust `crates/events/src/pose.rs` (`PoseConfig::default`) or, in motion mode, `crates/events/src/lib.rs` (`Config::default`) |
| Head gestures missed, or false nods | tune `crates/events/src/head.rs` (`HeadConfig::default`); sit closer; look at the camera at the start |
| Handshake offer not recognized | tune `crates/events/src/reach.rs` (`ReachConfig::default`); hold the hand out steadily for about a second |
| Text in the window looks mirrored | tell the maintainer: the font bit order assumption may be wrong |
| Build errors | the `ort` crate (pinned to 2.0.0-rc.12) and `nokhwa` change between versions; keep Rust up to date and read the first error |

## How it is built

```
neurosight/
  Cargo.toml  README.md  TIMELINE.md
  crates/
    common/        shared types: frames, pose, events, stimulation patterns, retinotopy
    capture/       webcam input (nokhwa): gray, square color and preview frames
    perception/    pose estimation with MoveNet through ONNX Runtime (ort)
    events/        motion analyzer, pose analyzer, head gestures, handshake offer
    encoder/       event -> stimulation pattern, safety limits, scheduler
    receiver-sim/  simulated phosphenes, pattern player, JSON log lines
    app/           the program: engine, window (minifb), control panel, stimulation glue
  models/          ONNX model file (not included) and how to get it
  crates/tactile-link/   packet, pattern ROM, scheduler model, golden-vector export, serial sender (feature `serial`)
  (planned) fpga/                  RTL, testbenches, constraints, board notes
  (repo root)      tms_wearable_concept.scad / .stl / .png and make_tms_wearable.py: non-functional CAD shells
```

Crates talk only through the types in `common`, so any stage can be swapped or rewritten without
touching the others. Everything real-time is Rust; Python is used only for the CAD generator
scripts and, if needed, offline model conversion.

## Next steps

1. Tune the gesture thresholds on real recordings (`--debug`).
2. Build the human recognition test: show only the simulated flashes and measure how well people
   tell the events apart (accuracy, reaction time, learning curve).
3. Compare with a tactile output (a grid of small vibration motors on the forehead, like 7Sense).
   Low-voltage and much safer to build; the 5 x 5 patterns already fit it. The FPGA extension above
   is the plan for this step.
4. Optional: a more realistic phosphene model (precomputed with SimNIBS / pulse2percept offline).

## Positioning

- Implant-based visual prosthetics (PRIMA, Blindsight and others) are crowded and clinical-grade;
  this project does not compete there.
- 7Sense already claims wave and handshake-reach detection, so the wave itself is not novel.
- The open niche: an open-source, hardware-free simulator plus a tested study of which event
  encodings people can learn fastest. Useful to university labs and assistive-tech teams.
