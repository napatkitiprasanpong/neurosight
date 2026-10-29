# NeuroSight-FPGA interface specification (v1 proposal)

Everything here is implemented and tested in `crates/tactile-link`; the RTL must match it.
Vectors for the testbenches: `cargo run -p neurosight-tactile-link --bin export-vectors -- fpga/vectors`.

## 1. Events

| id | event | wire name |
|---|---|---|
| 1 | Appeared | appeared |
| 2 | Left | left |
| 3 | Waving | waving |
| 4 | ReachingOut | reaching_out |
| 5 | Nodding | nodding |
| 6 | ShakingHead | shaking_head |
| 7 | LookedLeft | looked_left |
| 8 | LookedRight | looked_right |

Id 0 and ids above 8 are invalid. To change the set, edit `EVENTS` in `wire.rs` and re-export.
For the volunteer test the first five (waving, reaching out, nodding, shaking head, appeared) are enough.

## 2. UART packet (PC -> FPGA), 6 bytes

| byte | field | values |
|---|---|---|
| 0 | start | `0xA5` |
| 1 | action id | 1..=8 |
| 2 | zone | 0 left, 1 center, 2 right |
| 3 | distance | 0 near, 1 medium, 2 far |
| 4 | confidence | 0..=100 |
| 5 | checksum | XOR of bytes 1..=4 |

Baud rate is not fixed yet (115200 8N1 is enough: a packet takes about 0.5 ms).

**Decode order and error codes:** length (1), start byte (2), checksum (3), action (4), zone (5),
distance (6), confidence (7). A corrupted field therefore normally shows up as a checksum error.

**Receiver behaviour (byte stream):** while idle, any byte other than `0xA5` is ignored without an
error. After `0xA5` the receiver collects 6 bytes in total and decodes. A bad frame is dropped and the
receiver goes idle; it does not re-scan the dropped bytes. (No inter-byte timeout in v1; add one if
needed and add a vector for it.)

## 3. Pattern ROM

Key = (action, zone, distance); confidence does not select a pattern. 8 x 3 x 3 = 72 keys,
9 words per key. Word address = `(((id - 1) * 3 + zone) * 3 + distance) * 9 + word`.

* word 0: bits 31:16 duration in ms (offset of the last pulse), bits 7:0 pulse count (1..=8)
* words 1..=8: bits 31:16 pulse offset in ms, bits 15:8 motor index (0..=24), bits 7:0 PWM duty

Motor index = `row * 5 + col` of the **visual-field cell** (col 0 = wearer's left, row 0 = top).
The encoder's coil coordinates are mirrored for the TMS simulation, so they are converted back with
`retinotopy::phosphene_for_coil`. Duty = `round(intensity / 1.5 * 255)`; the RTL only reads the ROM.
Image: `pattern_rom.hex` (`$readmemh`, 648 words).

## 4. Safety scheduler (integer milliseconds)

Model: `MsScheduler` in `sched.rs`, mirroring `neurosight_encoder::Scheduler`.
Limits from `SafetyLimits::default()`: 8 pulses per pattern, 100 ms between pulses, rest 2000 ms after
a pattern ends, 12 patterns per minute, queue of 3, entries older than 2000 ms dropped.

* push: replace a waiting entry with the same action; if the queue is full drop the oldest; append.
* poll(now), every 1 ms:
  1. drop entries with `now - queued_at > 2000` (exactly 2000 is kept)
  2. nothing if the queue is empty or `now < rest_until` (exactly `rest_until` is allowed)
  3. forget starts with `now - start >= 60000`
  4. nothing if 12 starts remain
  5. start the oldest entry; `rest_until = now + duration + 2000`
* In a run, a push at millisecond t is applied before the poll at t.

Scenarios and expected starts: `scheduler_vectors.json`.

## 5. Not covered yet

Camera pipeline (downscale, gray, frame difference, wave counter), ToF/IMU/INA219 I2C, the FPGA-only
mode that raises events without the PC, and the physical PWM / 74HC595 timing. These need their own
reference model and vectors before RTL is written.
