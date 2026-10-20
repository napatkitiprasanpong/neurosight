//! NeuroSight: watch the webcam and describe what is happening in text.
//!
//! Usage:
//!   neurosight [--camera N] [--camera-off] [--model PATH] [--motion] [--no-stim]
//!              [--log FILE] [--no-window] [--mirror] [--debug]
//!
//!   --camera N     camera number (default 0)
//!   --camera-off   start with the camera turned off (turn it on in the window)
//!   --model PATH   MoveNet ONNX model (default: look in the models folder)
//!   --motion       start in motion-only mode instead of pose mode
//!   --no-stim      start with the simulated stimulation turned off
//!   --log FILE     append every stimulation pattern to FILE as one line of JSON
//!   --no-window    console only: no window and no buttons
//!   --mirror       start with the camera picture flipped like a mirror (the text still uses
//!                  the camera's own left and right)
//!   --debug        print numbers twice a second, for tuning thresholds
//!
//! The window has the camera picture on the left (with the detected body points, the latest
//! event and the simulated flash of light) and a control panel on the right with buttons for
//! camera on/off, mode (pose / motion only), stimulation and mirror. Keys: C camera, P pose,
//! M motion, S stim, R mirror, Esc quit.
//!
//! Pose mode (needs the model) reports: someone appeared / left, waving, reaching out a
//! hand, turning the head left or right, looking up or down, nodding and shaking the head.
//! Motion-only mode (no model) can only report still / moving / waving.
//!
//! Every event is also turned into an abstract stimulation pattern and played on a
//! SIMULATED receiver (the flashes of light a person would see). This is simulation only:
//! nothing is sent to any device.

use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant};

use neurosight_capture::{Captured, WebcamSource};
use neurosight_common::{Event, Pose};
use neurosight_events::{Config, MotionAnalyzer, PoseAnalyzer, PoseConfig};
use neurosight_perception::{PoseEstimator, LIGHTNING_INPUT_SIZE};

mod preview;
mod stim;
mod ui;
use preview::{Flash, Overlay, Preview};
use stim::Stimulation;
use ui::{ModeChoice, UiAction, UiState};

const GRAY_W: usize = 80;
const GRAY_H: usize = 60;
/// Width of the picture shown in the window.
const PREVIEW_W: usize = 640;
const USAGE: &str = "Usage: neurosight [--camera N] [--camera-off] [--model PATH] [--motion] [--no-stim] [--log FILE] [--no-window] [--mirror] [--debug]";

struct Args {
    camera: u32,
    camera_off: bool,
    model: Option<String>,
    motion_only: bool,
    stim: bool,
    log: Option<String>,
    window: bool,
    mirror: bool,
    debug: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        camera: 0,
        camera_off: false,
        model: None,
        motion_only: false,
        stim: true,
        log: None,
        window: true,
        mirror: false,
        debug: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--camera" => {
                let value = it.next().unwrap_or_default();
                args.camera = value.parse().unwrap_or_else(|_| {
                    eprintln!("--camera needs a number, got '{}'", value);
                    process::exit(2);
                });
            }
            "--camera-off" => args.camera_off = true,
            "--model" => {
                args.model = Some(it.next().unwrap_or_else(|| {
                    eprintln!("--model needs a file path");
                    process::exit(2);
                }));
            }
            "--motion" => args.motion_only = true,
            "--no-stim" => args.stim = false,
            "--log" => {
                args.log = Some(it.next().unwrap_or_else(|| {
                    eprintln!("--log needs a file path");
                    process::exit(2);
                }));
            }
            "--no-window" => args.window = false,
            "--mirror" => args.mirror = true,
            "--debug" => args.debug = true,
            "-h" | "--help" => {
                println!("{}", USAGE);
                process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                eprintln!("{}", USAGE);
                process::exit(2);
            }
        }
    }
    args
}

// ------------------------------------------------------------------- model

fn show(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

/// The project folder (two levels above this crate), so the program finds `models/` even
/// when it is started from another folder.
fn project_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Find the model file. Without `--model`, look in `models/` (in the current folder and in
/// the project folder) for the usual names, then for any `.onnx` file.
fn find_model(arg: &Option<String>) -> Result<PathBuf, String> {
    if let Some(given) = arg {
        let path = PathBuf::from(given);
        if path.exists() {
            return Ok(path);
        }
        let in_project = project_dir().join(given);
        if in_project.exists() {
            return Ok(in_project);
        }
        return Err(format!("the file '{}' does not exist", given));
    }

    let dirs = [PathBuf::from("models"), project_dir().join("models")];
    let names = ["movenet_lightning.onnx", "model.onnx", "movenet.onnx"];
    for dir in &dirs {
        for name in names {
            let candidate = dir.join(name);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }
    for dir in &dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("onnx") {
                    return Ok(path);
                }
            }
        }
    }
    Err(format!(
        "no pose model (.onnx file) found. Looked in: {} and {}",
        show(&dirs[0]),
        show(&dirs[1])
    ))
}

// ------------------------------------------------------------------ engine

struct PoseBackend {
    estimator: PoseEstimator,
    analyzer: PoseAnalyzer,
}

/// The analysis behind the mode buttons. The pose model is loaded the first time it is needed
/// and then kept, so switching modes back and forth is quick.
struct Engine {
    motion: MotionAnalyzer,
    pose: Option<PoseBackend>,
    use_pose: bool,
    model_arg: Option<String>,
}

impl Engine {
    fn new(model_arg: Option<String>) -> Self {
        Self {
            motion: MotionAnalyzer::new(Config::default()),
            pose: None,
            use_pose: false,
            model_arg,
        }
    }

    /// Load the pose model if that has not happened yet. On success returns a description
    /// (file and input format); on failure the reason.
    fn load_pose(&mut self) -> Result<String, String> {
        if let Some(p) = &self.pose {
            return Ok(format!("already loaded (input {})", p.estimator.input_description()));
        }
        let path = find_model(&self.model_arg)?;
        let estimator = PoseEstimator::load(&path, LIGHTNING_INPUT_SIZE)
            .map_err(|e| format!("the model {} could not be loaded: {}", show(&path), e))?;
        let description = format!("{} (input {})", show(&path), estimator.input_description());
        self.pose = Some(PoseBackend {
            estimator,
            analyzer: PoseAnalyzer::new(PoseConfig::default()),
        });
        Ok(description)
    }

    fn set_mode(&mut self, mode: ModeChoice) -> Result<(), String> {
        match mode {
            ModeChoice::Motion => {
                self.use_pose = false;
            }
            ModeChoice::Pose => {
                self.load_pose()?;
                self.use_pose = true;
            }
        }
        self.reset();
        Ok(())
    }

    fn mode(&self) -> ModeChoice {
        if self.use_pose {
            ModeChoice::Pose
        } else {
            ModeChoice::Motion
        }
    }

    /// Forget what was seen so far (after a mode change or when the camera comes back).
    fn reset(&mut self) {
        self.motion = MotionAnalyzer::new(Config::default());
        if let Some(p) = self.pose.as_mut() {
            p.analyzer = PoseAnalyzer::new(PoseConfig::default());
        }
    }

    /// Analyze one camera frame.
    fn analyse(
        &mut self,
        captured: &Captured,
        t: f64,
        last_pose: &mut Option<Pose>,
    ) -> Result<Option<Event>, String> {
        if self.use_pose {
            if let Some(p) = self.pose.as_mut() {
                let pose = p
                    .estimator
                    .estimate(&captured.rgb)
                    .map_err(|e| e.to_string())?;
                p.analyzer.set_aspect(captured.aspect);
                let event = p.analyzer.process(Some(&pose), t);
                *last_pose = Some(pose);
                return Ok(event);
            }
        }
        *last_pose = None;
        Ok(self.motion.process(&captured.gray, t))
    }

    /// One line for the top of the camera picture.
    fn status_line(&self) -> String {
        match (&self.pose, self.use_pose) {
            (Some(p), true) => {
                let d = p.analyzer.debug();
                if d.person_visible {
                    format!("POSE MODE   person seen   shoulders: {:.2}", d.shoulder_span)
                } else {
                    "POSE MODE   no person seen".to_string()
                }
            }
            _ => format!(
                "MOTION-ONLY MODE   state: {}   motion: {:.1}%",
                self.motion.state(),
                self.motion.stats().motion_ratio * 100.0
            ),
        }
    }

    /// Numbers for tuning (printed with --debug).
    fn debug_line(&self) -> String {
        match (&self.pose, self.use_pose) {
            (Some(p), true) => {
                let d = p.analyzer.debug();
                format!(
                    "    debug: person={} shoulder_width={:.2} hands_raised(L/R)={}/{} wave_changes(L/R)={}/{} head_offset(x/y)={:+.2}/{:+.2} shake_changes={} nod_changes={}",
                    d.person_visible,
                    d.shoulder_span,
                    d.left_raised,
                    d.right_raised,
                    d.left_reversals,
                    d.right_reversals,
                    d.head_dx,
                    d.head_dy,
                    d.shake_reversals,
                    d.nod_reversals
                )
            }
            _ => {
                let s = self.motion.stats();
                let cx = match s.centroid_x {
                    Some(x) => format!("{:.2}", x),
                    None => "--".to_string(),
                };
                format!(
                    "    debug: state={} motion={:.1}% center_x={} direction_changes={}",
                    self.motion.state(),
                    s.motion_ratio * 100.0,
                    cx,
                    s.reversals
                )
            }
        }
    }
}

// ------------------------------------------------------------------ camera

fn open_camera(index: u32) -> Result<WebcamSource, String> {
    WebcamSource::open(index, GRAY_W, GRAY_H, LIGHTNING_INPUT_SIZE, PREVIEW_W).map_err(|e| {
        format!(
            "Could not open camera {}: {}. Check that a webcam is connected and that this program is allowed to use it.",
            index, e
        )
    })
}

// -------------------------------------------------------------------- main

fn main() {
    let args = parse_args();

    // --- analysis and mode
    let mut engine = Engine::new(args.model.clone());
    let mut ui_state = UiState {
        camera_on: false,
        mode: ModeChoice::Motion,
        stim: args.stim,
        mirror: args.mirror,
        message: String::new(),
    };
    if args.motion_only {
        println!("MOTION-ONLY MODE: you used --motion");
    } else {
        match engine.load_pose() {
            Ok(description) => {
                engine.use_pose = true;
                ui_state.mode = ModeChoice::Pose;
                println!("POSE MODE: model {}", description);
            }
            Err(why) => {
                eprintln!();
                eprintln!("*** MOTION-ONLY MODE: {}", why);
                eprintln!("*** In this mode only still / moving / waving can be recognized.");
                eprintln!("*** Head turns, looking up/down, nodding, shaking and the handshake offer");
                eprintln!("*** need the pose model: see models/README.md.");
                eprintln!();
                ui_state.message = format!("Motion-only mode: {}", why);
            }
        }
    }

    // --- window
    let mut preview: Option<Preview> = if args.window {
        match Preview::open() {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("Could not open the window: {} (continuing in the console only)", e);
                None
            }
        }
    } else {
        None
    };

    // --- camera
    let mut camera: Option<WebcamSource> = None;
    if args.camera_off {
        if preview.is_none() {
            eprintln!("--camera-off needs the window (there is no way to turn the camera on).");
            process::exit(2);
        }
    } else {
        match open_camera(args.camera) {
            Ok(c) => {
                camera = Some(c);
                ui_state.camera_on = true;
            }
            Err(e) => {
                eprintln!("{}", e);
                if preview.is_none() {
                    process::exit(1);
                }
                ui_state.message = e;
            }
        }
    }

    // --- simulated stimulation
    let mut stimulation: Option<Stimulation> = if args.stim {
        Some(Stimulation::new(args.log.as_deref()))
    } else {
        None
    };

    println!(
        "NeuroSight is running in {}. Press Ctrl+C{} to stop.",
        engine.mode().name(),
        if preview.is_some() { " or Esc in the window" } else { "" }
    );

    let mut last_pose: Option<Pose> = None;
    let mut last_event_text = String::new();
    let mut last_flash: Option<Flash> = None;
    let start = Instant::now();
    let mut last_debug = 0.0_f64;

    loop {
        // 1. The next camera picture (or a short pause while the camera is off).
        let mut captured: Option<Captured> = None;
        if let Some(cam) = camera.as_mut() {
            match cam.next_frame() {
                Ok(f) => captured = Some(f),
                Err(e) => {
                    let message = format!("Lost the camera: {}", e);
                    eprintln!("{}", message);
                    if preview.is_none() {
                        process::exit(1);
                    }
                    camera = None;
                    ui_state.camera_on = false;
                    ui_state.message = message;
                    last_pose = None;
                }
            }
        } else {
            std::thread::sleep(Duration::from_millis(33));
        }
        let t = start.elapsed().as_secs_f64();

        // 2. Analysis.
        if let Some(c) = captured.as_ref() {
            match engine.analyse(c, t, &mut last_pose) {
                Ok(Some(event)) => {
                    println!("[{:7.2}s] {}", t, event.to_text());
                    last_event_text = format!("[{:.1}s] {}", t, event.to_text());
                    if let Some(stimulation) = stimulation.as_mut() {
                        stimulation.on_event(&event, t);
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    // The pose model failed on a frame: fall back to motion-only mode.
                    let message = format!("Pose model problem: {}. Switched to motion-only mode.", e);
                    eprintln!("{}", message);
                    let _ = engine.set_mode(ModeChoice::Motion);
                    ui_state.mode = ModeChoice::Motion;
                    ui_state.message = message;
                    last_pose = None;
                }
            }
            if args.debug && t - last_debug >= 0.5 {
                last_debug = t;
                println!("{}", engine.debug_line());
            }
        }

        // 3. Simulated stimulation (keeps playing even while the camera is off).
        if let Some(stimulation) = stimulation.as_mut() {
            if let Some(flash) = stimulation.tick(t) {
                last_flash = Some(Flash {
                    col: flash.col,
                    row: flash.row,
                    brightness: flash.brightness,
                    at_s: t,
                });
            }
        }

        // 4. Window: draw, then collect clicks and key presses.
        let mut actions: Vec<UiAction> = Vec::new();
        let mut stop = false;
        let mut drop_window = false;
        if let Some(p) = preview.as_mut() {
            let status = engine.status_line();
            let overlay = Overlay {
                status: &status,
                event_text: &last_event_text,
                pose: last_pose.as_ref(),
                flash: last_flash,
                now_s: t,
            };
            let frame = captured.as_ref().map(|c| &c.preview);
            match p.draw(frame, &overlay, &ui_state) {
                Ok(()) => {
                    if !p.is_open() {
                        stop = true;
                    }
                    actions = p.poll_actions();
                }
                Err(e) => {
                    eprintln!("Window problem: {} (closing it)", e);
                    drop_window = true;
                }
            }
        }
        if drop_window {
            preview = None;
            if camera.is_none() {
                process::exit(1);
            }
        }
        if stop {
            println!("Window closed. Stopping.");
            break;
        }

        // 5. Apply what the user asked for.
        for action in actions {
            match action {
                UiAction::ToggleCamera => {
                    if camera.is_some() {
                        camera = None;
                        ui_state.camera_on = false;
                        ui_state.message = "Camera turned off.".to_string();
                        last_pose = None;
                        engine.reset();
                        println!("Camera turned off.");
                    } else {
                        match open_camera(args.camera) {
                            Ok(c) => {
                                camera = Some(c);
                                ui_state.camera_on = true;
                                ui_state.message = "Camera turned on.".to_string();
                                engine.reset();
                                println!("Camera turned on.");
                            }
                            Err(e) => {
                                eprintln!("{}", e);
                                ui_state.message = e;
                            }
                        }
                    }
                }
                UiAction::SetMode(mode) => match engine.set_mode(mode) {
                    Ok(()) => {
                        ui_state.mode = mode;
                        ui_state.message = format!("Now in {}.", mode.name());
                        last_pose = None;
                        println!("Now in {}.", mode.name());
                    }
                    Err(why) => {
                        let message = format!("Could not switch to {}: {}", mode.name(), why);
                        eprintln!("{}", message);
                        ui_state.mode = engine.mode();
                        ui_state.message = message;
                    }
                },
                UiAction::ToggleStim => {
                    if stimulation.is_some() {
                        stimulation = None;
                        ui_state.stim = false;
                        ui_state.message = "Stimulation simulation turned off.".to_string();
                    } else {
                        stimulation = Some(Stimulation::new(args.log.as_deref()));
                        ui_state.stim = true;
                        ui_state.message = "Stimulation simulation turned on.".to_string();
                    }
                }
                UiAction::ToggleMirror => {
                    ui_state.mirror = !ui_state.mirror;
                    ui_state.message = if ui_state.mirror {
                        "Mirror on: the picture is flipped, but the text still uses the camera's left and right.".to_string()
                    } else {
                        "Mirror off.".to_string()
                    };
                }
            }
        }
    }
}
