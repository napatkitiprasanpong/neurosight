//! NeuroSight: watch the webcam and describe what is happening in text.
//!
//! Usage:
//!   neurosight [--camera N] [--model PATH] [--motion] [--no-stim] [--log FILE] [--debug]
//!
//!   --camera N     camera number (default 0)
//!   --model PATH   MoveNet ONNX model (default: look in the models folder)
//!   --motion       skip the pose model and use motion-only mode
//!   --no-stim      only print the text, skip the simulated stimulation
//!   --log FILE     append every stimulation pattern to FILE as one line of JSON
//!   --debug        print numbers twice a second, for tuning thresholds
//!
//! Pose mode (needs the model) reports: someone appeared / left, waving, reaching out a
//! hand, turning the head left or right, looking up or down, nodding and shaking the head.
//! Motion-only mode (no model) can only report still / moving / waving. The first lines
//! printed say which mode is running and why.
//!
//! Every event is also turned into an abstract stimulation pattern and played on a
//! SIMULATED receiver (the flashes of light a person would see). This is simulation only:
//! nothing is sent to any device.

use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use neurosight_capture::WebcamSource;
use neurosight_common::Event;
use neurosight_events::{Config, MotionAnalyzer, PoseAnalyzer, PoseConfig};
use neurosight_perception::{PoseEstimator, LIGHTNING_INPUT_SIZE};

mod stim;
use stim::Stimulation;

const GRAY_W: usize = 80;
const GRAY_H: usize = 60;
const USAGE: &str =
    "Usage: neurosight [--camera N] [--model PATH] [--motion] [--no-stim] [--log FILE] [--debug]";

struct Args {
    camera: u32,
    model: Option<String>,
    motion_only: bool,
    stim: bool,
    log: Option<String>,
    debug: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        camera: 0,
        model: None,
        motion_only: false,
        stim: true,
        log: None,
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

enum Mode {
    Motion(MotionAnalyzer),
    Pose {
        estimator: PoseEstimator,
        analyzer: PoseAnalyzer,
    },
}

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
        "no .onnx file found. Looked in: {} and {}",
        show(&dirs[0]),
        show(&dirs[1])
    ))
}

/// Pick pose mode if possible, otherwise motion-only mode, and say clearly which and why.
fn choose_mode(args: &Args) -> Mode {
    let reason: String;
    if args.motion_only {
        reason = "you used --motion".to_string();
    } else {
        match find_model(&args.model) {
            Err(why) => reason = format!("the pose model file was not found ({})", why),
            Ok(path) => match PoseEstimator::load(&path, LIGHTNING_INPUT_SIZE) {
                Ok(estimator) => {
                    println!(
                        "POSE MODE: model {} (input {})",
                        show(&path),
                        estimator.input_description()
                    );
                    return Mode::Pose {
                        estimator,
                        analyzer: PoseAnalyzer::new(PoseConfig::default()),
                    };
                }
                Err(e) => {
                    reason = format!("the model {} could not be loaded: {}", show(&path), e);
                }
            },
        }
    }
    eprintln!();
    eprintln!("*** MOTION-ONLY MODE: {}", reason);
    eprintln!("*** In this mode only still / moving / waving can be recognized.");
    eprintln!("*** Head turns, looking up/down, nodding, shaking and the handshake offer");
    eprintln!("*** need the pose model: see models/README.md.");
    eprintln!();
    Mode::Motion(MotionAnalyzer::new(Config::default()))
}

fn main() {
    let args = parse_args();
    let mut mode = choose_mode(&args);

    let mut camera = match WebcamSource::open(args.camera, GRAY_W, GRAY_H, LIGHTNING_INPUT_SIZE)
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Could not open camera {}: {}", args.camera, e);
            eprintln!("Check that a webcam is connected and that this program is allowed to use it.");
            process::exit(1);
        }
    };

    let mode_name = match &mode {
        Mode::Motion(_) => "motion-only mode",
        Mode::Pose { .. } => "pose mode",
    };
    println!(
        "NeuroSight is watching camera {} in {}. Press Ctrl+C to stop.",
        args.camera, mode_name
    );

    let mut stimulation = if args.stim {
        Some(Stimulation::new(args.log.as_deref()))
    } else {
        None
    };

    let start = Instant::now();
    let mut last_debug = 0.0_f64;

    loop {
        let captured = match camera.next_frame() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Lost the camera: {}", e);
                process::exit(1);
            }
        };
        let t = start.elapsed().as_secs_f64();

        let event: Option<Event> = match &mut mode {
            Mode::Motion(analyzer) => analyzer.process(&captured.gray, t),
            Mode::Pose { estimator, analyzer } => match estimator.estimate(&captured.rgb) {
                Ok(pose) => {
                    analyzer.set_aspect(captured.aspect);
                    analyzer.process(Some(&pose), t)
                }
                Err(e) => {
                    eprintln!("{}", e);
                    process::exit(1);
                }
            },
        };
        if let Some(event) = event {
            println!("[{:7.2}s] {}", t, event.to_text());
            if let Some(stimulation) = stimulation.as_mut() {
                stimulation.on_event(&event, t);
            }
        }
        if let Some(stimulation) = stimulation.as_mut() {
            stimulation.tick(t);
        }

        if args.debug && t - last_debug >= 0.5 {
            last_debug = t;
            match &mode {
                Mode::Motion(analyzer) => {
                    let s = analyzer.stats();
                    let cx = match s.centroid_x {
                        Some(x) => format!("{:.2}", x),
                        None => "--".to_string(),
                    };
                    println!(
                        "    debug: state={} motion={:.1}% center_x={} direction_changes={}",
                        analyzer.state(),
                        s.motion_ratio * 100.0,
                        cx,
                        s.reversals
                    );
                }
                Mode::Pose { analyzer, .. } => {
                    let d = analyzer.debug();
                    println!(
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
                    );
                }
            }
        }
    }
}
