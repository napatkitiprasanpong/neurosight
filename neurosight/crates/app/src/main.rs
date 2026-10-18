//! NeuroSight: watch the webcam and describe what is happening in text.
//!
//! Usage:
//!   neurosight [--camera N] [--model PATH] [--motion] [--debug]
//!
//!   --camera N     camera number (default 0)
//!   --model PATH   MoveNet ONNX model (default models/movenet_lightning.onnx)
//!   --motion       skip the pose model and use motion-only mode
//!   --debug        print numbers twice a second, for tuning thresholds
//!
//! If the model file is missing or fails to load, the program falls back to
//! motion-only mode and says so.

use std::path::Path;
use std::process;
use std::time::Instant;

use neurosight_capture::WebcamSource;
use neurosight_common::Event;
use neurosight_events::{Config, MotionAnalyzer, PoseAnalyzer, PoseConfig};
use neurosight_perception::{PoseEstimator, LIGHTNING_INPUT_SIZE};

const GRAY_W: usize = 80;
const GRAY_H: usize = 60;
const DEFAULT_MODEL: &str = "models/movenet_lightning.onnx";
const USAGE: &str = "Usage: neurosight [--camera N] [--model PATH] [--motion] [--debug]";

struct Args {
    camera: u32,
    model: String,
    motion_only: bool,
    debug: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        camera: 0,
        model: DEFAULT_MODEL.to_string(),
        motion_only: false,
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
                args.model = it.next().unwrap_or_else(|| {
                    eprintln!("--model needs a file path");
                    process::exit(2);
                });
            }
            "--motion" => args.motion_only = true,
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

fn choose_mode(args: &Args) -> Mode {
    if !args.motion_only {
        if !Path::new(&args.model).exists() {
            eprintln!(
                "Pose model not found at '{}'. Using motion-only mode (see models/README.md to get the model).",
                args.model
            );
        } else {
            match PoseEstimator::load(&args.model, LIGHTNING_INPUT_SIZE) {
                Ok(estimator) => {
                    return Mode::Pose {
                        estimator,
                        analyzer: PoseAnalyzer::new(PoseConfig::default()),
                    }
                }
                Err(e) => eprintln!("Could not load the pose model: {}. Using motion-only mode.", e),
            }
        }
    }
    Mode::Motion(MotionAnalyzer::new(Config::default()))
}

fn main() {
    let args = parse_args();
    let mut mode = choose_mode(&args);

    let mut camera =
        match WebcamSource::open(args.camera, GRAY_W, GRAY_H, LIGHTNING_INPUT_SIZE) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Could not open camera {}: {}", args.camera, e);
                eprintln!(
                    "Check that a webcam is connected and that this program is allowed to use it."
                );
                process::exit(1);
            }
        };

    let mode_name = match &mode {
        Mode::Motion(_) => "motion-only mode",
        Mode::Pose { .. } => "pose mode",
    };
    println!(
        "NeuroSight is watching camera {} in {}. Move or wave in front of it. Press Ctrl+C to stop.",
        args.camera, mode_name
    );

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
                Ok(pose) => analyzer.process(Some(&pose), t),
                Err(e) => {
                    eprintln!("{}", e);
                    process::exit(1);
                }
            },
        };
        if let Some(event) = event {
            println!("[{:7.2}s] {}", t, event.to_text());
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
                        "    debug: person={} shoulder_width={:.2} left_hand_raised={} right_hand_raised={} direction_changes(L/R)={}/{}",
                        d.person_visible,
                        d.shoulder_span,
                        d.left_raised,
                        d.right_raised,
                        d.left_reversals,
                        d.right_reversals
                    );
                }
            }
        }
    }
}
