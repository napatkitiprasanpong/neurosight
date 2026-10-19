//! Pose estimation with MoveNet SinglePose (Lightning or Thunder) through ONNX Runtime.
//!
//! Expected model (as exported from TensorFlow Hub):
//!   input : a square RGB image, values 0-255, as [1, size, size, 3] (NHWC) or
//!           [1, 3, size, size] (NCHW), integer or float
//!   output: [1, 1, 17, 3]  per keypoint: y, x, confidence (all 0.0-1.0)
//!
//! Exports differ, so `load` tries the four combinations (NHWC int32, NHWC float32,
//! NCHW float32, NCHW int32) with a blank test image and remembers the first that works.
//! If none works, the error lists what the model said about each attempt.
//!
//! See `models/README.md` for how to get the model file.

use std::fmt;
use std::path::Path;

use neurosight_common::{Keypoint, Pose, RgbFrame, KEYPOINT_COUNT};
use ort::session::Session;
use ort::value::Tensor;

/// Input size of MoveNet SinglePose Lightning. (Thunder uses 256.)
pub const LIGHTNING_INPUT_SIZE: usize = 192;

#[derive(Debug)]
pub struct PerceptionError(pub String);

impl fmt::Display for PerceptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pose model error: {}", self.0)
    }
}

impl std::error::Error for PerceptionError {}

fn err<E: fmt::Display>(e: E) -> PerceptionError {
    PerceptionError(e.to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// [1, height, width, 3]
    Nhwc,
    /// [1, 3, height, width]
    Nchw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dtype {
    Int32,
    Float32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InputKind {
    layout: Layout,
    dtype: Dtype,
}

impl InputKind {
    fn label(&self) -> &'static str {
        match (self.layout, self.dtype) {
            (Layout::Nhwc, Dtype::Int32) => "NHWC int32",
            (Layout::Nhwc, Dtype::Float32) => "NHWC float32",
            (Layout::Nchw, Dtype::Int32) => "NCHW int32",
            (Layout::Nchw, Dtype::Float32) => "NCHW float32",
        }
    }
}

/// The input formats to try, most likely first.
const CANDIDATES: [InputKind; 4] = [
    InputKind {
        layout: Layout::Nhwc,
        dtype: Dtype::Int32,
    },
    InputKind {
        layout: Layout::Nhwc,
        dtype: Dtype::Float32,
    },
    InputKind {
        layout: Layout::Nchw,
        dtype: Dtype::Float32,
    },
    InputKind {
        layout: Layout::Nchw,
        dtype: Dtype::Int32,
    },
];

pub struct PoseEstimator {
    session: Session,
    size: usize,
    kind: InputKind,
}

impl PoseEstimator {
    /// Load a MoveNet ONNX model. `input_size` is 192 for Lightning, 256 for Thunder.
    pub fn load(model_path: impl AsRef<Path>, input_size: usize) -> Result<Self, PerceptionError> {
        let mut session = Session::builder()
            .map_err(err)?
            .commit_from_file(model_path.as_ref())
            .map_err(err)?;

        // Find out which input format the model accepts, using a blank test image.
        let blank = vec![0u8; input_size * input_size * 3];
        let mut failures: Vec<String> = Vec::new();
        let mut chosen: Option<InputKind> = None;
        for kind in CANDIDATES {
            match run_raw(&mut session, kind, input_size, &blank) {
                Ok(out) if out.len() >= KEYPOINT_COUNT * 3 => {
                    chosen = Some(kind);
                    break;
                }
                Ok(out) => failures.push(format!(
                    "{}: ran, but the output has {} numbers (expected at least {})",
                    kind.label(),
                    out.len(),
                    KEYPOINT_COUNT * 3
                )),
                Err(e) => failures.push(format!("{}: {}", kind.label(), shorten(&e.0))),
            }
        }
        let kind = chosen.ok_or_else(|| {
            PerceptionError(format!(
                "the model did not accept any input format I know. What it said: {}",
                failures.join(" | ")
            ))
        })?;

        Ok(Self {
            session,
            size: input_size,
            kind,
        })
    }

    pub fn input_size(&self) -> usize {
        self.size
    }

    /// For example "NHWC int32" (shown in the start-up message).
    pub fn input_description(&self) -> &'static str {
        self.kind.label()
    }

    /// Estimate the pose in a square RGB frame of exactly `input_size` x `input_size`.
    pub fn estimate(&mut self, frame: &RgbFrame) -> Result<Pose, PerceptionError> {
        if frame.width != self.size || frame.height != self.size {
            return Err(PerceptionError(format!(
                "frame is {}x{} but the model needs {}x{}",
                frame.width, frame.height, self.size, self.size
            )));
        }
        let out = run_raw(&mut self.session, self.kind, self.size, &frame.data)?;
        parse_movenet_output(&out).ok_or_else(|| {
            PerceptionError(format!("model output too short: {} numbers", out.len()))
        })
    }
}

fn shorten(message: &str) -> String {
    let one_line: String = message.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() > 240 {
        let cut: String = one_line.chars().take(240).collect();
        format!("{}...", cut)
    } else {
        one_line
    }
}

/// Rearrange packed RGB (R,G,B,R,G,B,...) into planes (all R, then all G, then all B).
fn to_planar(rgb: &[u8], pixels: usize) -> Vec<u8> {
    let mut planar = Vec::with_capacity(pixels * 3);
    for channel in 0..3 {
        for i in 0..pixels {
            planar.push(rgb[i * 3 + channel]);
        }
    }
    planar
}

/// Run the model once and return the flat output numbers.
fn run_raw(
    session: &mut Session,
    kind: InputKind,
    size: usize,
    rgb: &[u8],
) -> Result<Vec<f32>, PerceptionError> {
    let (shape, bytes): ([usize; 4], Vec<u8>) = match kind.layout {
        Layout::Nhwc => ([1, size, size, 3], rgb.to_vec()),
        Layout::Nchw => ([1, 3, size, size], to_planar(rgb, size * size)),
    };
    let outputs = match kind.dtype {
        Dtype::Int32 => {
            let data: Vec<i32> = bytes.iter().map(|&v| v as i32).collect();
            let tensor = Tensor::from_array((shape, data)).map_err(err)?;
            session.run(ort::inputs![tensor]).map_err(err)?
        }
        Dtype::Float32 => {
            let data: Vec<f32> = bytes.iter().map(|&v| v as f32).collect();
            let tensor = Tensor::from_array((shape, data)).map_err(err)?;
            session.run(ort::inputs![tensor]).map_err(err)?
        }
    };
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(err)?;
    Ok(data.to_vec())
}

/// Turn the model's flat output (17 x [y, x, score]) into a `Pose`.
pub fn parse_movenet_output(data: &[f32]) -> Option<Pose> {
    if data.len() < KEYPOINT_COUNT * 3 {
        return None;
    }
    let mut keypoints = [Keypoint::default(); KEYPOINT_COUNT];
    for (i, k) in keypoints.iter_mut().enumerate() {
        *k = Keypoint {
            y: data[i * 3],
            x: data[i * 3 + 1],
            score: data[i * 3 + 2],
        };
    }
    Some(Pose::new(keypoints))
}

#[cfg(test)]
mod tests {
    use super::*;
    use neurosight_common::kp;

    #[test]
    fn parses_y_x_score_order() {
        // Keypoint i gets y = i/100, x = i/100 + 0.5, score = 0.9
        let mut data = Vec::new();
        for i in 0..17 {
            data.push(i as f32 / 100.0);
            data.push(i as f32 / 100.0 + 0.5);
            data.push(0.9);
        }
        let pose = parse_movenet_output(&data).expect("pose");
        let wrist = pose.keypoints[kp::LEFT_WRIST];
        assert!((wrist.y - 0.09).abs() < 1e-6);
        assert!((wrist.x - 0.59).abs() < 1e-6);
        assert!((wrist.score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn short_output_is_rejected() {
        assert!(parse_movenet_output(&[0.0; 10]).is_none());
    }

    #[test]
    fn planar_layout_groups_channels() {
        // Two pixels: (1,2,3) and (4,5,6) -> planes [1,4], [2,5], [3,6]
        let planar = to_planar(&[1, 2, 3, 4, 5, 6], 2);
        assert_eq!(planar, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn long_error_messages_are_shortened() {
        let long = "word ".repeat(200);
        assert!(shorten(&long).chars().count() <= 245);
        assert_eq!(shorten("a\n  b"), "a b");
    }
}
