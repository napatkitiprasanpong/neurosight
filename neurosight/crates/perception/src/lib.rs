//! Pose estimation with MoveNet SinglePose (Lightning or Thunder) through ONNX Runtime.
//!
//! Expected model (as exported from TensorFlow Hub with tf2onnx):
//!   input : [1, size, size, 3]  RGB, values 0-255, NHWC, int32 (float32 also accepted)
//!   output: [1, 1, 17, 3]       per keypoint: y, x, confidence (all 0.0-1.0)
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
enum InputKind {
    Int32,
    Float32,
}

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

        // Some exports want int32 input, some float32. Try int32 first, then float32,
        // using a blank test image, and remember which one works.
        let blank = vec![0u8; input_size * input_size * 3];
        let (kind, out) = match run_raw(&mut session, InputKind::Int32, input_size, &blank) {
            Ok(out) => (InputKind::Int32, out),
            Err(int_err) => match run_raw(&mut session, InputKind::Float32, input_size, &blank) {
                Ok(out) => (InputKind::Float32, out),
                Err(float_err) => {
                    return Err(PerceptionError(format!(
                        "the model rejected both int32 input ({}) and float32 input ({})",
                        int_err.0, float_err.0
                    )))
                }
            },
        };
        if out.len() < KEYPOINT_COUNT * 3 {
            return Err(PerceptionError(format!(
                "unexpected output size {} (expected at least {} = 17 keypoints x 3 numbers)",
                out.len(),
                KEYPOINT_COUNT * 3
            )));
        }

        Ok(Self {
            session,
            size: input_size,
            kind,
        })
    }

    pub fn input_size(&self) -> usize {
        self.size
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

/// Run the model once and return the flat output numbers.
fn run_raw(
    session: &mut Session,
    kind: InputKind,
    size: usize,
    rgb: &[u8],
) -> Result<Vec<f32>, PerceptionError> {
    let shape = [1usize, size, size, 3];
    let outputs = match kind {
        InputKind::Int32 => {
            let data: Vec<i32> = rgb.iter().map(|&v| v as i32).collect();
            let tensor = Tensor::from_array((shape, data)).map_err(err)?;
            session.run(ort::inputs![tensor]).map_err(err)?
        }
        InputKind::Float32 => {
            let data: Vec<f32> = rgb.iter().map(|&v| v as f32).collect();
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
}
