//! Webcam capture: opens a camera and returns each frame three times:
//! a small grayscale copy (for motion analysis), a square color copy (for the pose model)
//! and a color copy with the original shape (for the camera window).

use std::fmt;

use neurosight_common::{GrayFrame, RgbFrame};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;

#[derive(Debug)]
pub struct CaptureError(pub String);

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "camera error: {}", self.0)
    }
}

impl std::error::Error for CaptureError {}

impl From<nokhwa::NokhwaError> for CaptureError {
    fn from(e: nokhwa::NokhwaError) -> Self {
        CaptureError(e.to_string())
    }
}

/// One camera frame in the two forms the rest of the program needs.
pub struct Captured {
    /// Small grayscale image for motion analysis.
    pub gray: GrayFrame,
    /// Square color image for the pose model (the camera image is squeezed to fit).
    pub rgb: RgbFrame,
    /// Width divided by height of the original camera image (for example 1.33 for 640x480).
    pub aspect: f32,
    /// Color image with the original shape, for showing in a window.
    pub preview: RgbFrame,
}

pub struct WebcamSource {
    camera: Camera,
    gray_w: usize,
    gray_h: usize,
    rgb_size: usize,
    preview_w: usize,
}

impl WebcamSource {
    /// Open camera number `index` (0 = default webcam).
    /// `gray_w` x `gray_h` is the grayscale size; `rgb_size` is the side of the square color
    /// image; `preview_w` is the width of the preview image (its height follows the camera shape).
    pub fn open(
        index: u32,
        gray_w: usize,
        gray_h: usize,
        rgb_size: usize,
        preview_w: usize,
    ) -> Result<Self, CaptureError> {
        let requested = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::AbsoluteHighestFrameRate,
        );
        let mut camera = Camera::new(CameraIndex::Index(index), requested)?;
        camera.open_stream()?;
        Ok(Self {
            camera,
            gray_w,
            gray_h,
            rgb_size,
            preview_w,
        })
    }

    /// Block until the next frame arrives.
    pub fn next_frame(&mut self) -> Result<Captured, CaptureError> {
        let buffer = self.camera.frame()?;
        let img = buffer.decode_image::<RgbFormat>()?;
        let (w, h) = (img.width() as usize, img.height() as usize);
        let raw = img.as_raw();
        let aspect = w as f32 / h.max(1) as f32;
        let preview_h = ((self.preview_w as f32 / aspect).round() as usize).max(1);
        Ok(Captured {
            gray: GrayFrame::from_rgb_downscaled(raw, w, h, self.gray_w, self.gray_h),
            rgb: RgbFrame::from_rgb_downscaled(raw, w, h, self.rgb_size, self.rgb_size),
            aspect,
            preview: RgbFrame::from_rgb_downscaled(raw, w, h, self.preview_w, preview_h),
        })
    }
}
