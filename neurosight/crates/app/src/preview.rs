//! The NeuroSight window. On the left it shows what the program is looking at: the camera
//! picture with the detected body points drawn on it, the current mode, the latest event as
//! text, and a small 5 x 5 grid that lights up where the simulated stimulation would make a
//! flash. On the right is the control panel (see `ui.rs`).
//!
//! Drawing is done by hand on a pixel buffer (`Canvas`), and the window itself comes from
//! the `minifb` crate. Text uses the public-domain 8x8 `font8x8` bitmap font.

use font8x8::{UnicodeFonts, BASIC_FONTS};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use neurosight_common::{kp, Pose, RgbFrame, GRID};

use crate::ui::{
    button_at, draw_panel, ButtonId, ModeChoice, UiAction, UiState, VIDEO_H, VIDEO_W, WINDOW_H,
    WINDOW_W,
};

/// Body points below this confidence are not drawn.
const MIN_SCORE: f32 = 0.3;

const WHITE: u32 = 0xFFFFFF;
const BLACK: u32 = 0x000000;
const CYAN: u32 = 0x00E0FF;
const ORANGE: u32 = 0xFFA000;
const GREEN: u32 = 0x40E060;
const GRAY: u32 = 0x909090;
const YELLOW: u32 = 0xFFE040;
const BACKGROUND: u32 = 0x15181C;

/// Lines of the body drawing: (from, to, color).
const BONES: [(usize, usize, u32); 15] = [
    (kp::NOSE, kp::LEFT_EYE, CYAN),
    (kp::NOSE, kp::RIGHT_EYE, CYAN),
    (kp::LEFT_EYE, kp::LEFT_EAR, CYAN),
    (kp::RIGHT_EYE, kp::RIGHT_EAR, CYAN),
    (kp::LEFT_SHOULDER, kp::RIGHT_SHOULDER, GREEN),
    (kp::LEFT_SHOULDER, kp::LEFT_HIP, GREEN),
    (kp::RIGHT_SHOULDER, kp::RIGHT_HIP, GREEN),
    (kp::LEFT_HIP, kp::RIGHT_HIP, GREEN),
    (kp::LEFT_SHOULDER, kp::LEFT_ELBOW, ORANGE),
    (kp::LEFT_ELBOW, kp::LEFT_WRIST, ORANGE),
    (kp::RIGHT_SHOULDER, kp::RIGHT_ELBOW, ORANGE),
    (kp::RIGHT_ELBOW, kp::RIGHT_WRIST, ORANGE),
    (kp::LEFT_HIP, kp::LEFT_KNEE, GRAY),
    (kp::LEFT_KNEE, kp::LEFT_ANKLE, GRAY),
    (kp::RIGHT_HIP, kp::RIGHT_KNEE, GRAY),
];

/// The latest simulated flash of light.
#[derive(Clone, Copy, Debug)]
pub struct Flash {
    /// Visual-field cell: col 0 = far left, row 0 = top.
    pub col: u8,
    pub row: u8,
    /// 0.0 to 1.0
    pub brightness: f32,
    /// Program time (seconds) when it happened. The light fades out over a third of a second.
    pub at_s: f64,
}

/// Everything drawn on top of the camera picture.
pub struct Overlay<'a> {
    /// First line, for example "POSE MODE  person: true".
    pub status: &'a str,
    /// The latest event as text (may be empty).
    pub event_text: &'a str,
    pub pose: Option<&'a Pose>,
    pub flash: Option<Flash>,
    /// Current program time in seconds.
    pub now_s: f64,
}

// ------------------------------------------------------------------ canvas

/// A pixel buffer (0x00RRGGBB) with a few drawing helpers.
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub buf: Vec<u32>,
}

fn pack(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

fn mix(a: u32, b: u32, alpha: f32) -> u32 {
    let alpha = alpha.clamp(0.0, 1.0);
    let channel = |shift: u32| -> u8 {
        let x = ((a >> shift) & 0xFF) as f32;
        let y = ((b >> shift) & 0xFF) as f32;
        (x * (1.0 - alpha) + y * alpha).round() as u8
    };
    pack(channel(16), channel(8), channel(0))
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            buf: vec![0; width * height],
        }
    }

    pub fn get(&self, x: i32, y: i32) -> Option<u32> {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            None
        } else {
            Some(self.buf[y as usize * self.width + x as usize])
        }
    }

    pub fn set(&mut self, x: i32, y: i32, color: u32) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.buf[y as usize * self.width + x as usize] = color;
        }
    }

    /// Draw the camera picture. With `mirror` the picture is flipped like a mirror.
    pub fn paint_rgb(&mut self, frame: &RgbFrame, mirror: bool) {
        let w = self.width.min(frame.width);
        let h = self.height.min(frame.height);
        for y in 0..h {
            for x in 0..w {
                let sx = if mirror { frame.width - 1 - x } else { x };
                let i = (y * frame.width + sx) * 3;
                self.buf[y * self.width + x] =
                    pack(frame.data[i], frame.data[i + 1], frame.data[i + 2]);
            }
        }
    }

    /// Copy another canvas onto this one with its top-left corner at (ox, oy); parts outside
    /// this canvas are cut off.
    pub fn blit(&mut self, src: &Canvas, ox: i32, oy: i32) {
        for y in 0..src.height as i32 {
            for x in 0..src.width as i32 {
                self.set(ox + x, oy + y, src.buf[y as usize * src.width + x as usize]);
            }
        }
    }

    /// Fill a rectangle; `alpha` 1.0 = solid, 0.5 = half see-through.
    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: u32, alpha: f32) {
        for yy in y..y + h {
            for xx in x..x + w {
                if let Some(old) = self.get(xx, yy) {
                    self.set(xx, yy, mix(old, color, alpha));
                }
            }
        }
    }

    /// A straight line (Bresenham).
    pub fn line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let (mut x, mut y) = (x0, y0);
        let mut err = dx + dy;
        loop {
            self.set(x, y, color);
            self.set(x + 1, y, color); // slightly thicker
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// A filled circle.
    pub fn dot(&mut self, cx: i32, cy: i32, r: i32, color: u32) {
        for y in -r..=r {
            for x in -r..=r {
                if x * x + y * y <= r * r {
                    self.set(cx + x, cy + y, color);
                }
            }
        }
    }

    /// Text in the 8x8 font; `scale` makes it bigger. Returns the width drawn in pixels.
    pub fn text(&mut self, x: i32, y: i32, text: &str, color: u32, scale: i32) -> i32 {
        let mut pen = x;
        for ch in text.chars() {
            if let Some(glyph) = BASIC_FONTS.get(ch) {
                for (row, bits) in glyph.iter().enumerate() {
                    for col in 0..8 {
                        // In this font the lowest bit is the leftmost pixel.
                        if bits & (1 << col) != 0 {
                            self.fill_rect(
                                pen + col * scale,
                                y + row as i32 * scale,
                                scale,
                                scale,
                                color,
                                1.0,
                            );
                        }
                    }
                }
            }
            pen += 8 * scale;
        }
        pen - x
    }
}

/// Break text into lines of at most `max_chars` characters, at spaces where possible.
pub fn wrap_text(text: &str, max_chars: usize) -> Vec<String> {
    let max_chars = max_chars.max(1);
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let needed = if current.is_empty() {
            word.chars().count()
        } else {
            current.chars().count() + 1 + word.chars().count()
        };
        if needed > max_chars && !current.is_empty() {
            lines.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Draw the body points and lines of a pose. Coordinates in a pose are 0..1 of the image.
fn draw_pose(canvas: &mut Canvas, pose: &Pose, mirror: bool) {
    let to_px = |k: usize| -> Option<(i32, i32)> {
        let p = pose.keypoints[k];
        if p.score < MIN_SCORE {
            return None;
        }
        let x = if mirror { 1.0 - p.x } else { p.x };
        Some((
            (x * canvas.width as f32).round() as i32,
            (p.y * canvas.height as f32).round() as i32,
        ))
    };
    let points: Vec<Option<(i32, i32)>> = (0..neurosight_common::KEYPOINT_COUNT).map(to_px).collect();
    for (from, to, color) in BONES {
        if let (Some(a), Some(b)) = (points[from], points[to]) {
            canvas.line(a.0, a.1, b.0, b.1, color);
        }
    }
    for p in points.iter().flatten() {
        canvas.dot(p.0, p.1, 3, WHITE);
    }
}

/// The 5 x 5 visual field, with the cell of the latest flash lit.
fn draw_flash_grid(canvas: &mut Canvas, flash: Option<Flash>, now_s: f64) {
    let cell = 14;
    let size = cell * GRID as i32;
    let x0 = canvas.width as i32 - size - 8;
    let y0 = 26;
    canvas.fill_rect(x0 - 3, y0 - 3, size + 6, size + 6, BLACK, 0.6);
    for r in 0..GRID as i32 {
        for c in 0..GRID as i32 {
            canvas.fill_rect(x0 + c * cell + 1, y0 + r * cell + 1, cell - 2, cell - 2, GRAY, 0.25);
        }
    }
    if let Some(f) = flash {
        let fade = (1.0 - (now_s - f.at_s) / 0.33).clamp(0.0, 1.0) as f32;
        let strength = f.brightness * fade;
        if strength > 0.02 {
            canvas.fill_rect(
                x0 + f.col as i32 * cell + 1,
                y0 + f.row as i32 * cell + 1,
                cell - 2,
                cell - 2,
                YELLOW,
                strength,
            );
        }
    }
    canvas.text(x0 - 3, y0 + size + 6, "seen", WHITE, 1);
}

/// Draw the whole picture: camera frame plus everything in the overlay.
pub fn render(canvas: &mut Canvas, frame: &RgbFrame, overlay: &Overlay, mirror: bool) {
    canvas.paint_rgb(frame, mirror);

    if let Some(pose) = overlay.pose {
        draw_pose(canvas, pose, mirror);
    }

    // Status line on top.
    canvas.fill_rect(0, 0, canvas.width as i32, 18, BLACK, 0.6);
    canvas.text(6, 5, overlay.status, WHITE, 1);

    draw_flash_grid(canvas, overlay.flash, overlay.now_s);

    // Latest event at the bottom, wrapped to at most three lines.
    if !overlay.event_text.is_empty() {
        let max_chars = (canvas.width / 8).saturating_sub(2).max(10);
        let mut lines = wrap_text(overlay.event_text, max_chars);
        lines.truncate(3);
        let strip_h = lines.len() as i32 * 11 + 8;
        let y0 = canvas.height as i32 - strip_h;
        canvas.fill_rect(0, y0, canvas.width as i32, strip_h, BLACK, 0.65);
        for (i, line) in lines.iter().enumerate() {
            canvas.text(6, y0 + 5 + i as i32 * 11, line, WHITE, 1);
        }
    }
}

/// Draw the whole window: camera area on the left (or a "camera is off" note), control panel
/// on the right. `view` is a scratch canvas the size of the camera picture.
pub fn render_window(
    canvas: &mut Canvas,
    view: &mut Canvas,
    frame: Option<&RgbFrame>,
    overlay: &Overlay,
    ui: &UiState,
    hover: Option<ButtonId>,
) {
    canvas.fill_rect(0, 0, canvas.width as i32, canvas.height as i32, BACKGROUND, 1.0);
    match frame {
        Some(frame) => {
            if view.width != frame.width || view.height != frame.height {
                *view = Canvas::new(frame.width, frame.height);
            }
            render(view, frame, overlay, ui.mirror);
            let ox = (VIDEO_W as i32 - view.width as i32) / 2;
            let oy = (VIDEO_H as i32 - view.height as i32) / 2;
            canvas.blit(view, ox, oy);
        }
        None => {
            let title = "Camera is off";
            let title_w = title.chars().count() as i32 * 24;
            canvas.text((VIDEO_W as i32 - title_w) / 2, 200, title, WHITE, 3);
            let hint = "Click the Camera button or press C";
            let hint_w = hint.chars().count() as i32 * 8;
            canvas.text((VIDEO_W as i32 - hint_w) / 2, 240, hint, GRAY, 1);
        }
    }
    draw_panel(canvas, ui, hover);
}

// ------------------------------------------------------------------ window

/// Keys that work like the buttons.
const KEY_ACTIONS: [(Key, UiAction); 5] = [
    (Key::C, UiAction::ToggleCamera),
    (Key::P, UiAction::SetMode(ModeChoice::Pose)),
    (Key::M, UiAction::SetMode(ModeChoice::Motion)),
    (Key::S, UiAction::ToggleStim),
    (Key::R, UiAction::ToggleMirror),
];

pub struct Preview {
    window: Window,
    canvas: Canvas,
    view: Canvas,
    mouse_was_down: bool,
}

impl Preview {
    /// Open the window (camera area + control panel).
    pub fn open() -> Result<Self, String> {
        let window = Window::new(
            "NeuroSight (Esc closes it)",
            WINDOW_W,
            WINDOW_H,
            WindowOptions::default(),
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            window,
            canvas: Canvas::new(WINDOW_W, WINDOW_H),
            view: Canvas::new(1, 1),
            mouse_was_down: false,
        })
    }

    /// False once the window was closed or Esc was pressed.
    pub fn is_open(&self) -> bool {
        self.window.is_open() && !self.window.is_key_down(Key::Escape)
    }

    /// Draw one frame of the window. `frame` is the camera picture, or None while the camera
    /// is off.
    pub fn draw(
        &mut self,
        frame: Option<&RgbFrame>,
        overlay: &Overlay,
        ui: &UiState,
    ) -> Result<(), String> {
        let hover = self
            .window
            .get_mouse_pos(MouseMode::Discard)
            .and_then(|(x, y)| button_at(x as i32, y as i32));
        render_window(&mut self.canvas, &mut self.view, frame, overlay, ui, hover);
        self.window
            .update_with_buffer(&self.canvas.buf, self.canvas.width, self.canvas.height)
            .map_err(|e| e.to_string())
    }

    /// Clicks and key presses since the last call (call after `draw`, which handles the
    /// window's events).
    pub fn poll_actions(&mut self) -> Vec<UiAction> {
        let mut actions = Vec::new();

        let down = self.window.get_mouse_down(MouseButton::Left);
        if down && !self.mouse_was_down {
            if let Some((x, y)) = self.window.get_mouse_pos(MouseMode::Discard) {
                if let Some(button) = button_at(x as i32, y as i32) {
                    actions.push(button.action());
                }
            }
        }
        self.mouse_was_down = down;

        for (key, action) in KEY_ACTIONS {
            if self.window.is_key_pressed(key, KeyRepeat::No) {
                actions.push(action);
            }
        }
        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neurosight_common::Keypoint;

    fn solid(width: usize, height: usize, rgb: [u8; 3]) -> RgbFrame {
        let mut data = Vec::new();
        for _ in 0..width * height {
            data.extend_from_slice(&rgb);
        }
        RgbFrame::new(width, height, data)
    }

    #[test]
    fn setting_pixels_outside_the_canvas_is_ignored() {
        let mut c = Canvas::new(4, 4);
        c.set(-1, 0, WHITE);
        c.set(0, -1, WHITE);
        c.set(4, 0, WHITE);
        c.set(0, 4, WHITE);
        assert!(c.buf.iter().all(|&p| p == 0));
        c.set(3, 3, WHITE);
        assert_eq!(c.get(3, 3), Some(WHITE));
        assert_eq!(c.get(4, 3), None);
    }

    #[test]
    fn camera_picture_can_be_mirrored() {
        // 2x1 picture: red, blue
        let frame = RgbFrame::new(2, 1, vec![255, 0, 0, 0, 0, 255]);
        let mut c = Canvas::new(2, 1);
        c.paint_rgb(&frame, false);
        assert_eq!(c.buf, vec![0xFF0000, 0x0000FF]);
        c.paint_rgb(&frame, true);
        assert_eq!(c.buf, vec![0x0000FF, 0xFF0000]);
    }

    #[test]
    fn line_touches_both_end_points() {
        let mut c = Canvas::new(20, 20);
        c.line(2, 3, 15, 12, WHITE);
        assert_eq!(c.get(2, 3), Some(WHITE));
        assert_eq!(c.get(15, 12), Some(WHITE));
    }

    #[test]
    fn half_see_through_rectangle_mixes_colors() {
        let mut c = Canvas::new(2, 2);
        c.fill_rect(0, 0, 2, 2, 0xFFFFFF, 0.5);
        let p = c.get(0, 0).unwrap();
        assert_eq!(p & 0xFF, 128); // black mixed half way to white
    }

    #[test]
    fn text_draws_pixels_and_advances_8_per_letter() {
        let mut c = Canvas::new(40, 10);
        let advance = c.text(0, 0, "Hi", WHITE, 1);
        assert_eq!(advance, 16);
        assert!(c.buf.iter().any(|&p| p == WHITE));
        // Nothing is drawn below the 8 pixel high glyphs.
        assert!((0..40).all(|x| c.get(x, 9) == Some(0)));
    }

    #[test]
    fn wrap_text_respects_the_line_length() {
        let lines = wrap_text("Someone is waving at you on the center of the image", 20);
        assert!(lines.len() >= 3);
        assert!(lines.iter().all(|l| l.chars().count() <= 20), "{:?}", lines);
        assert_eq!(lines.join(" "), "Someone is waving at you on the center of the image");
    }

    #[test]
    fn wrap_text_keeps_a_long_word_whole() {
        assert_eq!(wrap_text("abcdefghijkl", 5), vec!["abcdefghijkl".to_string()]);
    }

    #[test]
    fn flash_lights_its_cell_and_fades_out() {
        let flash = Flash {
            col: 0,
            row: 0,
            brightness: 1.0,
            at_s: 10.0,
        };
        let cell_pixel = |now: f64| {
            let mut c = Canvas::new(200, 120);
            draw_flash_grid(&mut c, Some(flash), now);
            let x0 = 200 - 14 * GRID as i32 - 8;
            // a pixel in the middle of cell (0, 0)
            c.get(x0 + 7, 26 + 7).unwrap()
        };
        let lit = cell_pixel(10.0);
        let fading = cell_pixel(10.2);
        let gone = cell_pixel(10.5);
        let yellow_amount = |p: u32| (p >> 16) & 0xFF; // red channel (yellow has high red)
        assert!(yellow_amount(lit) > yellow_amount(fading));
        assert!(yellow_amount(fading) > yellow_amount(gone));
    }

    #[test]
    fn render_draws_pose_status_and_event_text() {
        let frame = solid(320, 240, [10, 10, 10]);
        let mut keypoints = [Keypoint::default(); neurosight_common::KEYPOINT_COUNT];
        keypoints[kp::LEFT_SHOULDER] = Keypoint { x: 0.6, y: 0.4, score: 0.9 };
        keypoints[kp::RIGHT_SHOULDER] = Keypoint { x: 0.4, y: 0.4, score: 0.9 };
        let pose = Pose::new(keypoints);
        let overlay = Overlay {
            status: "POSE MODE",
            event_text: "Someone is waving",
            pose: Some(&pose),
            flash: None,
            now_s: 0.0,
        };
        let mut c = Canvas::new(320, 240);
        render(&mut c, &frame, &overlay, false);
        // The shoulder line (green) is drawn between the two shoulder points.
        assert_eq!(c.get(160, (0.4_f32 * 240.0).round() as i32), Some(GREEN));
        // The status text uses white pixels in the top strip.
        assert!((0..320).any(|x| (0..18).any(|y| c.get(x, y) == Some(WHITE))));
        // The event text strip at the bottom contains white pixels too.
        assert!((0..320).any(|x| (200..240).any(|y| c.get(x, y) == Some(WHITE))));
    }

    fn ui_state(camera_on: bool) -> UiState {
        UiState {
            camera_on,
            mode: ModeChoice::Pose,
            stim: true,
            mirror: false,
            message: String::new(),
        }
    }

    fn empty_overlay() -> Overlay<'static> {
        Overlay {
            status: "",
            event_text: "",
            pose: None,
            flash: None,
            now_s: 0.0,
        }
    }

    #[test]
    fn window_with_camera_off_shows_a_note_and_the_panel() {
        let mut canvas = Canvas::new(WINDOW_W, WINDOW_H);
        let mut view = Canvas::new(1, 1);
        render_window(
            &mut canvas,
            &mut view,
            None,
            &empty_overlay(),
            &ui_state(false),
            None,
        );
        // The note "Camera is off" is white text in the camera area.
        assert!((0..VIDEO_W as i32).any(|x| (200..224).any(|y| canvas.get(x, y) == Some(WHITE))));
        // The camera button is red.
        let r = ButtonId::Camera.rect();
        assert_eq!(canvas.get(r.x + 3, r.y + 3), Some(0xB23B3B));
    }

    #[test]
    fn window_with_camera_on_centers_the_picture() {
        let mut canvas = Canvas::new(WINDOW_W, WINDOW_H);
        let mut view = Canvas::new(1, 1);
        let frame = solid(320, 240, [200, 0, 0]);
        render_window(
            &mut canvas,
            &mut view,
            Some(&frame),
            &empty_overlay(),
            &ui_state(true),
            None,
        );
        // The picture is centered in the 640 x 480 camera area: (160, 120) is its corner.
        assert_eq!(canvas.get(160 + 10, 120 + 120), Some(0xC80000));
        // Outside the picture there is the dark background.
        assert_eq!(canvas.get(10, 10), Some(BACKGROUND));
        // The camera button is green.
        let r = ButtonId::Camera.rect();
        assert_eq!(canvas.get(r.x + 3, r.y + 3), Some(0x2E7D32));
    }

    #[test]
    fn blit_cuts_off_what_does_not_fit() {
        let mut big = Canvas::new(4, 4);
        let mut small = Canvas::new(2, 2);
        small.buf = vec![1, 2, 3, 4];
        big.blit(&small, 3, 3); // only the top-left pixel of `small` fits
        assert_eq!(big.get(3, 3), Some(1));
        big.blit(&small, -1, -1); // only the bottom-right pixel of `small` fits
        assert_eq!(big.get(0, 0), Some(4));
    }

    #[test]
    fn low_confidence_points_are_not_drawn() {
        let frame = solid(100, 100, [0, 0, 0]);
        let mut keypoints = [Keypoint::default(); neurosight_common::KEYPOINT_COUNT];
        keypoints[kp::LEFT_SHOULDER] = Keypoint { x: 0.6, y: 0.5, score: 0.1 };
        keypoints[kp::RIGHT_SHOULDER] = Keypoint { x: 0.4, y: 0.5, score: 0.1 };
        let pose = Pose::new(keypoints);
        let overlay = Overlay {
            status: "",
            event_text: "",
            pose: Some(&pose),
            flash: None,
            now_s: 0.0,
        };
        let mut c = Canvas::new(100, 100);
        render(&mut c, &frame, &overlay, false);
        assert_ne!(c.get(50, 50), Some(GREEN));
        assert_ne!(c.get(60, 50), Some(WHITE));
    }
}
