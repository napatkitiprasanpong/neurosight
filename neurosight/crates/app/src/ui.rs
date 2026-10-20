//! The control panel on the right side of the NeuroSight window: buttons to turn the camera
//! on and off, to choose the mode, and to switch stimulation and mirroring.
//!
//! Everything here is plain data and drawing code (no window calls), so it can be tested.
//! The window itself (mouse and keyboard) lives in `preview.rs`.

use crate::preview::{wrap_text, Canvas};

/// Size of the camera area, of the panel, and of the whole window.
pub const VIDEO_W: usize = 640;
pub const VIDEO_H: usize = 480;
pub const PANEL_W: usize = 200;
pub const WINDOW_W: usize = VIDEO_W + PANEL_W;
pub const WINDOW_H: usize = VIDEO_H;

const PANEL_BG: u32 = 0x1E2227;
const BUTTON: u32 = 0x3A3F45;
const GREEN: u32 = 0x2E7D32;
const RED: u32 = 0xB23B3B;
const BLUE: u32 = 0x1565C0;
const WHITE: u32 = 0xFFFFFF;
const GRAY: u32 = 0x9AA0A6;
const NOTE: u32 = 0xFFE082;

/// Which analysis the program runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeChoice {
    /// Body-pose model: waving, handshake offer, head gestures, appearing / leaving.
    Pose,
    /// Motion only (no model needed): still / moving / waving.
    Motion,
}

impl ModeChoice {
    pub fn name(&self) -> &'static str {
        match self {
            ModeChoice::Pose => "pose mode",
            ModeChoice::Motion => "motion-only mode",
        }
    }
}

/// Something the user asked for with a click or a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiAction {
    ToggleCamera,
    SetMode(ModeChoice),
    ToggleStim,
    ToggleMirror,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonId {
    Camera,
    ModePose,
    ModeMotion,
    Stim,
    Mirror,
}

pub const ALL_BUTTONS: [ButtonId; 5] = [
    ButtonId::Camera,
    ButtonId::ModePose,
    ButtonId::ModeMotion,
    ButtonId::Stim,
    ButtonId::Mirror,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}

impl ButtonId {
    pub fn action(self) -> UiAction {
        match self {
            ButtonId::Camera => UiAction::ToggleCamera,
            ButtonId::ModePose => UiAction::SetMode(ModeChoice::Pose),
            ButtonId::ModeMotion => UiAction::SetMode(ModeChoice::Motion),
            ButtonId::Stim => UiAction::ToggleStim,
            ButtonId::Mirror => UiAction::ToggleMirror,
        }
    }

    pub fn rect(self) -> Rect {
        let x = VIDEO_W as i32 + 8;
        let w = PANEL_W as i32 - 16;
        let h = 36;
        let y = match self {
            ButtonId::Camera => 44,
            ButtonId::ModePose => 120,
            ButtonId::ModeMotion => 162,
            ButtonId::Stim => 222,
            ButtonId::Mirror => 266,
        };
        Rect { x, y, w, h }
    }
}

/// The button under a mouse position (window coordinates), if any.
pub fn button_at(x: i32, y: i32) -> Option<ButtonId> {
    ALL_BUTTONS.into_iter().find(|b| b.rect().contains(x, y))
}

/// What the panel shows.
#[derive(Clone, Debug)]
pub struct UiState {
    pub camera_on: bool,
    pub mode: ModeChoice,
    pub stim: bool,
    pub mirror: bool,
    /// A short note for the user (why something failed, what just changed).
    pub message: String,
}

impl UiState {
    pub fn label(&self, button: ButtonId) -> &'static str {
        match button {
            ButtonId::Camera => {
                if self.camera_on {
                    "Camera ON"
                } else {
                    "Camera OFF"
                }
            }
            ButtonId::ModePose => "Pose mode",
            ButtonId::ModeMotion => "Motion only",
            ButtonId::Stim => {
                if self.stim {
                    "Stim ON"
                } else {
                    "Stim OFF"
                }
            }
            ButtonId::Mirror => {
                if self.mirror {
                    "Mirror ON"
                } else {
                    "Mirror OFF"
                }
            }
        }
    }

    /// Button color: green = on, red = camera off, blue = selected mode, gray = off / not selected.
    fn color(&self, button: ButtonId) -> u32 {
        match button {
            ButtonId::Camera => {
                if self.camera_on {
                    GREEN
                } else {
                    RED
                }
            }
            ButtonId::ModePose => {
                if self.mode == ModeChoice::Pose {
                    BLUE
                } else {
                    BUTTON
                }
            }
            ButtonId::ModeMotion => {
                if self.mode == ModeChoice::Motion {
                    BLUE
                } else {
                    BUTTON
                }
            }
            ButtonId::Stim => {
                if self.stim {
                    GREEN
                } else {
                    BUTTON
                }
            }
            ButtonId::Mirror => {
                if self.mirror {
                    GREEN
                } else {
                    BUTTON
                }
            }
        }
    }
}

/// Draw the panel (the right-hand strip of the window).
pub fn draw_panel(canvas: &mut Canvas, ui: &UiState, hover: Option<ButtonId>) {
    let x0 = VIDEO_W as i32;
    canvas.fill_rect(x0, 0, PANEL_W as i32, canvas.height as i32, PANEL_BG, 1.0);
    canvas.fill_rect(x0, 0, 2, canvas.height as i32, 0x000000, 0.6);
    canvas.text(x0 + 10, 14, "NeuroSight", WHITE, 2);
    canvas.text(x0 + 10, 104, "MODE", GRAY, 1);

    for button in ALL_BUTTONS {
        let r = button.rect();
        let mut color = ui.color(button);
        if hover == Some(button) {
            color = lighten(color);
        }
        canvas.fill_rect(r.x, r.y, r.w, r.h, color, 1.0);
        let label = ui.label(button);
        let text_w = label.chars().count() as i32 * 16;
        canvas.text(
            r.x + (r.w - text_w) / 2,
            r.y + (r.h - 16) / 2,
            label,
            WHITE,
            2,
        );
    }

    // A note for the user (wrapped to the panel width, at most six lines).
    let mut lines = wrap_text(&ui.message, 24);
    if lines.len() > 6 {
        lines.truncate(6);
        if let Some(last) = lines.last_mut() {
            let keep: String = last.chars().take(21).collect();
            *last = format!("{}...", keep);
        }
    }
    for (i, line) in lines.iter().enumerate() {
        canvas.text(x0 + 10, 318 + i as i32 * 11, line, NOTE, 1);
    }

    // Key hints at the bottom.
    let hints = ["Keys: C camera", "P pose   M motion", "S stim   R mirror", "Esc quit"];
    let top = canvas.height as i32 - hints.len() as i32 * 11 - 8;
    for (i, hint) in hints.iter().enumerate() {
        canvas.text(x0 + 10, top + i as i32 * 11, hint, GRAY, 1);
    }
}

fn lighten(color: u32) -> u32 {
    let channel = |shift: u32| -> u32 {
        let c = (color >> shift) & 0xFF;
        (c + (255 - c) / 4).min(255)
    };
    (channel(16) << 16) | (channel(8) << 8) | channel(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> UiState {
        UiState {
            camera_on: true,
            mode: ModeChoice::Pose,
            stim: true,
            mirror: false,
            message: String::new(),
        }
    }

    #[test]
    fn buttons_do_not_overlap_and_stay_inside_the_panel() {
        for (i, a) in ALL_BUTTONS.iter().enumerate() {
            let ra = a.rect();
            assert!(ra.x >= VIDEO_W as i32 && ra.x + ra.w <= WINDOW_W as i32);
            assert!(ra.y >= 0 && ra.y + ra.h <= WINDOW_H as i32);
            for b in ALL_BUTTONS.iter().skip(i + 1) {
                let rb = b.rect();
                let apart = ra.y + ra.h <= rb.y || rb.y + rb.h <= ra.y;
                assert!(apart, "{:?} and {:?} overlap", a, b);
            }
        }
    }

    #[test]
    fn clicking_the_middle_of_each_button_finds_it() {
        for b in ALL_BUTTONS {
            let r = b.rect();
            assert_eq!(button_at(r.x + r.w / 2, r.y + r.h / 2), Some(b));
        }
    }

    #[test]
    fn clicking_the_camera_area_finds_no_button() {
        assert_eq!(button_at(100, 100), None);
        assert_eq!(button_at(0, 0), None);
        assert_eq!(button_at(WINDOW_W as i32 - 1, WINDOW_H as i32 - 1), None);
    }

    #[test]
    fn buttons_map_to_the_right_actions() {
        assert_eq!(ButtonId::Camera.action(), UiAction::ToggleCamera);
        assert_eq!(
            ButtonId::ModePose.action(),
            UiAction::SetMode(ModeChoice::Pose)
        );
        assert_eq!(
            ButtonId::ModeMotion.action(),
            UiAction::SetMode(ModeChoice::Motion)
        );
        assert_eq!(ButtonId::Stim.action(), UiAction::ToggleStim);
        assert_eq!(ButtonId::Mirror.action(), UiAction::ToggleMirror);
    }

    #[test]
    fn labels_follow_the_state() {
        let mut s = state();
        assert_eq!(s.label(ButtonId::Camera), "Camera ON");
        assert_eq!(s.label(ButtonId::Stim), "Stim ON");
        assert_eq!(s.label(ButtonId::Mirror), "Mirror OFF");
        s.camera_on = false;
        s.stim = false;
        s.mirror = true;
        assert_eq!(s.label(ButtonId::Camera), "Camera OFF");
        assert_eq!(s.label(ButtonId::Stim), "Stim OFF");
        assert_eq!(s.label(ButtonId::Mirror), "Mirror ON");
    }

    /// A pixel inside the button but outside its text (left margin).
    fn margin_pixel(c: &Canvas, b: ButtonId) -> u32 {
        let r = b.rect();
        c.get(r.x + 3, r.y + 3).unwrap()
    }

    #[test]
    fn panel_colors_show_the_state() {
        let mut c = Canvas::new(WINDOW_W, WINDOW_H);
        let mut s = state();
        draw_panel(&mut c, &s, None);
        assert_eq!(margin_pixel(&c, ButtonId::Camera), GREEN);
        assert_eq!(margin_pixel(&c, ButtonId::ModePose), BLUE);
        assert_eq!(margin_pixel(&c, ButtonId::ModeMotion), BUTTON);
        assert_eq!(margin_pixel(&c, ButtonId::Stim), GREEN);
        assert_eq!(margin_pixel(&c, ButtonId::Mirror), BUTTON);

        s.camera_on = false;
        s.mode = ModeChoice::Motion;
        s.stim = false;
        s.mirror = true;
        draw_panel(&mut c, &s, None);
        assert_eq!(margin_pixel(&c, ButtonId::Camera), RED);
        assert_eq!(margin_pixel(&c, ButtonId::ModePose), BUTTON);
        assert_eq!(margin_pixel(&c, ButtonId::ModeMotion), BLUE);
        assert_eq!(margin_pixel(&c, ButtonId::Stim), BUTTON);
        assert_eq!(margin_pixel(&c, ButtonId::Mirror), GREEN);
    }

    #[test]
    fn hovering_lightens_a_button() {
        let mut c = Canvas::new(WINDOW_W, WINDOW_H);
        let s = state();
        draw_panel(&mut c, &s, None);
        let normal = margin_pixel(&c, ButtonId::ModeMotion);
        draw_panel(&mut c, &s, Some(ButtonId::ModeMotion));
        let hovered = margin_pixel(&c, ButtonId::ModeMotion);
        assert!(hovered > normal);
    }

    #[test]
    fn labels_fit_inside_their_buttons() {
        let s = state();
        for b in ALL_BUTTONS {
            let text_w = s.label(b).chars().count() as i32 * 16;
            assert!(text_w <= b.rect().w, "{:?}", b);
        }
        let off = UiState {
            camera_on: false,
            stim: false,
            mirror: true,
            ..state()
        };
        for b in ALL_BUTTONS {
            let text_w = off.label(b).chars().count() as i32 * 16;
            assert!(text_w <= b.rect().w, "{:?}", b);
        }
    }

    #[test]
    fn long_messages_are_cut_to_six_lines() {
        let mut c = Canvas::new(WINDOW_W, WINDOW_H);
        let mut s = state();
        s.message = "word ".repeat(100);
        draw_panel(&mut c, &s, None); // must not panic or draw outside the canvas
        // Nothing of the note is drawn below its six-line area (y 318 + 6 * 11 = 384).
        let x_range = VIDEO_W as i32 + 2..WINDOW_W as i32;
        let hints_top = WINDOW_H as i32 - 4 * 11 - 8;
        for y in 386..hints_top - 1 {
            for x in x_range.clone() {
                assert_eq!(c.get(x, y), Some(PANEL_BG), "pixel {},{}", x, y);
            }
        }
    }
}
