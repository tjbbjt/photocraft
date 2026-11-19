//! Edit › Fill… (Shift+F5, Shift+Backspace): Photoshop's Fill dialog in front of `edit.fill`.
//!
//! Contents (Foreground, Background, Color…, Content-Aware, Pattern, History, Black, 50% Gray,
//! White) with the content's own options, then Blending: Mode, Opacity and Preserve
//! Transparency. The fields are the command's params, so `ui.dialog.set` / `ui.dialog.confirm`
//! drive it like any dialog. OK remembers the choices in the preferences (`dialogs["edit.fill"]`),
//! which are saved to disk, so the dialog opens with them again after a restart.
//!
//! A PhotoCraft features switch at the top: off (parity), the dialog and its fill are
//! Photoshop's (`edit.fill`, on the spot); on, Content-Aware runs the Content-Aware Fill command
//! in the background with the same sampling, and adds its fill methods (Quality, LaMa, MI-GAN, their mix)
//! and the edge band that blends the fill into the picture.

use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

pub const COMMAND: &str = "edit.fill";
const MARK: &str = "__fill";

/// The Contents menu: (param, label), in Photoshop's order.
pub const CONTENTS: [(&str, &str); 9] = [
    ("foreground", "Foreground Color"),
    ("background", "Background Color"),
    ("color", "Color…"),
    ("contentAware", "Content-Aware"),
    ("pattern", "Pattern"),
    ("history", "History"),
    ("black", "Black"),
    ("gray", "50% Gray"),
    ("white", "White"),
];

/// Params the dialog edits and remembers.
const KEYS: [&str; 12] =
    ["contents", "color", "pattern", "colorAdaptation", "mode", "opacity", "preserveTransparency", "fillMode", "method", "detail", "edgeGrow", "edgeFeather"];

/// Content-Aware's methods: (param, label).
pub const METHODS: [(&str, &str); 5] =
    [("classic", "Classic"), ("quality", "Quality (Classic + LaMa)"), ("lama", "LaMa"), ("migan", "MI-GAN"), ("mix", "Mix (LaMa + MI-GAN)")];

/// The edge band PhotoCraft starts with, in px: grown outright, then feathered.
pub const EDGE_GROW: f64 = 2.0;
pub const EDGE_FEATHER: f64 = 8.0;
/// The most either may be (the engine's `Edge::MAX`).
pub const EDGE_MAX: f64 = 256.0;

/// Is this dialog the Fill dialog?
pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

fn hex(c: [f32; 4]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

/// A remembered value, if it is one the dialog can show (a corrupt preference is ignored).
fn valid(app: &PhotocraftApp, key: &str, v: &Value) -> bool {
    match key {
        "contents" => v.as_str().is_some_and(|c| CONTENTS.iter().any(|(k, _)| *k == c)),
        "mode" | "opacity" | "preserveTransparency" => crate::dialog_blend_ui::valid(key, v),
        "color" => v.as_str().is_some_and(|h| h.len() == 7 && h.starts_with('#') && h.get(1..).is_some_and(|d| d.chars().all(|c| c.is_ascii_hexdigit()))),
        "pattern" => v.as_str().is_some_and(|p| app.session.patterns.items.iter().any(|q| q.id == p)),
        "colorAdaptation" | "detail" => v.is_boolean(),
        "fillMode" => matches!(v.as_str(), Some("parity" | "photocraft")),
        "method" => v.as_str().is_some_and(|m| METHODS.iter().any(|(k, _)| *k == m)),
        "edgeGrow" | "edgeFeather" => v.as_f64().is_some_and(|n| (0.0..=EDGE_MAX).contains(&n)),
        _ => false,
    }
}

/// Photoshop's Fill defaults (no last-used prefs). Used by open and Alt+Reset.
pub fn default_fields(app: &PhotocraftApp) -> Map<String, Value> {
    let mut f = Map::new();
    f.insert(MARK.into(), json!(true));
    f.insert("__command".into(), json!(COMMAND));
    f.insert("__label".into(), json!("Fill"));
    f.insert("contents".into(), json!("foreground"));
    f.insert("color".into(), json!(hex(app.session.tools.foreground)));
    if let Some(p) = app.session.patterns.items.first() {
        f.insert("pattern".into(), json!(p.id));
    }
    f.insert("colorAdaptation".into(), json!(true));
    f.insert("fillMode".into(), json!("photocraft"));
    f.insert("method".into(), json!("classic"));
    f.insert("detail".into(), json!(true));
    f.insert("edgeGrow".into(), json!(EDGE_GROW));
    f.insert("edgeFeather".into(), json!(EDGE_FEATHER));
    f.insert("mode".into(), json!("normal"));
    f.insert("opacity".into(), json!(100.0));
    f.insert("preserveTransparency".into(), json!(false));
    f
}

/// The dialog's fields: Photoshop's defaults, overridden by the last choices.
pub fn fields(app: &PhotocraftApp) -> Map<String, Value> {
    let mut f = default_fields(app);
    if let Some(Value::Object(saved)) = app.session.prefs().dialogs.get(COMMAND) {
        for k in KEYS {
            if let Some(v) = saved.get(k).filter(|v| valid(app, k, v)) {
                f.insert(k.into(), v.clone());
            }
        }
    }
    f
}

/// Alt+Cancel (Reset): Photoshop Fill defaults, not last-used prefs. Dialog stays open.
pub fn reset(app: &PhotocraftApp, fields: &mut Map<String, Value>) {
    if !owns(fields) {
        return;
    }
    let defaults = default_fields(app);
    for k in KEYS {
        if let Some(v) = defaults.get(k) {
            fields.insert(k.into(), v.clone());
        }
    }
}

/// Open Fill with the current editing target, so live UI changes cannot redirect confirmation.
pub fn open(app: &mut PhotocraftApp) -> u64 {
    let mut f = fields(app);
    let target = app.with_mask_target(COMMAND, json!({}));
    let target = target.get("target").cloned().unwrap_or_else(|| crate::canvas::paint_target(app));
    if let Some(st) = app.session.active() {
        f.insert("__document".into(), json!(st.doc.id.0));
        // Alpha and Quick Mask belong to the document, independently of its active layer.
        if target.as_str() != Some("quickMask") && target.get("channel").is_none() {
            f.insert("__layer".into(), json!(st.active_layer));
        }
    }
    f.insert("__target".into(), target);
    app.ui.open_dialog(crate::state::DialogKind::Command, f)
}

/// The `edit.fill` params of the dialog's fields: only the options of the chosen contents.
pub fn params(f: &Map<String, Value>) -> Value {
    let contents = f.get("contents").and_then(Value::as_str).unwrap_or("foreground");
    let mut p = json!({ "contents": contents });
    for k in ["mode", "opacity", "preserveTransparency"] {
        if let Some(v) = f.get(k) {
            p[k] = v.clone();
        }
    }
    let extra = match contents {
        "color" => Some("color"),
        "pattern" => Some("pattern"),
        "contentAware" => Some("colorAdaptation"),
        _ => None,
    };
    if let Some(k) = extra
        && let Some(v) = f.get(k)
    {
        p[k] = v.clone();
    }
    if let Some(target) = f.get("__target") {
        p["target"] = target.clone();
    }
    p
}

/// Content-Aware the PhotoCraft way: the Content-Aware Fill command's params (sampling the whole
/// window, as Fill does), or `None` when Fill itself runs (Parity, other contents, or a target
/// that is not layer pixels, which Fill refuses with its own message).
pub fn photocraft_params(f: &Map<String, Value>) -> Option<Value> {
    let get = |k: &str| f.get(k).and_then(Value::as_str);
    if get("contents") != Some("contentAware") || get("fillMode").unwrap_or("photocraft") != "photocraft" {
        return None;
    }
    if f.get("__target").is_some_and(|t| t.as_str() != Some("pixels")) {
        return None;
    }
    let adapt = f.get("colorAdaptation").and_then(Value::as_bool).unwrap_or(true);
    let px = |k: &str, d: f64| f.get(k).and_then(Value::as_f64).unwrap_or(d).clamp(0.0, EDGE_MAX).round();
    let mut p = json!({
        "sampling": "rectangular",
        "colorAdaptation": if adapt { "default" } else { "none" },
        "method": get("method").unwrap_or("classic"),
        "detail": f.get("detail").and_then(Value::as_bool).unwrap_or(true),
        "edgeGrow": px("edgeGrow", EDGE_GROW),
        "edgeFeather": px("edgeFeather", EDGE_FEATHER),
        "output": "current",
    });
    if let Some(layer) = f.get("__layer").filter(|l| l.is_u64()) {
        p["layer"] = layer.clone();
    }
    Some(p)
}

/// OK: remember the choices, then fill.
pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    let st = app.session.active().ok_or(tl!("The document is closed"))?;
    if f.get("__document").and_then(Value::as_u64) != Some(st.doc.id.0) || f.get("__layer").is_some_and(|layer| layer != &json!(st.active_layer)) {
        return Err(tl!("The document or active layer changed. Open Fill again.").into());
    }
    let remembered: Map<String, Value> = KEYS.iter().filter_map(|k| f.get(*k).map(|v| (k.to_string(), v.clone()))).collect();
    app.session.prefs.edit(|p| p.dialogs.insert(COMMAND.into(), Value::Object(remembered)));
    match photocraft_params(f) {
        Some(p) => app.run("edit.contentAwareFill", p),
        None => app.run(COMMAND, params(f)),
    }
}

/// Label column bounds for [`crate::widgets::label_column_width`]: the English labels' width, and a cap.
const LABEL_MIN_W: f32 = 112.0;
const LABEL_MAX_W: f32 = 240.0;
/// The labels above the Blending section, measured with its own in the current language.
const LABELS: [&str; 3] = ["Contents:", "Color:", "Custom Pattern:"];

fn get_str(f: &Map<String, Value>, k: &str, d: &str) -> String {
    f.get(k).and_then(Value::as_str).unwrap_or(d).to_string()
}

fn parse_hex(h: &str) -> [u8; 3] {
    let d = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).unwrap_or(0);
    [d(1), d(3), d(5)]
}

/// The dialog body.
pub fn body(app: &PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    ui.spacing_mut().item_spacing.y = 6.0;
    let labels: Vec<&str> = LABELS.iter().chain(&crate::dialog_blend_ui::LABELS).copied().map(crate::i18n::t).collect();
    let label_w = crate::widgets::label_column_width(ui, &labels, LABEL_MIN_W, LABEL_MAX_W);
    let label = |ui: &mut egui::Ui, text: &str| {
        crate::widgets::column_label(ui, text, label_w);
    };
    let photocraft = get_str(f, "fillMode", "photocraft") == "photocraft";
    let mut contents = get_str(f, "contents", "foreground");
    ui.horizontal(|ui| {
        label(ui, tl!("Contents:"));
        let opts: Vec<(String, &str)> = CONTENTS.iter().map(|(k, l)| (k.to_string(), *l)).collect();
        if crate::widgets::dropdown(ui, "fill-contents", tl!("Contents:"), &mut contents, &opts, 170.0) {
            f.insert("contents".into(), json!(contents));
            // Choosing Color… opens the Color Picker, as in Photoshop.
            if contents == "color" {
                crate::color_picker_ui::request(f, tl!("Color Picker (Fill Color)"));
            }
        }
    });
    match contents.as_str() {
        "color" => {
            ui.horizontal(|ui| {
                label(ui, tl!("Color:"));
                // Photoshop shows no swatch; this one reopens the Color Picker on the colour.
                let rgb = parse_hex(&get_str(f, "color", "#000000"));
                if crate::widgets::color_swatch_button(ui, egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]), tl!("Fill color")).clicked() {
                    crate::color_picker_ui::request(f, tl!("Color Picker (Fill Color)"));
                }
            });
        }
        "pattern" => {
            ui.horizontal(|ui| {
                label(ui, tl!("Custom Pattern:"));
                let pat = get_str(f, "pattern", "");
                if app.session.patterns.items.is_empty() {
                    ui.label(egui::RichText::new(tl!("No patterns")).color(Tokens::get(ui.ctx()).text_faint));
                } else {
                    // Photoshop shows the pattern itself and opens a grid of swatches.
                    if let Some(id) = crate::preset_panels::pattern_picker(app, ui, &pat) {
                        f.insert("pattern".into(), json!(id));
                    }
                    let name = app.session.patterns.items.iter().find(|p| p.id == get_str(f, "pattern", "")).map(|p| p.display_name().to_string());
                    ui.label(egui::RichText::new(name.unwrap_or_default()).color(Tokens::get(ui.ctx()).text_dim));
                }
            });
        }
        "contentAware" => {
            ui.horizontal(|ui| {
                label(ui, "");
                let mut on = f.get("colorAdaptation").and_then(Value::as_bool).unwrap_or(true);
                if crate::widgets::checkbox(ui, &mut on, "Color Adaptation").changed() {
                    f.insert("colorAdaptation".into(), json!(on));
                }
            });
            if photocraft {
                content_aware_body(ui, f, &label);
            }
        }
        _ => {}
    }
    // The Content-Aware Fill command fills at full strength into the layer: Blending is Fill's.
    let blending = photocraft_params(f).is_none();
    ui.add_enabled_ui(blending, |ui| crate::dialog_blend_ui::body(ui, f, "fill", label_w));
    if !blending {
        ui.label(egui::RichText::new(tl!("Blending applies to Parity fills.")).color(Tokens::get(ui.ctx()).text_faint).size(11.0));
    }
    // At the bottom left: PhotoCraft's additions, or the dialog as Photoshop has it.
    ui.add_space(6.0);
    let mut on = photocraft;
    if behavior_switch(ui, &mut on)
        .on_hover_text(tl!("PhotoCraft: Content-Aware gets fill methods and the edge band, and runs in the background. Parity: Fill as in Photoshop."))
        .changed()
    {
        f.insert("fillMode".into(), json!(if on { "photocraft" } else { "parity" }));
    }
}

/// PhotoCraft's switch between its own behaviour (left) and Photoshop's (Parity, right); a click
/// on the switch flips it, a click on a side picks that side. Edit › Fill and the Content-Aware
/// Fill workspace share it.
pub fn behavior_switch(ui: &mut egui::Ui, photocraft: &mut bool) -> egui::Response {
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        let side = |ui: &mut egui::Ui, text: &str, on: bool| {
            ui.add(egui::Label::new(egui::RichText::new(text).color(if on { t.text } else { t.text_dim })).sense(egui::Sense::click()))
        };
        let left = side(ui, tl!("PhotoCraft"), *photocraft);
        let (rect, switch) = ui.allocate_exact_size(egui::vec2(34.0, 18.0), egui::Sense::click());
        let how = ui.ctx().animate_bool(switch.id, !*photocraft);
        ui.painter().rect_filled(rect, 9.0, if *photocraft { t.accent } else { t.field_border });
        let x = egui::lerp((rect.left() + 9.0)..=(rect.right() - 9.0), how);
        ui.painter().circle_filled(egui::pos2(x, rect.center().y), 7.0, egui::Color32::WHITE);
        let right = side(ui, tl!("Parity"), !*photocraft);
        let was = *photocraft;
        if left.clicked() {
            *photocraft = true;
        }
        if right.clicked() {
            *photocraft = false;
        }
        if switch.clicked() {
            *photocraft = !was;
        }
        let mut r = left.union(switch).union(right);
        if *photocraft != was {
            r.mark_changed();
        }
        r
    })
    .inner
}

/// PhotoCraft's Content-Aware options: the method and the edge band.
fn content_aware_body(ui: &mut egui::Ui, f: &mut Map<String, Value>, label: &dyn Fn(&mut egui::Ui, &str)) {
    let mut method = get_str(f, "method", "classic");
    ui.horizontal(|ui| {
        label(ui, tl!("Method:"));
        let opts: Vec<(String, &str)> = METHODS.iter().map(|(k, l)| (k.to_string(), *l)).collect();
        if crate::widgets::dropdown(ui, "fill-method", tl!("Method:"), &mut method, &opts, 170.0) {
            f.insert("method".into(), json!(method));
        }
    });
    ui.horizontal(|ui| {
        label(ui, "");
        let mut on = f.get("detail").and_then(Value::as_bool).unwrap_or(true);
        ui.add_enabled_ui(!matches!(method.as_str(), "classic" | "quality"), |ui| {
            if crate::widgets::checkbox(ui, &mut on, tl!("Rebuild Detail from the Photo")).changed() {
                f.insert("detail".into(), json!(on));
            }
        });
    });
    for (key, name, default) in [("edgeGrow", tl!("Edge Grow:"), EDGE_GROW), ("edgeFeather", tl!("Edge Feather:"), EDGE_FEATHER)] {
        ui.horizontal(|ui| {
            label(ui, name);
            let mut v = f.get(key).and_then(Value::as_f64).unwrap_or(default) as f32;
            if crate::widgets::value_field(ui, &mut v, 0.0..=EDGE_MAX as f32, "px", 64.0).accessible_name(name).changed() {
                f.insert(key.into(), json!(f64::from(v.round())));
            }
        });
    }
}

#[cfg(test)]
mod tests;
