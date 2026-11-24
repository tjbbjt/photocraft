//! Edit › Fill… (Shift+F5, Shift+Backspace): Photoshop's Fill dialog in front of `edit.fill`.
//!
//! Contents (Foreground, Background, Color, Content-Aware, Pattern, History, Black, 50% Gray,
//! White) with the content's own options, then Blending: Mode, Opacity and Preserve
//! Transparency. OK remembers the choices in the saved tier.
//!
//! Content-Aware has a PhotoCraft / Parity choice. Parity is Photoshop's (`edit.fill`, on the
//! spot); PhotoCraft runs the Content-Aware Fill command in the background with the same
//! sampling, and adds its fill methods (LaMa, MI-GAN, their mix, each optionally stamped with
//! Classic) and the edge band that blends the fill into the picture.

use craft::ui::contract::Bind;
use craft::ui::dialog::DialogId;
use craft::ui::patterns::form_dialogs::{DialogFields, FieldValue, Param, ParamDialog, ParamKind, ParamResult};
use craft::ui::runtime::env::Label;
use craft::ui::runtime::memory::SaveKey;
use photocraft_color::BlendMode;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::dialogs::blend_options::{blending_params, blending_rows};
use crate::dialogs::params::{captured, open_param};

/// The command of the dialog, which is also its dialog key.
pub const COMMAND: &str = "edit.fill";
const REMEMBER: &str = "photocraft.dialog.edit.fill";

/// The Contents menu: (param, label), in Photoshop's order. Color reads without an ellipsis: choosing it opens nothing.
pub const CONTENTS: [(&str, &str); 9] = [
    ("foreground", "Foreground Color"),
    ("background", "Background Color"),
    ("color", "Color"),
    ("contentAware", "Content-Aware"),
    ("pattern", "Pattern"),
    ("history", "History"),
    ("black", "Black"),
    ("gray", "50% Gray"),
    ("white", "White"),
];

/// Content-Aware's behaviours: (param, label).
pub const FILL_MODES: [(&str, &str); 2] = [("photocraft", "PhotoCraft"), ("parity", "Parity")];

/// Content-Aware's methods: (param, label).
pub const METHODS: [(&str, &str); 5] =
    [("classic", "Classic"), ("classicDepth", "Classic + Depth"), ("lama", "LaMa"), ("migan", "MI-GAN"), ("mix", "Mix (LaMa + MI-GAN)")];

/// The edge band PhotoCraft starts with, in px: grown outright, then feathered.
pub const EDGE_GROW: f64 = 2.0;
pub const EDGE_FEATHER: f64 = 8.0;
/// The most either may be (the engine's `Edge::MAX`).
pub const EDGE_MAX: f64 = 256.0;

/// `#rrggbb` of an sRGB colour: the form the engine's colour params take.
pub(crate) fn hex(c: [f32; 3]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

/// The dialog's rows.
fn rows(foreground: [f32; 3], first_pattern: Option<String>) -> Vec<Param> {
    let contents = CONTENTS.iter().map(|(k, l)| (k.to_string(), Label::from(tl!(l)))).collect();
    let mut rows = vec![
        Param::new("contents", tl!("Contents:"), ParamKind::choice(contents, Some("foreground".into()))),
        Param::new("color", tl!("Color:"), ParamKind::color(Some(foreground))).visible_when("contents", "color"),
        Param::new("pattern", tl!("Custom Pattern:"), ParamKind::pattern(first_pattern)).visible_when("contents", "pattern"),
        Param::new("colorAdaptation", tl!("Color Adaptation"), ParamKind::bool(true)).visible_when("contents", "contentAware"),
    ];
    // PhotoCraft's Content-Aware: its behaviour, then its method and edge band (Parity greys them).
    let modes = FILL_MODES.iter().map(|(k, l)| (k.to_string(), Label::from(tl!(l)))).collect();
    let methods = METHODS.iter().map(|(k, l)| (k.to_string(), Label::from(tl!(l)))).collect();
    let photocraft = |row: Param| row.visible_when("contents", "contentAware").enabled_when("fillMode", "photocraft");
    rows.extend([
        Param::new("fillMode", tl!("Behavior:"), ParamKind::radio(modes, Some("photocraft".into())))
            .visible_when("contents", "contentAware")
            .tooltip(tl!("PhotoCraft: Content-Aware gets fill methods and the edge band, and runs in the background. Parity: Fill as in Photoshop.")),
        photocraft(Param::new("method", tl!("Method:"), ParamKind::choice(methods, Some("classic".into())))),
        photocraft(
            Param::new("stamp", tl!("Stamp with Classic"), ParamKind::bool(true))
                .tooltip(tl!("LaMa, MI-GAN and Mix: Classic fills around the model's layout.")),
        ),
        photocraft(
            Param::new("matchDepth", tl!("Match Depth"), ParamKind::bool(true))
                .tooltip(tl!("Stamped fills copy from the same depth (Classic + Depth always does).")),
        ),
        photocraft(Param::new("detail", tl!("Rebuild Detail from the Photo"), ParamKind::bool(true)).tooltip(tl!("LaMa, MI-GAN and Mix without stamping."))),
        photocraft(Param::new("edgeGrow", tl!("Edge Grow:"), ParamKind::number(Some(0.0), Some(EDGE_MAX), EDGE_GROW, 1.0, 0))),
        photocraft(Param::new("edgeFeather", tl!("Edge Feather:"), ParamKind::number(Some(0.0), Some(EDGE_MAX), EDGE_FEATHER, 1.0, 0))),
    ]);
    rows.extend(blending_rows());
    rows
}

/// Photoshop's Fill defaults, which Alt+Reset restores instead of the last-used values.
fn defaults(foreground: [f32; 3], first_pattern: Option<String>) -> DialogFields {
    let mut f = DialogFields::new();
    f.set("contents", "foreground");
    f.set("color", FieldValue::Color(foreground));
    if let Some(id) = first_pattern {
        f.set("pattern", id);
    }
    f.set("colorAdaptation", true);
    f.set("fillMode", "photocraft");
    f.set("method", "classic");
    f.set("stamp", true);
    f.set("matchDepth", true);
    f.set("detail", true);
    f.set("edgeGrow", EDGE_GROW);
    f.set("edgeFeather", EDGE_FEATHER);
    f.set("mode", BlendMode::Normal.key().to_string());
    f.set("opacity", 100.0);
    f.set("preserveTransparency", false);
    f
}

/// Open Fill with the current editing target captured, so live UI changes cannot redirect confirmation.
pub fn open(app: &mut PhotocraftApp) -> DialogId {
    let target = app.with_mask_target(COMMAND, json!({}));
    let target = target.get("target").cloned().unwrap_or_else(|| crate::canvas::paint_target(app));
    // Alpha and Quick Mask belong to the document, independently of its active layer.
    let layer_bound = target.as_str() != Some("quickMask") && target.get("channel").is_none();
    let mut held = json!({ "target": target });
    if let Some(st) = app.session.active() {
        held["document"] = json!(st.doc.id.0);
        if layer_bound {
            held["layer"] = json!(st.active_layer);
        }
    }
    let [r, g, b, _] = app.session.tools.foreground;
    let first = app.session.patterns.items.first().map(|p| p.id.clone());
    let patterns = crate::panels::preset_browser::pattern_swatches(app);
    let rows = rows([r, g, b], first.clone());
    let reset = defaults([r, g, b], first);
    let title = Label::from(tl!("Fill"));
    open_param(app, COMMAND, DialogFields::new(), held, move |fields| {
        ParamDialog::new(COMMAND, title.clone(), rows.clone())
            .values(Bind::bound(fields))
            .remember(SaveKey::named(REMEMBER))
            .reset_to(reset.clone())
            .patterns(crate::panels::preset_browser::pattern_source(patterns.clone()))
    })
}

/// The `edit.fill` params of the dialog's fields: only the options of the chosen contents, and the captured target.
pub fn params(f: &DialogFields, held: &Value) -> Value {
    let contents = f.text("contents").unwrap_or("foreground");
    let mut p = blending_params(f);
    p.insert("contents".into(), json!(contents));
    match contents {
        "color" => {
            if let Some(c) = f.color("color") {
                p.insert("color".into(), json!(hex(c)));
            }
        }
        "pattern" => {
            if let Some(id) = f.text("pattern") {
                p.insert("pattern".into(), json!(id));
            }
        }
        "contentAware" => {
            if let Some(on) = f.bool("colorAdaptation") {
                p.insert("colorAdaptation".into(), json!(on));
            }
        }
        _ => {}
    }
    if let Some(target) = held.get("target") {
        p.insert("target".into(), target.clone());
    }
    Value::Object(p)
}

/// Content-Aware the PhotoCraft way: the Content-Aware Fill command's params (sampling the whole
/// window, as Fill does), or `None` when Fill itself runs (Parity, other contents, or a target
/// that is not layer pixels, which Fill refuses with its own message).
pub fn photocraft_params(f: &DialogFields, held: &Value) -> Option<Value> {
    if f.text("contents") != Some("contentAware") || f.text("fillMode").unwrap_or("photocraft") != "photocraft" {
        return None;
    }
    if held.get("target").is_some_and(|t| t.as_str() != Some("pixels")) {
        return None;
    }
    let px = |k: &str, d: f64| f.number(k).unwrap_or(d).clamp(0.0, EDGE_MAX).round();
    let mut p = json!({
        "sampling": "rectangular",
        "colorAdaptation": if f.bool("colorAdaptation").unwrap_or(true) { "default" } else { "none" },
        "method": f.text("method").unwrap_or("classic"),
        "detail": f.bool("detail").unwrap_or(true),
        "stamp": f.bool("stamp").unwrap_or(true),
        "matchDepth": f.bool("matchDepth").unwrap_or(true),
        "edgeGrow": px("edgeGrow", EDGE_GROW),
        "edgeFeather": px("edgeFeather", EDGE_FEATHER),
        "output": "current",
    });
    if let Some(layer) = held.get("layer").filter(|l| l.is_u64()) {
        p["layer"] = layer.clone();
    }
    Some(p)
}

/// Whether the dialog `key` is the Fill dialog.
pub(crate) fn owns(key: &str) -> bool {
    key == COMMAND
}

/// OK is held until the document and layer are the ones the dialog opened on.
pub(crate) fn check(app: &mut PhotocraftApp, result: &ParamResult) -> Result<(), Label> {
    let captured = &captured(app, &result.key);
    let st = app.session.active().ok_or_else(|| Label::from(tl!("The document is closed")))?;
    if captured.get("document").and_then(Value::as_u64) != Some(st.doc.id.0) || captured.get("layer").is_some_and(|layer| layer != &json!(st.active_layer)) {
        return Err(Label::from(tl!("The document or active layer changed. Open Fill again.")));
    }
    Ok(())
}

/// Fill with the confirmed choices; they are remembered only when the fill succeeded.
pub(crate) fn run(app: &mut PhotocraftApp, ctx: &egui::Context, result: ParamResult) {
    let captured = captured(app, &result.key);
    let ran = match photocraft_params(&result.fields, &captured) {
        Some(p) => app.run("edit.contentAwareFill", p),
        None => app.run(COMMAND, params(&result.fields, &captured)),
    };
    match ran {
        Ok(_) => result.remember(ctx),
        Err(e) => crate::notices::error(app, e),
    }
}

#[cfg(test)]
mod tests;
