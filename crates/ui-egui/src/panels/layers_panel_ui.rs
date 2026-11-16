//! Layers panel flyout and display preferences, observed from Photoshop's panel menu.
//! UI options do not edit document pixels or add history steps.
use egui::{self, Context, Ui};
use photocraft_doc::Document;
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::{LayerThumbnailContents, LayerThumbnailSize, LayersPanelDialog, LayersPanelOptions};

const OPTIONS_PREF: &str = "ui.layersPanelOptions";

/// Save independently of Workspace > Remember Workspace Changes, as Photoshop does. A pick of the
/// current value writes nothing unless the saved entry is missing or stale.
pub fn persist_options(app: &mut PhotocraftApp) {
    if let Ok(value) = serde_json::to_value(&app.ui.layers_panel_options) {
        if app.session.prefs().dialogs.get(OPTIONS_PREF) == Some(&value) {
            return;
        }
        app.session.prefs.edit(|p| {
            p.dialogs.insert(OPTIONS_PREF.into(), value);
        });
    }
}

/// Restore independently from a saved workspace. Older preferences retain the defaults.
pub fn restore_options(app: &mut PhotocraftApp) {
    if let Some(opts) = app.session.prefs().dialogs.get(OPTIONS_PREF)
        && let Ok(value) = serde_json::from_value(opts.clone())
    {
        app.ui.layers_panel_options = value;
    }
}

/// A panel-menu item running menu command `id` with `params` (`{}` for none) through the real
/// menu command dispatcher, which handles rename's in-place editor, opens a command's dialog where
/// it has one and does other shell routing. Greyed when `params` is `None` (nothing to act on) or
/// where the menu bar greys the command.
pub fn run_menu_command(app: &mut PhotocraftApp, ui: &mut Ui, label: &str, id: &str, params: Option<serde_json::Value>) {
    let on = params.is_some() && crate::menus::is_enabled(app, id);
    if ui.add_enabled(on, egui::Button::new(label)).clicked()
        && let Some(params) = params
    {
        let ctx = ui.ctx().clone();
        if let Err(e) = crate::menus::invoke(app, &ctx, id, params) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
        ui.close();
    }
}

/// Thumbnail Contents as the shared preference has it: Layer Bounds is Clip Thumbnails to Layer
/// Bounds (#4281), which frames each layer's pixels instead of the canvas.
pub fn thumbnail_contents(app: &PhotocraftApp) -> LayerThumbnailContents {
    if app.session.prefs().interface.clip_thumbnails_to_layer_bounds { LayerThumbnailContents::LayerBounds } else { LayerThumbnailContents::EntireDocument }
}

/// Panel Options › OK: the panel's own options, and Thumbnail Contents into the shared preference.
fn apply_options(app: &mut PhotocraftApp, options: LayersPanelOptions) -> Result<(), String> {
    let contents = options.thumbnail_contents;
    app.ui.layers_panel_options = options;
    persist_options(app);
    if contents == thumbnail_contents(app) {
        return Ok(());
    }
    let clip = contents == LayerThumbnailContents::LayerBounds;
    app.run("prefs.set", json!({"path": "interface.clipThumbnailsToLayerBounds", "value": clip})).map(|_| ())
}

/// Preview cache recomputes only when hovering a new target, not each pointer frame.
/// Compositor sees it as a temporary document and never commits it to history.
pub(crate) struct ReorderPreview {
    pub source: photocraft_doc::DocId,
    pub revision: u64,
    pub signature: u64,
    pub key: u64,
    pub document: std::sync::Arc<Document>,
    /// The key of what the canvas showed before this preview (the previous preview, or 0 for the
    /// document) and the area where the two differ (`step_damage`): moving to this preview redraws
    /// only that area. `None` when it isn't known.
    pub from: Option<(u64, photocraft_geom::Rect)>,
}

/// Canvases of this many pixels or more (about 67 MP) show a drop's preview once the pointer rests
/// on the target for [`SETTLE_MS`].
const SETTLE_FROM_PX: u64 = 1 << 26;
/// How long the pointer rests on a drop target before a large canvas shows the layers moved there.
const SETTLE_MS: f64 = 150.0;

/// Cache the on-canvas preview of the layer drop in `payload` (a `layer.moveTo` request), unless Preview on Canvas is off.
/// Marks the destination as hovered every frame; the preview itself is rebuilt only when the target changes. On a large
/// canvas a target the pointer only passes over isn't previewed: each preview was composited in a frame of 50–120 ms on
/// a storyboard, so the drop line lagged behind the pointer. The last preview stays until the pointer rests on the new
/// target for [`SETTLE_MS`].
pub(crate) fn preview_reorder(app: &mut PhotocraftApp, ctx: &Context, payload: &serde_json::Value) {
    if !app.ui.layers_panel_options.preview_on_canvas_when_reordering_layers {
        app.reorder_preview = None;
        return;
    }
    // Mark the destination as hovered even if the cached preview already matches.
    app.reorder_preview_hit = true;
    let Some(st) = app.session.active() else { return };
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    payload.to_string().hash(&mut hasher);
    st.doc.id.0.hash(&mut hasher);
    st.revision.hash(&mut hasher);
    let signature = hasher.finish();
    if app.reorder_preview.as_ref().is_some_and(|cache| cache.signature == signature) {
        return;
    }
    if u64::from(st.doc.size.width) * u64::from(st.doc.size.height) >= SETTLE_FROM_PX {
        let now = crate::gpu_canvas::now_ms();
        let since = match app.reorder_hover {
            Some((hovered, since)) if hovered == signature => since,
            _ => now,
        };
        app.reorder_hover = Some((signature, since));
        let waited = now - since;
        if waited < SETTLE_MS {
            // No input arrives while the pointer rests: a frame comes when it has rested long enough.
            ctx.request_repaint_after(std::time::Duration::from_secs_f64((SETTLE_MS - waited) / 1000.0));
            return;
        }
    }
    let Some(st) = app.session.active() else { return };
    let (source, revision) = (st.doc.id, st.revision);
    let add_copy = app.ui.layers_panel_options.add_copy_to_copied_layers_and_groups;
    let Some(document) = photocraft_engine::layer_multi_cmds::preview_move(&st.doc, payload, add_copy) else {
        app.reorder_preview = None;
        return;
    };
    // What the canvas shows now: the previous preview, or the document. Each preview redrew the
    // whole storyboard (1.1 GP, about a second) where only the layers changing places differ.
    let (from_key, before) = match app.reorder_preview.as_ref().filter(|p| p.source == source && p.revision == revision) {
        Some(p) => (p.key, p.document.clone()),
        None => (0, st.doc.clone()),
    };
    let from = photocraft_engine::layer_multi_cmds::step_damage(&before, &document).map(|area| (from_key, area));
    let key = (1u64 << 60) | (signature & ((1u64 << 60) - 1));
    let document = std::sync::Arc::new(document);
    app.reorder_shown = Some((source, revision, key, document.clone()));
    app.reorder_preview = Some(ReorderPreview { source, revision, signature, key, document, from });
}

/// The area of document `doc` (at `revision`) that differs between what its canvas showed
/// (`seen`: a drop preview's key, or 0 for the document) and what it shows now (`now`): from one
/// drop preview to the next, from the document to a preview, or from the last preview back to
/// the document when the drag is cancelled. `None` (the whole canvas) for anything else.
pub(crate) fn reorder_damage(app: &PhotocraftApp, doc: &Document, revision: u64, seen: u64, now: u64) -> Option<photocraft_geom::Rect> {
    if seen == now {
        return None;
    }
    if let Some(p) = app.reorder_preview.as_ref().filter(|p| p.source == doc.id && p.revision == revision && p.key == now)
        && let Some((from, area)) = p.from
        && from == seen
    {
        return Some(area);
    }
    let (source, shown_revision, key, shown) = app.reorder_shown.as_ref()?;
    if *source != doc.id || *shown_revision != revision || *key != seen || now != 0 {
        return None;
    }
    photocraft_engine::layer_multi_cmds::step_damage(shown, doc)
}

/// The document and key of the drop preview the active canvas shows, if it was built for the
/// document as it is: a drop that lands redraws only where it differs from that preview.
pub(crate) fn shown_reorder(app: &PhotocraftApp) -> Option<(photocraft_doc::DocId, u64)> {
    let st = app.session.active()?;
    let (source, revision, key, _) = app.reorder_shown.as_ref()?;
    (*source == st.doc.id && *revision == st.revision).then_some((*source, *key))
}

/// The cached reorder preview and its compositor cache key for document `index`, if it was built for that document's current revision.
pub(crate) fn display_reorder(app: &PhotocraftApp, index: usize) -> Option<(std::sync::Arc<Document>, u64)> {
    if !app.ui.layers_panel_options.preview_on_canvas_when_reordering_layers {
        return None;
    }
    let st = app.session.documents().get(index)?;
    let cached = app.reorder_preview.as_ref()?;
    (cached.source == st.doc.id && cached.revision == st.revision).then(|| (cached.document.clone(), cached.key))
}

/// Photoshop shows the same flower at three resolutions. This example is original,
/// vector-painted artwork, not an extracted Adobe asset, and it is independent of
/// whichever layer happens to be selected (including adjustment layers).
fn paint_flower_example(ui: &Ui, rect: egui::Rect) {
    use egui::{Color32, Rect as UiRect, Shape, Stroke, pos2, vec2};
    let painter = ui.painter().with_clip_rect(rect);
    let side = rect.width();
    let point = |x: f32, y: f32| pos2(rect.min.x + x * side, rect.min.y + y * side);
    // A monochrome miniature landscape: light sky, shaded hills and a dark meadow.
    for n in 0..22 {
        let y0 = n as f32 / 22.0;
        let y1 = (n + 1) as f32 / 22.0;
        let f = n as f32 / 22.0;
        painter.rect_filled(UiRect::from_min_max(point(0.0, y0), point(1.0, y1)), 0.0, Color32::from_gray((119.0 + f * 30.0) as u8));
    }
    painter.add(Shape::convex_polygon(
        vec![point(0.0, 0.60), point(0.22, 0.36), point(0.46, 0.56), point(0.72, 0.33), point(1.0, 0.62)],
        Color32::from_gray(80),
        Stroke::NONE,
    ));
    painter.add(Shape::convex_polygon(
        vec![point(0.0, 0.76), point(0.32, 0.60), point(0.57, 0.69), point(1.0, 0.51), point(1.0, 1.0), point(0.0, 1.0)],
        Color32::from_gray(64),
        Stroke::NONE,
    ));
    for n in 0..16 {
        let x = ((n * 43 + 13) % 97) as f32 / 96.0;
        let y = 0.78 + ((n * 11 % 19) as f32) / 100.0;
        painter.line_segment([point(x, 1.0), point((x + 0.02).min(1.0), y)], Stroke::new(side * 0.01, Color32::from_gray(104)));
    }
    // A white daisy with layered petals and a dark central disc, in the same
    // composition at all three sizes, so the controls genuinely preview scale.
    let center = point(0.49, 0.57);
    let unit = side * 0.34;
    for n in 0..15 {
        let theta = (n as f32 + 0.5) * std::f32::consts::TAU / 15.0;
        let direction = vec2(theta.cos(), theta.sin());
        let normal = vec2(-theta.sin(), theta.cos());
        let mut outline = Vec::with_capacity(20);
        for i in 0..=9 {
            let t = i as f32 / 9.0;
            let base = center + direction * unit * (0.22 + 0.83 * t);
            outline.push(base + normal * (unit * 0.22 * (std::f32::consts::PI * t).sin()));
        }
        for i in (0..=9).rev() {
            let t = i as f32 / 9.0;
            let base = center + direction * unit * (0.22 + 0.83 * t);
            outline.push(base - normal * (unit * 0.22 * (std::f32::consts::PI * t).sin()));
        }
        painter.add(Shape::convex_polygon(
            outline,
            if n % 3 == 0 { Color32::from_gray(213) } else { Color32::from_gray(238) },
            Stroke::new(0.3, Color32::from_gray(155)),
        ));
    }
    painter.circle_filled(center, unit * 0.27, Color32::from_gray(33));
    painter.circle_filled(center - vec2(unit * 0.065, unit * 0.055), unit * 0.16, Color32::from_gray(55));
    for n in 0..16 {
        let theta = (n as f32) * std::f32::consts::TAU / 16.0;
        let r = unit * if n % 2 == 0 { 0.16 } else { 0.21 };
        painter.circle_filled(center + vec2(theta.cos(), theta.sin()) * r, (side * 0.008).max(0.3), Color32::from_gray(119));
    }
    painter.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::from_gray(92)), egui::StrokeKind::Inside);
}

/// Returns both the actual thumbnail width and the height allocated by the Layers
/// row. Used by painting and by tests to keep the preference functional.
pub(crate) fn thumbnail_geometry(size: LayerThumbnailSize, pro: bool) -> (f32, f32) {
    let thumb: f32 = match size {
        LayerThumbnailSize::None => 0.0,
        LayerThumbnailSize::Small => 18.0,
        LayerThumbnailSize::Medium => {
            if pro {
                24.0
            } else {
                34.0
            }
        }
        LayerThumbnailSize::Large => 48.0,
    };
    let row = if thumb == 0.0 { 23.0 } else { (thumb + 8.0).max(if pro { 32.0 } else { 40.0 }) };
    (thumb, row)
}

/// Content column of the Photoshop-style dialog, with fixed-size radio/preview
/// slots rather than text labels that change the position of each thumbnail.
fn options_body(ui: &mut Ui, draft: &mut LayersPanelOptions) {
    ui.group(|ui| {
        ui.set_min_width(308.0);
        ui.label(egui::RichText::new(tl!("Thumbnail Size")).strong());
        ui.add_space(7.0);
        // The original Photoshop dialog has None with a label, then progressively
        // larger previews, with the radio controls aligned vertically.
        ui.horizontal(|ui| {
            let r = ui.radio_value(&mut draft.thumbnail_size, LayerThumbnailSize::None, tl!("None"));
            r.on_hover_text(tl!("None"));
        });
        ui.add_space(3.0);
        for (size, side, name) in [
            (LayerThumbnailSize::Small, 38.0, tl!("Small")),
            (LayerThumbnailSize::Medium, 68.0, tl!("Medium")),
            (LayerThumbnailSize::Large, 96.0, tl!("Large")),
        ] {
            ui.horizontal(|ui| {
                // Empty radio captions are intentional: the sample itself conveys
                // size. The accessible/hover label still names the setting.
                let radio = ui.radio_value(&mut draft.thumbnail_size, size, "");
                radio.on_hover_text(name);
                ui.add_space(8.0);
                let (rect, response) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click());
                paint_flower_example(ui, rect);
                if response.clicked() {
                    draft.thumbnail_size = size;
                }
                response.on_hover_text(name);
            });
            ui.add_space(5.0);
        }
    });
    ui.add_space(9.0);
    ui.group(|ui| {
        ui.set_min_width(308.0);
        ui.label(egui::RichText::new(tl!("Thumbnail Contents")).strong());
        ui.add_space(10.0);
        ui.radio_value(&mut draft.thumbnail_contents, LayerThumbnailContents::LayerBounds, tl!("Layer Bounds"));
        ui.add_space(4.0);
        ui.radio_value(&mut draft.thumbnail_contents, LayerThumbnailContents::EntireDocument, tl!("Entire Document"));
        ui.add_space(4.0);
    });
    ui.add_space(9.0);
    ui.checkbox(&mut draft.use_default_masks_on_fill_layers, tl!("Use Default Masks on Fill Layers"));
    ui.add_space(3.0);
    ui.checkbox(&mut draft.expand_new_effects, tl!("Expand New Effects"));
    ui.add_space(3.0);
    ui.checkbox(&mut draft.add_copy_to_copied_layers_and_groups, tl!("Add “copy” to Copied Layers and Groups"));
    ui.add_space(3.0);
    ui.checkbox(&mut draft.preview_on_canvas_when_reordering_layers, tl!("Preview on canvas when reordering layers"));
    ui.add_space(3.0);
    ui.checkbox(&mut draft.show_layer_mask_badges, tl!("Show layer mask badges"));
}

/// Show the pending Layers panel dialog (Panel Options, New Layer, New Group, Lock Layers) as a modal and apply it on OK.
pub fn show_dialog(app: &mut PhotocraftApp, ctx: &Context) {
    let Some(mut dialog) = app.ui.layers_panel_dialog.take() else { return };
    let mut confirmed = false;
    let mut cancelled = ctx.input(|i| i.key_pressed(egui::Key::Escape));
    let title = match &dialog {
        LayersPanelDialog::PanelOptions(_) => tl!("Layers Panel Options"),
        LayersPanelDialog::NewLayer(_) => tl!("New Layer"),
        LayersPanelDialog::NewGroup(_) => tl!("New Group"),
        LayersPanelDialog::LockLayers { .. } => tl!("Lock Layers"),
    };
    egui::Modal::new(egui::Id::unique("layers-panel-options-modal")).show(ctx, |ui| {
        ui.set_width(if matches!(dialog, LayersPanelDialog::PanelOptions(_)) { 480.0 } else { 380.0 });
        ui.horizontal(|ui| {
            ui.heading(title);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("×").on_hover_text(tl!("Cancel")).clicked() {
                    cancelled = true;
                }
            });
        });
        ui.separator();
        match &mut dialog {
            LayersPanelDialog::PanelOptions(opts) => {
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                        ui.set_width(345.0);
                        ui.vertical(|ui| options_body(ui, opts));
                    });
                    ui.add_space(10.0);
                    ui.vertical(|ui| {
                        ui.set_min_width(74.0);
                        if ui.add_sized([74.0, 29.0], egui::Button::new(tl!("OK"))).clicked() {
                            confirmed = true;
                        }
                        ui.add_space(6.0);
                        if ui.add_sized([74.0, 29.0], egui::Button::new(tl!("Cancel"))).clicked() {
                            cancelled = true;
                        }
                    });
                });
            }
            LayersPanelDialog::NewLayer(name) | LayersPanelDialog::NewGroup(name) => {
                ui.label(tl!("Name:"));
                ui.text_edit_singleline(name);
            }
            LayersPanelDialog::LockLayers { transparency, pixels, position, artboard, all } => {
                ui.checkbox(transparency, tl!("Transparency"));
                ui.checkbox(pixels, tl!("Image"));
                ui.checkbox(position, tl!("Position"));
                ui.checkbox(artboard, tl!("Prevent Auto-Nesting"));
                ui.checkbox(all, tl!("All"));
            }
        }
        if !matches!(dialog, LayersPanelDialog::PanelOptions(_)) {
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button(tl!("OK")).clicked() {
                    confirmed = true;
                }
                if ui.button(tl!("Cancel")).clicked() {
                    cancelled = true;
                }
            });
        }
    });
    if confirmed && !cancelled {
        let r = match dialog {
            LayersPanelDialog::PanelOptions(options) => apply_options(app, options),
            LayersPanelDialog::NewLayer(name) => {
                app.run("layer.new.layer", if name.trim().is_empty() { json!({}) } else { json!({"name": name.trim()}) }).map(|_| ())
            }
            LayersPanelDialog::NewGroup(name) => {
                app.run("layer.new.group", if name.trim().is_empty() { json!({}) } else { json!({"name": name.trim()}) }).map(|_| ())
            }
            LayersPanelDialog::LockLayers { transparency, pixels, position, artboard, all } => app
                .run("layer.lockLayers", json!({"transparency": transparency, "pixels": pixels, "position": position, "artboard": artboard, "all": all}))
                .map(|_| ()),
        };
        if let Err(e) = r {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    } else if !cancelled {
        app.ui.layers_panel_dialog = Some(dialog);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A document of `width` × `height` with layers "a" and "b" over the Background, and the
    /// drop of "a" above "b" as the Layers panel sends it.
    fn reorder_app(width: u32, height: u32) -> (PhotocraftApp, serde_json::Value) {
        let mut session = photocraft_engine::Session::new();
        session.execute("file.new", json!({"width": width, "height": height})).unwrap();
        let a = session.execute("layer.new.layer", json!({"name": "a"})).unwrap()["layer"].as_u64().unwrap();
        let b = session.execute("layer.new.layer", json!({"name": "b"})).unwrap()["layer"].as_u64().unwrap();
        (PhotocraftApp::new(session, crate::Services::default()), json!({"layer": a, "target": b, "position": "above"}))
    }

    /// Going from the document to a drop preview, and back to the document when the drag is
    /// cancelled, redraws only where the two differ, not the whole canvas.
    #[test]
    fn a_drop_preview_redraws_only_where_it_differs() {
        let ctx = egui::Context::default();
        let (mut app, drop) = reorder_app(64, 64);
        preview_reorder(&mut app, &ctx, &drop);
        let st = app.session.active().unwrap();
        let (doc, revision) = (st.doc.clone(), st.revision);
        let p = app.reorder_preview.as_ref().expect("a preview");
        let (key, there) = (p.key, photocraft_engine::layer_multi_cmds::step_damage(&doc, &p.document));
        assert!(there.is_some(), "a reorder's damage is known");
        assert_eq!(reorder_damage(&app, &doc, revision, 0, key), there);
        assert_eq!(shown_reorder(&app).map(|(_, k)| k), Some(key), "the drop would land on what the canvas shows");
        // The drag is cancelled: the canvas goes back to the document.
        app.reorder_preview = None;
        let back = photocraft_engine::layer_multi_cmds::step_damage(&app.reorder_shown.as_ref().unwrap().3, &doc);
        assert_eq!(reorder_damage(&app, &doc, revision, key, 0), back);
        assert_eq!(reorder_damage(&app, &doc, revision + 1, key, 0), None, "not once the document has changed");
    }

    /// On a large canvas the drop's preview waits until the pointer rests on the target, so
    /// passing over rows doesn't composite a preview for each; a small canvas shows it at once.
    #[test]
    fn a_large_canvas_previews_a_drop_once_the_pointer_rests_on_it() {
        let ctx = egui::Context::default();
        let (mut small, drop) = reorder_app(64, 64);
        preview_reorder(&mut small, &ctx, &drop);
        assert!(small.reorder_preview.is_some(), "a small canvas previews at once");

        let (mut large, drop) = reorder_app(9000, 8000);
        preview_reorder(&mut large, &ctx, &drop);
        assert!(large.reorder_preview.is_none(), "not while the pointer passes over the target");
        assert!(large.reorder_preview_hit, "the target still counts as hovered");
        std::thread::sleep(std::time::Duration::from_millis((SETTLE_MS as u64) + 50));
        preview_reorder(&mut large, &ctx, &drop);
        assert!(large.reorder_preview.is_some(), "the preview once it has rested there");
    }

    /// A fresh install shows Medium thumbnails of the whole document, like Photoshop's Panel Options defaults.
    #[test]
    fn default_panel_options_match_medium_document_previews() {
        let options = LayersPanelOptions::default();
        assert_eq!(options.thumbnail_size, LayerThumbnailSize::Medium);
        assert_eq!(options.thumbnail_contents, LayerThumbnailContents::EntireDocument);
        assert!(options.show_filters);
        assert!(options.use_default_masks_on_fill_layers);
        assert!(options.expand_new_effects);
        assert!(options.add_copy_to_copied_layers_and_groups);
        assert!(options.preview_on_canvas_when_reordering_layers);
        assert!(!options.show_layer_mask_badges);
    }

    /// Thumbnail Contents is the same setting as the panel menu's Clip Thumbnails to Layer Bounds
    /// (#4281): the dialog shows it and OK sets it, so the two never disagree.
    #[test]
    fn thumbnail_contents_is_clip_thumbnails_to_layer_bounds() {
        let mut session = photocraft_engine::Session::new();
        session.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        let mut app = PhotocraftApp::new(session, crate::Services::default());
        assert_eq!(thumbnail_contents(&app), LayerThumbnailContents::EntireDocument);
        let mut options = app.ui.layers_panel_options.clone();
        options.thumbnail_contents = LayerThumbnailContents::LayerBounds;
        apply_options(&mut app, options.clone()).unwrap();
        assert!(app.session.prefs().interface.clip_thumbnails_to_layer_bounds);
        assert_eq!(thumbnail_contents(&app), LayerThumbnailContents::LayerBounds);
        assert!(app.session.prefs().dialogs[OPTIONS_PREF].get("thumbnailContents").is_none(), "saved once, in the shared preference");
        options.thumbnail_contents = LayerThumbnailContents::EntireDocument;
        apply_options(&mut app, options).unwrap();
        assert!(!app.session.prefs().interface.clip_thumbnails_to_layer_bounds);
    }

    /// The options survive a UiState round trip; an open dialog is view state and is never serialized.
    #[test]
    fn options_round_trip_without_a_pending_dialog() {
        let mut ui = crate::state::UiState::default();
        ui.layers_panel_options.thumbnail_size = LayerThumbnailSize::Large;
        ui.layers_panel_options.show_filters = false;
        ui.layers_panel_dialog = Some(LayersPanelDialog::NewGroup("test".into()));
        let json = serde_json::to_value(&ui).unwrap();
        assert!(json.get("layers_panel_dialog").is_none());
        let restored: crate::state::UiState = serde_json::from_value(json).unwrap();
        assert_eq!(restored.layers_panel_options, ui.layers_panel_options);
        assert!(restored.layers_panel_dialog.is_none());
    }

    /// With the option off a new fill layer has no mask and the mask is not targeted; with it on the new layer's mask is targeted.
    #[test]
    fn disabling_default_fill_masks_never_targets_an_absent_mask() {
        let mut session = photocraft_engine::Session::new();
        session.execute("file.new", json!({"width": 32, "height": 24})).unwrap();
        let mut app = PhotocraftApp::new(session, crate::Services::default());
        app.ui.layers_panel_options.use_default_masks_on_fill_layers = false;
        persist_options(&mut app);
        app.run("layer.newFillLayer.solidColor", json!({"color": "#778899"})).unwrap();
        let layer = app.session.active().and_then(|st| st.active_layer.and_then(|id| st.doc.layer(id))).unwrap();
        assert!(layer.mask.is_none());
        assert!(!app.ui.mask_target && !app.ui.vector_mask_target);
        app.ui.layers_panel_options.use_default_masks_on_fill_layers = true;
        persist_options(&mut app);
        app.run("layer.newFillLayer.gradient", json!({})).unwrap();
        let layer = app.session.active().and_then(|st| st.active_layer.and_then(|id| st.doc.layer(id))).unwrap();
        assert!(layer.mask.is_some());
        assert!(app.ui.mask_target);
    }

    /// Every option, including the mask badges, goes to the preferences and comes back from them.
    #[test]
    fn prefs_options_save_and_restore_all_switches() {
        let mut session = photocraft_engine::Session::new();
        session.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        let mut app = PhotocraftApp::new(session, crate::Services::default());
        let saved = LayersPanelOptions {
            thumbnail_size: LayerThumbnailSize::Large,
            thumbnail_contents: LayerThumbnailContents::EntireDocument,
            show_filters: false,
            use_default_masks_on_fill_layers: false,
            expand_new_effects: false,
            add_copy_to_copied_layers_and_groups: false,
            preview_on_canvas_when_reordering_layers: false,
            show_layer_mask_badges: true,
        };
        app.ui.layers_panel_options = saved.clone();
        persist_options(&mut app);
        assert_eq!(app.session.prefs().dialogs[OPTIONS_PREF]["showLayerMaskBadges"], true);
        app.ui.layers_panel_options = LayersPanelOptions::default();
        restore_options(&mut app);
        assert_eq!(app.ui.layers_panel_options, saved);
    }

    /// The flyout's Flatten Image runs the real `layer.flattenImage` command, and Undo brings the layers back.
    #[test]
    fn flyout_flatten_routes_to_real_engine_and_undo_restores_layers() {
        let mut session = photocraft_engine::Session::new();
        session.execute("file.new", json!({"width": 18, "height": 12})).unwrap();
        session.execute("layer.new.layer", json!({"name": "Red pixels"})).unwrap();
        session.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        let mut app = PhotocraftApp::new(session, crate::Services::default());
        let count = app.session.active().unwrap().doc.layers.len();
        assert!(count >= 2);
        // The flyout uses menus::invoke with this exact command id.
        crate::menus::invoke(&mut app, &egui::Context::default(), "layer.flattenImage", json!({})).unwrap();
        let flattened = app.session.active().unwrap();
        assert_eq!(flattened.doc.layers.len(), 1);
        assert_eq!(flattened.doc.layers[0].name, "Background");
        assert!(flattened.doc.layers[0].surface().is_some());
        app.run("edit.undo", json!({})).unwrap();
        assert_eq!(app.session.active().unwrap().doc.layers.len(), count);
    }

    /// Thumbnail and row sizes grow from None to Large in both themes, with the rows at least as tall as the thumbnails.
    #[test]
    fn geometry_matches_four_real_panel_sizes() {
        for pro in [false, true] {
            let mut previous = -1.0;
            for size in [LayerThumbnailSize::None, LayerThumbnailSize::Small, LayerThumbnailSize::Medium, LayerThumbnailSize::Large] {
                let (thumb, row) = thumbnail_geometry(size, pro);
                assert!(thumb > previous && row >= thumb, "{size:?}: {thumb}/{row}");
                previous = thumb;
            }
        }
        assert_eq!(thumbnail_geometry(LayerThumbnailSize::None, true), (0.0, 23.0));
        assert_eq!(thumbnail_geometry(LayerThumbnailSize::Large, true), (48.0, 56.0));
    }

    /// Preferences saved before an option existed still load, with the default for the missing ones.
    #[test]
    fn panel_options_missing_fields_deserialize_with_defaults() {
        let options: LayersPanelOptions = serde_json::from_value(json!({})).unwrap();
        assert_eq!(options, LayersPanelOptions::default());
        let saved = serde_json::to_value(&options).unwrap();
        assert_eq!(saved["thumbnailSize"], "medium");
    }
}
