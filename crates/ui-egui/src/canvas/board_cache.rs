//! Storyboard cache, step 1: what the canvas shows over a layer (a storyboard's artboard, most
//! usefully) kept aside when the layer is hidden, and put back when it is shown again, instead of
//! compositing it once more.
//!
//! When a layer is hidden, what the display texture shows over its reach is copied aside (GPU to
//! GPU, no readback); when it is shown again with nothing that composites there changed since,
//! the copy goes back. On a storyboard file of ~450 passes, showing a board recomposited ~150 MP
//! (about a second); the copy is milliseconds.
//!
//! The copy is used only when it is exact: the document as it was when the copy was taken and
//! the document now must composite the same over that area ([`super::doc_diff`]), on the same
//! display transform, with no preview showing. Otherwise the layer redraws as before.
//! Off with `PHOTOCRAFT_NO_BOARD_CACHE` (to compare).

use std::collections::HashMap;
use std::sync::Arc;

use photocraft_doc::{DocId, Document, LayerContent, LayerId};
use photocraft_geom::Rect;

use crate::PhotocraftApp;
use crate::gpu_canvas::{CanvasSnapshot, GpuCanvas};

/// One kept picture: the board's rectangle as the canvas showed it, and the document it showed.
struct Shot {
    doc: Arc<Document>,
    display_key: u64,
    snap: CanvasSnapshot,
    /// For least-recently-used eviction.
    used: u64,
}

/// The kept pictures of every open document's hidden layers.
#[derive(Default)]
pub(crate) struct BoardCache {
    shots: HashMap<(DocId, LayerId), Shot>,
    clock: u64,
    /// What lay under the layers of the last snapshot drag where they were dropped.
    under: Option<Under>,
}

/// The canvas where a snapshot drag dropped its layers, as it showed with them hidden.
struct Under {
    doc: DocId,
    layers: Vec<LayerId>,
    hidden: Arc<Document>,
    display_key: u64,
    snap: CanvasSnapshot,
}

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PHOTOCRAFT_NO_BOARD_CACHE").is_none())
}

/// GPU memory the kept pictures may take together: a quarter of the compositor's layer budget
/// (about 1.5 GB on a 16 GB card, a few hundred MB on a small one).
fn budget(gpu: &GpuCanvas) -> u64 {
    gpu.memory_budget().map_or(512 << 20, |b| b / 4)
}

/// Smallest area (pixels) worth keeping: smaller layers redraw quickly anyway.
const MIN_PX: u64 = 4 << 20;

/// Where showing layer `l` of `doc` composites: an artboard's rectangle, else the layer's reach
/// on the canvas. `None` when that can't be bounded (an adjustment layer reaching everything).
fn reach(doc: &Document, l: &photocraft_doc::Layer) -> Option<Rect> {
    let canvas = doc.bounds();
    let r = match &l.content {
        LayerContent::Group(g) if g.artboard.is_some() => g.artboard.as_ref().map(|a| a.rect),
        _ => photocraft_compose::change_bounds(l, canvas),
    }?;
    let r = r.intersect(&canvas);
    (r.size().area() >= MIN_PX).then_some(r)
}

fn meets(a: &Rect, b: &Rect) -> bool {
    !a.intersect(b).is_empty()
}

/// Called with the area about to be redrawn on document `id`'s display texture (`damage`), when
/// the canvas last showed `shown` and now shows `doc`: keeps the pictures of boards being hidden,
/// puts back those of boards being shown, and returns what is still to redraw (empty when a
/// put-back picture covers it all).
pub(crate) fn apply(app: &mut PhotocraftApp, gpu: &GpuCanvas, id: DocId, doc: &Arc<Document>, display_key: u64, damage: Rect) -> Rect {
    if !enabled() || damage.is_empty() {
        return damage;
    }
    retain(app);
    let Some((shown, shown_key)) = app.canvas_states.get(&id).and_then(|s| s.shown.clone()) else { return damage };
    if shown_key != display_key || Arc::ptr_eq(&shown, doc) {
        return damage;
    }
    let pending = super::sweep::pending(app, id);
    let before: HashMap<LayerId, &photocraft_doc::Layer> = shown.walk().into_iter().map(|(_, _, l)| (l.id, l)).collect();
    let mut restored: Vec<Rect> = Vec::new();
    for (_, _, l) in doc.walk() {
        let Some(was) = before.get(&l.id) else { continue };
        if was.visible == l.visible {
            continue;
        }
        if was.visible {
            // Being hidden: keep what the canvas shows over it, if that is all drawn.
            let Some(rect) = reach(&shown, was).filter(|r| meets(r, &damage)) else { continue };
            if pending.iter().any(|b| meets(b, &rect)) {
                continue;
            }
            keep(app, gpu, id, l.id, &l.name, rect, &shown, display_key);
        } else if let Some(rect) = reach(doc, l).filter(|r| meets(r, &damage))
            && put_back(app, gpu, id, l.id, &l.name, rect, doc, display_key)
        {
            restored.push(rect);
        }
    }
    if restored.iter().any(|r| r.contains_rect(&damage)) { Rect::EMPTY } else { damage }
}

#[allow(clippy::too_many_arguments)]
fn keep(app: &mut PhotocraftApp, gpu: &GpuCanvas, id: DocId, board: LayerId, name: &str, rect: Rect, shown: &Arc<Document>, display_key: u64) {
    let Some(snap) = gpu.snapshot_rect(id.0, rect) else { return };
    let limit = budget(gpu);
    if snap.bytes > limit {
        return;
    }
    let cache = &mut app.board_cache;
    cache.shots.remove(&(id, board));
    // Least recently used first, until the new one fits.
    while cache.shots.values().map(|s| s.snap.bytes).sum::<u64>() + snap.bytes > limit {
        let Some(oldest) = cache.shots.iter().min_by_key(|(_, s)| s.used).map(|(k, _)| *k) else { break };
        cache.shots.remove(&oldest);
    }
    cache.clock += 1;
    log::info!("storyboard cache: kept \"{name}\" ({}x{}, {} MB; {} kept)", rect.width(), rect.height(), snap.bytes >> 20, cache.shots.len() + 1);
    cache.shots.insert((id, board), Shot { doc: shown.clone(), display_key, snap, used: cache.clock });
}

#[allow(clippy::too_many_arguments)]
fn put_back(app: &mut PhotocraftApp, gpu: &GpuCanvas, id: DocId, board: LayerId, name: &str, rect: Rect, doc: &Document, display_key: u64) -> bool {
    let cache = &mut app.board_cache;
    let Some(shot) = cache.shots.get_mut(&(id, board)) else { return false };
    // Exact only if nothing that composites inside the rectangle changed since it was kept.
    let same = shot.display_key == display_key && shot.snap.rect == rect && still_exact(&shot.doc, doc, rect);
    if !same {
        log::info!("storyboard cache: \"{name}\" changed since it was kept; redrawing it");
        cache.shots.remove(&(id, board));
        return false;
    }
    if !gpu.restore_rect(id.0, &shot.snap) {
        cache.shots.remove(&(id, board));
        return false;
    }
    cache.clock += 1;
    shot.used = cache.clock;
    log::info!("storyboard cache: put back \"{name}\" instead of redrawing {:.0} MP", rect.size().area() as f64 / 1_048_576.0);
    true
}

/// Whether a picture of `rect` taken when the canvas showed `kept` is what `now` shows there:
/// nothing that composites inside it changed since.
fn still_exact(kept: &Document, now: &Document, rect: Rect) -> bool {
    super::doc_diff::same_pixels(kept, now) || super::doc_diff::changed_area(kept, now).is_some_and(|r| r.is_empty() || !meets(&r, &rect))
}

/// Forget the pictures of closed documents.
fn retain(app: &mut PhotocraftApp) {
    let live: Vec<DocId> = app.session.documents().iter().map(|d| d.doc.id).collect();
    app.board_cache.shots.retain(|(doc, _), _| live.contains(doc));
    if app.board_cache.under.as_ref().is_some_and(|u| !live.contains(&u.doc)) {
        app.board_cache.under = None;
    }
}

/// At a snapshot drag's drop, before the move: the canvas shows `hidden` (the document with the
/// moving `layers` hidden); keep what it shows in `rect`, where they land, if that is all drawn.
pub(crate) fn keep_under(app: &mut PhotocraftApp, doc: DocId, layers: &[LayerId], hidden: &Arc<Document>, rect: Rect) {
    app.board_cache.under = None;
    if !enabled() {
        return;
    }
    let Some(gpu) = app.gpu.clone() else { return };
    let Some(display_key) = app.canvas_states.get(&doc).and_then(|s| s.shown.as_ref()).filter(|(d, _)| Arc::ptr_eq(d, hidden)).map(|(_, k)| *k) else { return };
    if super::sweep::pending(app, doc).iter().any(|b| meets(b, &rect)) {
        return;
    }
    let Some(snap) = gpu.snapshot_rect(doc.0, rect) else { return };
    if snap.bytes > budget(&gpu) {
        return;
    }
    log::info!("storyboard cache: kept what is under the dropped layer(s) ({:.0} MP, {} MB)", rect.size().area() as f64 / 1_048_576.0, snap.bytes >> 20);
    app.board_cache.under = Some(Under { doc, layers: layers.to_vec(), hidden: hidden.clone(), display_key, snap });
}

/// A snapshot drag hiding its layers: when they were last dropped here and nothing under them
/// changed since, put back what lies under them instead of redrawing the place they leave.
/// Returns what is still to redraw of `damage`.
pub(crate) fn drag_start(app: &mut PhotocraftApp, gpu: &GpuCanvas, id: DocId, doc: &Document, display_key: u64, damage: Rect) -> Rect {
    if !enabled() || damage.is_empty() {
        return damage;
    }
    let Some((drag_doc, layers, rect)) = crate::move_ui::hiding(app) else { return damage };
    let Some(u) = app.board_cache.under.take() else { return damage };
    let fits = u.doc == id && drag_doc == id && u.layers == layers && u.snap.rect == rect && u.display_key == display_key;
    if !fits || !still_exact(&u.hidden, doc, rect) || !gpu.restore_rect(id.0, &u.snap) {
        return damage;
    }
    log::info!("storyboard cache: put back what is under the dragged layer(s) instead of redrawing {:.0} MP", rect.size().area() as f64 / 1_048_576.0);
    if rect.contains_rect(&damage) { Rect::EMPTY } else { damage }
}
