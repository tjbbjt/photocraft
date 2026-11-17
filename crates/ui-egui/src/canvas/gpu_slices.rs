//! A full GPU refresh of a very large canvas, a slice per frame. Compositing a 1.1 GP storyboard
//! in one frame took 1.0–1.9 s (on opening it, on switching back to its tab, after moving layers
//! or undoing), and the desktop's compositor closes a window that stops answering for about
//! 1.5–2 s: the app was closed while it reopened the file after a crash. From [`SLICE_FROM_PX`] up,
//! the canvas is composited in bands of about [`BAND_PX`], the bands the view shows first, until a
//! frame has spent [`FRAME_MS`]; the rest follow on the next frame while the window keeps
//! answering. The budget is large on purpose: a frame's bands cost about 100 ms before their pixels,
//! and small bands at 120 ms a frame took 4–6 s to fill the storyboard the compositor drew in one
//! 1.0–1.6 s frame. Now a refresh the compositor finished in one frame under the budget still does,
//! and a longer one takes two frames of about a second, close to its old speed.
//! An edit whose damage is that large (undoing a move of a board, say) is drawn the same way.
//!
//! Zoomed out to 50% or less, a worker first composites a reduced copy of what the view shows
//! (`background_redraw::preview_document`, as #5561 does for the CPU redraw) and the first frames
//! wait for it, up to [`PREVIEW_WAIT_MS`]: it goes into the mip level the view samples, under the
//! bands still to come, and shows for a frame before they start, so a whole picture is there in
//! about half a second while the bands replace it at full resolution.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};

use photocraft_doc::DocId;
use photocraft_engine::display_color::CanvasDisplay;
use photocraft_geom::Rect;

use super::{Document, PhotocraftApp};
use crate::gpu_canvas::GpuCanvas;

/// Canvases of this many pixels or more refresh in slices (about 134 MP).
pub(crate) const SLICE_FROM_PX: u64 = 1 << 27;
/// About this many pixels a band (about 134 MP, some 150 ms on the storyboard).
const BAND_PX: u64 = 1 << 27;
/// Bands are composited until a frame has spent this long on them (at least one band a frame): with
/// the band that crosses it, a frame stays near a second, short of the 1.5–2 s after which the
/// desktop closes a window that isn't answering.
const FRAME_MS: f64 = 800.0;
/// How long the first frames wait for the reduced first picture before drawing bands anyway.
const PREVIEW_WAIT_MS: f64 = 700.0;

/// A reduced first picture being composited off the UI thread for a refresh in slices.
pub(crate) struct SlicePreview {
    /// The document it shows: an edit since then makes it stale.
    doc: Arc<Document>,
    /// The mip level it goes to (`1 << level` document pixels a texel).
    level: u32,
    /// Its rect in that level and its texels, once composited.
    rx: Receiver<(Rect, Vec<u8>)>,
    started_ms: f64,
}

/// The bands of `bounds`, full rows of about [`BAND_PX`] each: those meeting `visible` first, the
/// nearest to its middle first, then the rest by their distance from it.
pub(crate) fn bands(bounds: Rect, visible: Rect) -> VecDeque<Rect> {
    if bounds.is_empty() {
        return VecDeque::new();
    }
    let rows = (BAND_PX / u64::from(bounds.width())).clamp(1, u64::from(bounds.height())) as i32;
    let mut out = Vec::new();
    let mut y = bounds.y0;
    while y < bounds.y1 {
        let y1 = y.saturating_add(rows).min(bounds.y1);
        out.push(Rect::new(bounds.x0, y, bounds.x1, y1));
        y = y1;
    }
    let middle = i64::from(visible.y0) + i64::from(visible.y1);
    out.sort_by_key(|b| (b.intersect(&visible).is_empty(), (i64::from(b.y0) + i64::from(b.y1) - middle).abs()));
    out.into()
}

/// Starts refreshing all of `doc` in slices, in place of one full refresh. The bands start on the
/// next frame, so this one shows the window as it now is (a switched tab, the zoom it shows)
/// without waiting for them; zoomed out, a reduced first picture is on its way and the first
/// frames wait for it. `None` (nothing started) when the canvas is small enough to refresh at once,
/// or the wgpu compositor doesn't draw it (still being built, say: the CPU compositor then redraws
/// in the background), or its texture can't be made: the ordinary refresh decides. A refresh in
/// slices still running for document `id` starts over.
pub(crate) fn start(
    app: &mut PhotocraftApp,
    ctx: &egui::Context,
    gpu: &GpuCanvas,
    (id, doc): (DocId, &Arc<Document>),
    display: Option<&Arc<CanvasDisplay>>,
    visible: Rect,
    scale: f32,
) -> Option<()> {
    app.gpu_slices.remove(&id);
    app.gpu_slice_previews.remove(&id);
    if !sliced(gpu, (id, doc), doc.bounds()) {
        return None;
    }
    let in_view = visible.intersect(&doc.bounds());
    app.gpu_slices.insert(id, bands(doc.bounds(), in_view));
    if crate::background_redraw::enabled(app)
        && let Some(format) = gpu.prepare_texture(id.0, doc)
        && let Some(preview) = spawn_preview(ctx, doc, display, in_view, scale, format)
    {
        app.gpu_slice_previews.insert(id, preview);
    }
    ctx.request_repaint();
    Some(())
}

/// Composites the reduced first picture of `in_view` on a worker, at the mip level a view at
/// `scale` samples: `None` when the view isn't zoomed out to 50% or less, or shows nothing.
fn spawn_preview(
    ctx: &egui::Context,
    doc: &Arc<Document>,
    display: Option<&Arc<CanvasDisplay>>,
    in_view: Rect,
    scale: f32,
    format: eframe::wgpu::TextureFormat,
) -> Option<SlicePreview> {
    let level = crate::background_redraw::preview_level(scale)?;
    if in_view.is_empty() {
        return None;
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let (shown, display, ctx) = (doc.clone(), display.cloned(), ctx.clone());
    std::thread::Builder::new()
        .name("canvas-first-picture".into())
        .spawn(move || {
            // A panic only loses the picture: the bands draw the canvas regardless.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let k = 1u32 << level;
                let small = crate::background_redraw::preview_document(&shown, k);
                let rect = crate::background_redraw::reduce(in_view, k as i32).intersect(&small.bounds());
                let texels = crate::gpu_canvas::display_texels(format, display.as_deref(), &photocraft_compose::render(&small, rect));
                let _ = tx.send((rect, texels));
            }));
            ctx.request_repaint();
        })
        .ok()?;
    Some(SlicePreview { doc: doc.clone(), level, rx, started_ms: crate::gpu_canvas::now_ms() })
}

/// Starts refreshing an edit's `damage` in slices, in place of one partial refresh, when it covers
/// [`SLICE_FROM_PX`] or more: undoing a move of a storyboard's board redrew 550–690 MP in one frame
/// of 1.0–1.5 s, three undos in a row, and the desktop closed the window. Its bands go first, the
/// view's first, then those of a refresh in slices already running that it doesn't cover. Like
/// [`start`], from the next frame; `None` when the ordinary refresh draws it.
pub(crate) fn start_damage(
    app: &mut PhotocraftApp,
    ctx: &egui::Context,
    gpu: &GpuCanvas,
    (id, doc): (DocId, &Document),
    damage: Rect,
    visible: Rect,
) -> Option<()> {
    let damage = damage.intersect(&doc.bounds());
    if !sliced(gpu, (id, doc), damage) {
        return None;
    }
    let old = app.gpu_slices.remove(&id);
    app.gpu_slices.insert(id, queue_damage(old, damage, visible));
    ctx.request_repaint();
    Some(())
}

/// Whether a refresh of `area` of `doc` goes in slices: [`SLICE_FROM_PX`] or more, and drawn by
/// the wgpu compositor into a texture it could make.
fn sliced(gpu: &GpuCanvas, (id, doc): (DocId, &Document), area: Rect) -> bool {
    let px = u64::from(area.width()) * u64::from(area.height());
    px >= SLICE_FROM_PX && doc.id == id && !gpu.compositor_building() && gpu.supports(doc) && gpu.prepare_texture(id.0, doc).is_some()
}

/// The bands to draw after an edit's `damage`: its own, those meeting `visible` first, then the
/// bands of the running refresh (`old`) not wholly inside it.
pub(crate) fn queue_damage(old: Option<VecDeque<Rect>>, damage: Rect, visible: Rect) -> VecDeque<Rect> {
    let mut queue = bands(damage, visible.intersect(&damage));
    queue.extend(old.into_iter().flatten().filter(|b| !damage.contains_rect(b)));
    queue
}

/// Where a reduced picture covering `rect` of mip `level` goes: under the bands still to draw, in
/// that level's texels. Every band still to draw (a fresh refresh) takes the whole picture; once
/// some have landed, only the texels wholly inside a pending band, so the picture doesn't cover
/// the finished full-resolution mip levels next to it.
pub(crate) fn preview_pieces(bands: &VecDeque<Rect>, all: Rect, level: u32, rect: Rect) -> Vec<Rect> {
    let k = 1i32 << level;
    let covered: u64 = bands.iter().map(|b| u64::from(b.width()) * u64::from(b.height())).sum();
    if covered >= u64::from(all.width()) * u64::from(all.height()) {
        return vec![rect];
    }
    let inner = |b: &Rect| {
        Rect::new(
            b.x0.div_euclid(k) + i32::from(b.x0.rem_euclid(k) != 0),
            b.y0.div_euclid(k) + i32::from(b.y0.rem_euclid(k) != 0),
            b.x1.div_euclid(k),
            b.y1.div_euclid(k),
        )
    };
    bands.iter().map(|b| inner(b).intersect(&rect)).filter(|r| !r.is_empty()).collect()
}

/// The reduced first picture of document `id`'s refresh in slices: put in once it has arrived,
/// unless an edit since made it stale. True while this frame draws no bands: the first frames
/// wait for the picture (up to [`PREVIEW_WAIT_MS`]), and the frame it lands in shows it.
fn land_preview(app: &mut PhotocraftApp, ctx: &egui::Context, gpu: &GpuCanvas, (id, doc): (DocId, &Document)) -> bool {
    let Some(p) = app.gpu_slice_previews.get(&id) else { return false };
    let (level, started, current) = (p.level, p.started_ms, std::ptr::eq(Arc::as_ptr(&p.doc), doc));
    match p.rx.try_recv() {
        Ok((rect, texels)) => {
            app.gpu_slice_previews.remove(&id);
            let px = rect.width() as usize * rect.height() as usize;
            if !current || px == 0 || !texels.len().is_multiple_of(px) {
                return false;
            }
            let (bpp, stride) = (texels.len() / px, rect.width());
            let bands = app.gpu_slices.get(&id).cloned().unwrap_or_default();
            for at in preview_pieces(&bands, doc.bounds(), level, rect) {
                let offset = ((at.y0 - rect.y0) as usize * stride as usize + (at.x0 - rect.x0) as usize) * bpp;
                if let Some(t) = texels.get(offset..) {
                    gpu.upload_level(id.0, level, [at.x0 as u32, at.y0 as u32], [at.width(), at.height()], stride, t);
                }
            }
            log::info!(
                "canvas: the reduced first picture landed after {:.0} ms ({:.1} MP at 1/{})",
                crate::gpu_canvas::now_ms() - started,
                px as f64 / 1_048_576.0,
                1u32 << level
            );
            ctx.request_repaint();
            true
        }
        // An edit made it stale: the bands draw the edited document now.
        Err(TryRecvError::Empty) if !current => {
            app.gpu_slice_previews.remove(&id);
            false
        }
        Err(TryRecvError::Empty) if crate::gpu_canvas::now_ms() - started < PREVIEW_WAIT_MS => {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
            true
        }
        Err(_) => false,
    }
}

/// Whether document `id` still has bands to draw.
pub(crate) fn pending(app: &PhotocraftApp, id: DocId) -> bool {
    app.gpu_slices.get(&id).is_some_and(|q| !q.is_empty())
}

/// Draws the next bands of document `id`'s refresh in slices, as `doc` is now, for about
/// [`FRAME_MS`] and asks for another frame while some are left; none while the reduced first
/// picture is awaited or lands. False when the GPU canvas can't draw them (lost, or over its
/// budget): the refresh in slices ends and this frame goes to the CPU canvas.
pub(crate) fn step(
    app: &mut PhotocraftApp,
    ctx: &egui::Context,
    gpu: &GpuCanvas,
    (id, doc): (DocId, &Document),
    display: Option<&photocraft_engine::display_color::CanvasDisplay>,
) -> bool {
    if land_preview(app, ctx, gpu, (id, doc)) {
        return true;
    }
    let start = crate::gpu_canvas::now_ms();
    while let Some(band) = app.gpu_slices.get_mut(&id).and_then(VecDeque::pop_front) {
        let r = gpu.refresh(id.0, doc, Some(band), display);
        if r.kind == "lost" || r.kind == crate::gpu_canvas::REFRESH_OVER_BUDGET {
            app.gpu_slices.remove(&id);
            app.gpu_slice_previews.remove(&id);
            if let Some(reason) = r.fallback {
                crate::render_path::note(app, id.0, crate::render_path::Path::Cpu(reason));
            }
            return false;
        }
        app.perf.record_refresh(&r);
        if crate::gpu_canvas::now_ms() - start >= FRAME_MS {
            break;
        }
    }
    if pending(app, id) {
        ctx.request_repaint();
    } else {
        app.gpu_slices.remove(&id);
        app.gpu_slice_previews.remove(&id);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_cover_the_canvas_once_the_visible_ones_first() {
        let bounds = Rect::new(0, 0, 40_000, 30_000);
        let visible = Rect::new(10_000, 20_000, 12_000, 21_000);
        let bands = bands(bounds, visible);
        let rows: u64 = bands.iter().map(|b| u64::from(b.height())).sum();
        assert_eq!(rows, 30_000, "every row once");
        assert!(bands.iter().all(|b| b.x0 == 0 && b.x1 == 40_000 && u64::from(b.width()) * u64::from(b.height()) <= BAND_PX));
        assert!(!bands[0].intersect(&visible).is_empty(), "the view's band first: {:?}", bands[0]);
        let first_hidden = bands.iter().position(|b| b.intersect(&visible).is_empty()).unwrap();
        assert!(bands.iter().skip(first_hidden).all(|b| b.intersect(&visible).is_empty()), "then the rest");
    }

    #[test]
    fn a_narrow_tall_or_empty_canvas_still_bands() {
        assert!(bands(Rect::new(0, 0, 0, 0), Rect::new(0, 0, 0, 0)).is_empty());
        let one = bands(Rect::new(0, 0, 100, 100), Rect::new(0, 0, 100, 100));
        assert_eq!(Vec::from(one), [Rect::new(0, 0, 100, 100)]);
        // Wider than a band: one row at a time.
        let wide = bands(Rect::new(0, 0, 200_000_000, 3), Rect::new(0, 0, 10, 1));
        assert_eq!(wide.len(), 3);
    }

    #[test]
    fn an_edit_s_bands_go_before_the_rest_of_a_running_refresh() {
        let damage = Rect::new(0, 0, 20_000, 20_000);
        let old: VecDeque<Rect> = [Rect::new(0, 0, 40_000, 3_000), Rect::new(0, 30_000, 40_000, 33_000), Rect::new(0, 0, 10_000, 10_000)].into();
        let q = queue_damage(Some(old), damage, Rect::new(0, 15_000, 1_000, 16_000));
        let own = q.iter().take_while(|b| damage.contains_rect(b)).count();
        let rows: u64 = q.iter().take(own).map(|b| u64::from(b.height())).sum();
        assert_eq!(rows, 20_000, "the edit's bands first, every row once");
        assert!(q[0].y0 <= 15_000 && q[0].y1 > 15_000, "the view's band first: {:?}", q[0]);
        // Of the running refresh: the bands the edit doesn't cover, still in order.
        assert_eq!(Vec::from(q).split_off(own), [Rect::new(0, 0, 40_000, 3_000), Rect::new(0, 30_000, 40_000, 33_000)]);
        assert!(queue_damage(None, Rect::new(0, 0, 0, 0), Rect::new(0, 0, 0, 0)).is_empty());
    }

    #[test]
    fn a_first_picture_goes_only_under_bands_still_to_draw() {
        let all = Rect::new(0, 0, 1000, 1000);
        let rect = Rect::new(0, 0, 250, 250);
        let fresh: VecDeque<Rect> = [Rect::new(0, 0, 1000, 500), Rect::new(0, 500, 1000, 1000)].into();
        assert_eq!(preview_pieces(&fresh, all, 2, rect), [rect], "a fresh refresh: the whole picture");
        // The band at rows 0..502 landed: only level-2 texels wholly inside the rest (rows 126..),
        // not the one straddling the edge, which the landed band's mips drew.
        let rest: VecDeque<Rect> = [Rect::new(0, 502, 1000, 1000)].into();
        assert_eq!(preview_pieces(&rest, all, 2, rect), [Rect::new(0, 126, 250, 250)]);
        // The view's part of the picture only.
        assert_eq!(preview_pieces(&rest, all, 2, Rect::new(10, 10, 50, 50)), Vec::<Rect>::new());
        assert!(preview_pieces(&VecDeque::new(), all, 2, rect).is_empty(), "nothing left to draw");
    }
}
