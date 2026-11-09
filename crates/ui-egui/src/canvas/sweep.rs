//! Gradual redraws of a large GPU canvas. Where the canvas would recomposite a large area in one
//! frame (the whole document for a change nothing bounded, the first draw of a document, or a
//! change that bounded but large, like committing a transform of a big layer), it redraws it in
//! blocks instead, a frame's budget at a time, the blocks in view first. On a gigapixel canvas the one-frame redraw took over
//! a second on the UI thread, long enough for a Wayland compositor to drop the window; in blocks
//! the window keeps answering, and until a block is redrawn it shows what it showed before.
//!
//! A sweep is also the safety net behind [`super::doc_diff`]: after a change whose area was worked
//! out by comparing documents, a sweep redraws everything, so if that comparison ever missed a
//! pixel, it shows for a moment at most, never for good.
//!
//! Only on the GPU path, only with background jobs (the desktop app), and only for canvases of at
//! least [`SWEEP_MIN_PX`]; elsewhere the canvas redraws in one go as before. A block that doesn't
//! come back as a plain GPU refresh (the CPU compositor stepping in, the texture replaced) ends
//! the sweep, and the document redraws in one go until the next full refresh succeeds.

use std::collections::HashMap;
use std::sync::Arc;

use photocraft_doc::{DocId, Document};
use photocraft_engine::display_color::CanvasDisplay;
use photocraft_geom::Rect;

use crate::PhotocraftApp;
use crate::gpu_canvas::GpuCanvas;

/// Canvases this large (32 MP, about 8192 × 4096) redraw in blocks; smaller ones are quick in
/// one go. (A 46 MP image's first draw took 0.8 s in one frame at 64 MP.)
pub(crate) const SWEEP_MIN_PX: u64 = 32 << 20;

/// On such a canvas, a redraw of at least this area goes in blocks too (16 MP, a few tenths of
/// a second in one frame on the gigapixel canvas this was measured on).
pub(crate) const SWEEP_REGION_PX: u64 = 16 << 20;

/// Block sizes in document pixels, whole compositor chunks (1024 px) and all under
/// [`crate::gpu_canvas::DEFER_PX`], so the compositor treats each as a small refresh (a larger
/// one is checked against the layer memory budget and could fall back to the CPU). The larger
/// size is used once blocks are known to be cheap: fewer refreshes, less per-block overhead.
const BLOCK_SIZES: [(i32, i32); 2] = [(2048, 1024), (3072, 1024)];

/// A block should take about this long on the GPU to be worth the larger size (ms).
const TARGET_BLOCK_MS: f64 = 6.0;

/// Time a frame spends on blocks (at least one block is drawn per frame): the UI thread's time
/// encoding them, and the GPU's estimated time running them, whichever is more. Counting the
/// GPU keeps a frame from queueing more work than the GPU finishes before the next one, which
/// would make every later frame wait for it.
const FRAME_BUDGET_MS: f64 = 24.0;

/// A sweep's running totals, logged when it ends.
#[derive(Clone, Copy, Debug, Default)]
struct Totals {
    started_ms: f64,
    blocks: u32,
    frames: u32,
    px: u64,
    cpu_ms: f64,
    gpu_ms: f64,
    gpu_px: u64,
    /// Layer tiles the blocks uploaded to the GPU (pages it didn't hold).
    uploads: u64,
    /// What the compositor did over the sweep (summed) and planning the document once (timed when
    /// the sweep starts): where a slow sweep's UI-thread time goes.
    work: Work,
}

/// Compositor work summed over a sweep's blocks, for its log line.
#[derive(Clone, Copy, Debug, Default)]
struct Work {
    plan_ms: f64,
    passes: usize,
    fx_layers: usize,
    chunks: u64,
    batched: u64,
    prefix_hits: u64,
    fx_programs: u64,
    fx_px: u64,
    evicted: u64,
    flushes: u64,
    /// Pass bindings skipped as outside their blocks, of all considered.
    culled: u64,
    bindings: u64,
    /// The scratch disk's counters when the sweep started.
    scratch: crate::frame_stages::ScratchMark,
}

impl Work {
    /// Plan `doc` once and time it: every block plans the whole document again.
    fn planned(doc: &Document) -> Self {
        let t0 = crate::gpu_canvas::now_ms();
        let (passes, fx_layers) = photocraft_gpu::plan(doc).map_or((0, 0), |p| (p.passes.len(), p.fx.len()));
        Work { plan_ms: crate::gpu_canvas::now_ms() - t0, passes, fx_layers, scratch: crate::frame_stages::ScratchMark::now(), ..Work::default() }
    }

    fn add(&mut self, s: &photocraft_gpu::Stats) {
        self.chunks += s.chunks as u64;
        self.batched += s.layers_batched as u64;
        self.prefix_hits += s.prefix_hits as u64;
        self.fx_programs += s.fx_programs as u64;
        self.fx_px += s.fx_pixels;
        self.evicted += s.evicted as u64;
        self.flushes += s.flushes as u64;
        self.culled += s.passes_culled as u64;
        self.bindings += s.pass_bindings as u64;
    }

    /// "; plan 0.4 ms × 378 blocks, 12 passes (1 with effects), 4536 chunks, …"
    fn describe(&self, blocks: u32) -> String {
        format!(
            "; plan {:.1} ms x {blocks} blocks, {} passes ({} effect layers), {} chunks, {} layers batched, {} prefix reuses, {} effect maps ({:.0} MP), {} pages evicted, {} flushes, {}% of pass bindings skipped outside their blocks",
            self.plan_ms,
            self.passes,
            self.fx_layers,
            self.chunks,
            self.batched,
            self.prefix_hits,
            self.fx_programs,
            self.fx_px as f64 / 1_048_576.0,
            self.evicted,
            self.flushes,
            (self.culled * 100).checked_div(self.bindings).unwrap_or(0),
        ) + &self.scratch.note_until(&crate::frame_stages::ScratchMark::now())
    }
}

/// Per document: the document its GPU canvas last showed (what [`super::doc_diff`] compares
/// against) and any sweep in progress.
#[derive(Default)]
pub(crate) struct CanvasState {
    pub(crate) shown: Option<(Arc<Document>, u64)>,
    pending: Vec<Rect>,
    /// A block came back other than as a GPU refresh: redraw in one go until a full one lands.
    failed: bool,
    /// GPU milliseconds per megapixel of a block, from the timestamps of earlier blocks (a
    /// moving average); `None` until the GPU has reported one.
    gpu_ms_per_mp: Option<f64>,
    totals: Option<Totals>,
}

pub(crate) type CanvasStates = HashMap<DocId, CanvasState>;

/// The outcome of drawing a frame's blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// Blocks remain.
    Continuing,
    /// The canvas is all redrawn.
    Done,
    /// A block didn't redraw on the GPU: the sweep is dropped; redraw in one go.
    Failed,
}

/// Whether `doc`'s canvas redraws everything in blocks rather than in one frame.
pub(crate) fn eligible(app: &PhotocraftApp, doc: &Document) -> bool {
    app.background_jobs && doc.bounds().size().area() >= SWEEP_MIN_PX && !app.canvas_states.get(&doc.id).is_some_and(|s| s.failed)
}

/// The blocks of `doc`'s canvas a sweep hasn't drawn yet (none without a sweep).
pub(crate) fn pending(app: &PhotocraftApp, doc: DocId) -> Vec<Rect> {
    app.canvas_states.get(&doc).map(|s| s.pending.clone()).unwrap_or_default()
}

/// Whether a sweep of `doc` still has blocks to draw.
pub(crate) fn running(app: &PhotocraftApp, doc: DocId) -> bool {
    app.canvas_states.get(&doc).is_some_and(|s| !s.pending.is_empty())
}

/// (Re)start a sweep of all of `doc`'s canvas, the blocks nearest `visible` first.
pub(crate) fn start(app: &mut PhotocraftApp, doc: &Document, visible: Rect) {
    let state = app.canvas_states.entry(doc.id).or_default();
    state.pending = blocks(doc.bounds(), doc.bounds(), visible, block_size(state.gpu_ms_per_mp));
}

/// The safety net behind a change bounded by comparing documents ([`super::doc_diff`]): redraw
/// everything in blocks. A sweep already running is that net for the blocks it hasn't drawn, and
/// restarting it would throw away the ones it has (each change, such as a transform's commit and
/// the selection update after it, restarted a whole-canvas redraw), so it carries on instead.
///
/// Off unless `PHOTOCRAFT_SWEEP_SAFETY_NET` is set: on a storyboard of hundreds of layers the net
/// redrew the whole canvas (about 5 s) after every bounded change, such as a layer dropped from
/// another document, though the comparison had already redrawn what changed.
pub(crate) fn safety_net(app: &mut PhotocraftApp, doc: &Document, visible: Rect) {
    if std::env::var_os("PHOTOCRAFT_SWEEP_SAFETY_NET").is_none() {
        return;
    }
    if !running(app, doc.id) {
        start(app, doc, visible);
    }
}

/// Add `region` of `doc`'s canvas to its sweep, drawn before what was already pending.
pub(crate) fn add(app: &mut PhotocraftApp, doc: &Document, region: Rect, visible: Rect) {
    let region = region.intersect(&doc.bounds());
    if region.is_empty() {
        return;
    }
    let state = app.canvas_states.entry(doc.id).or_default();
    // Blocks sit on one grid over the canvas, so a block already pending is the same rectangle:
    // it moves to the front instead of being queued again. A Move drag of a large layer adds its
    // old and new place every frame; queued again each time, the sweep grew to several times
    // the canvas (6375 MP over a minute) and the blocks of earlier places waited behind the
    // newer ones. Popped from the end: the new blocks first.
    let new = blocks(region, doc.bounds(), visible, block_size(state.gpu_ms_per_mp));
    let queued: std::collections::HashSet<(i32, i32, i32, i32)> = new.iter().map(|b| (b.x0, b.y0, b.x1, b.y1)).collect();
    state.pending.retain(|b| !queued.contains(&(b.x0, b.y0, b.x1, b.y1)));
    state.pending.extend(new);
}

/// Forget a sweep that failed, after a full refresh landed.
pub(crate) fn full_refresh_landed(app: &mut PhotocraftApp, doc: DocId) {
    if let Some(s) = app.canvas_states.get_mut(&doc) {
        s.failed = false;
        s.pending.clear();
        s.totals = None;
    }
}

/// The larger block size once blocks have proved cheap on this GPU.
fn block_size(gpu_ms_per_mp: Option<f64>) -> (i32, i32) {
    let [small, large] = BLOCK_SIZES;
    let mp = |(w, h): (i32, i32)| f64::from(w) * f64::from(h) / 1_048_576.0;
    match gpu_ms_per_mp {
        Some(rate) if rate * mp(large) <= TARGET_BLOCK_MS => large,
        _ => small,
    }
}

/// Remember what the canvas of `doc` shows now (with display key `display_key`).
pub(crate) fn shown(app: &mut PhotocraftApp, doc: &Arc<Document>, display_key: u64) {
    app.canvas_states.entry(doc.id).or_default().shown = Some((doc.clone(), display_key));
}

/// The area where `doc` differs from what its canvas last showed, if that can be bounded.
pub(crate) fn changed_since_shown(app: &PhotocraftApp, doc: &Document, display_key: u64) -> Option<Rect> {
    let (prev, key) = app.canvas_states.get(&doc.id)?.shown.as_ref()?;
    if *key != display_key {
        return None;
    }
    super::doc_diff::changed_area(prev, doc)
}

/// Draw blocks of `doc`'s sweep until this frame's budget is spent, and ask for another frame
/// while blocks remain. Logs one line per sweep when it ends: its size, blocks, frames and time.
pub(crate) fn step(app: &mut PhotocraftApp, ctx: &egui::Context, gpu: &GpuCanvas, doc: &Document, display: Option<&CanvasDisplay>) -> Step {
    let started = crate::gpu_canvas::now_ms();
    let mut gpu_estimate = 0.0;
    if let Some(s) = app.canvas_states.get_mut(&doc.id) {
        let totals = s.totals.get_or_insert_with(|| Totals { started_ms: started, work: Work::planned(doc), ..Totals::default() });
        totals.frames += 1;
    }
    let mut batches = true;
    loop {
        // Once a block's cost is known, the blocks this frame's budget holds go in one render:
        // one plan of the document for all of them instead of one per block.
        let n = if batches { batch_len(app, doc.id, started, gpu_estimate) } else { 1 };
        if n > 1 {
            let batch: Vec<Rect> = app.canvas_states.get(&doc.id).map(|s| s.pending.iter().rev().take(n).copied().collect()).unwrap_or_default();
            if batch.len() > 1 {
                match gpu.refresh_blocks(doc.id.0, doc, &batch, display) {
                    Some(r) => {
                        if let Some(s) = app.canvas_states.get_mut(&doc.id) {
                            let keep = s.pending.len().saturating_sub(batch.len());
                            s.pending.truncate(keep);
                            gpu_estimate += s.account(r.px, batch.len() as u32, &r);
                        }
                        if !running(app, doc.id) {
                            return finish(app, gpu, doc.id, Step::Done);
                        }
                        let spent = (crate::gpu_canvas::now_ms() - started).max(gpu_estimate);
                        if spent >= FRAME_BUDGET_MS {
                            ctx.request_repaint();
                            return Step::Continuing;
                        }
                        continue;
                    }
                    // One at a time for the rest of this frame, each with its own fallback.
                    None => batches = false,
                }
            }
        }
        let Some(block) = app.canvas_states.get_mut(&doc.id).and_then(|s| s.pending.pop()) else { return finish(app, gpu, doc.id, Step::Done) };
        let r = gpu.refresh_with(doc.id.0, doc, Some(block), display, false);
        match (r.kind, &r.fallback) {
            ("gpu-rect" | "rect", None) => {}
            // The texture had to be made again: that refresh drew everything.
            ("gpu-full", None) => {
                if let Some(s) = app.canvas_states.get_mut(&doc.id) {
                    s.pending.clear();
                }
                return finish(app, gpu, doc.id, Step::Done);
            }
            _ => {
                log::info!("canvas: redrawing in blocks stopped ({}); redrawing in one go", r.kind);
                if let Some(s) = app.canvas_states.get_mut(&doc.id) {
                    s.pending.clear();
                    s.failed = true;
                }
                return finish(app, gpu, doc.id, Step::Failed);
            }
        }
        if let Some(s) = app.canvas_states.get_mut(&doc.id) {
            gpu_estimate += s.account(block.size().area(), 1, &r);
        }
        if !running(app, doc.id) {
            return finish(app, gpu, doc.id, Step::Done);
        }
        let spent = (crate::gpu_canvas::now_ms() - started).max(gpu_estimate);
        if spent >= FRAME_BUDGET_MS {
            ctx.request_repaint();
            return Step::Continuing;
        }
    }
}

impl CanvasState {
    /// Count `blocks` drawn blocks of `px` pixels in all in the sweep's totals and learn the GPU's
    /// rate from `r`; returns the GPU time they are estimated to take.
    fn account(&mut self, px: u64, blocks: u32, r: &crate::gpu_canvas::Refresh) -> f64 {
        // A GPU measurement comes back a frame or two late, for an earlier refresh: it carries
        // its own pixel count, so it still gives a rate.
        if let Some(t) = r.gpu.filter(|t| t.px > 0) {
            let rate = t.ms / (t.px as f64 / 1_048_576.0);
            self.gpu_ms_per_mp = Some(self.gpu_ms_per_mp.map_or(rate, |old| old * 0.7 + rate * 0.3));
        }
        if let Some(totals) = self.totals.as_mut() {
            totals.blocks += blocks;
            totals.px += px;
            totals.cpu_ms += r.composite_ms;
            totals.uploads += r.uploads;
            if let Some(stats) = &r.stats {
                totals.work.add(stats);
            }
            if let Some(t) = r.gpu {
                totals.gpu_ms += t.ms;
                totals.gpu_px += t.px;
            }
        }
        self.gpu_ms_per_mp.unwrap_or(0.0) * (px as f64 / 1_048_576.0)
    }
}

/// How many of `doc`'s pending blocks fit what is left of this frame's budget, going by what the
/// sweep's blocks have cost so far (encoding on this thread, and the GPU's rate): 1 until a block
/// has been drawn, at most [`MAX_BATCH`].
fn batch_len(app: &PhotocraftApp, doc: DocId, started: f64, gpu_estimate: f64) -> usize {
    let Some(s) = app.canvas_states.get(&doc) else { return 1 };
    let Some(t) = s.totals.as_ref().filter(|t| t.blocks > 0) else { return 1 };
    let Some(next) = s.pending.last() else { return 1 };
    let cpu = t.cpu_ms / f64::from(t.blocks);
    let gpu = s.gpu_ms_per_mp.unwrap_or(0.0) * (next.size().area() as f64 / 1_048_576.0);
    let left = FRAME_BUDGET_MS - (crate::gpu_canvas::now_ms() - started).max(gpu_estimate);
    let each = cpu.max(gpu).max(0.1);
    ((left / each).floor().max(1.0) as usize).min(MAX_BATCH)
}

/// Most blocks drawn in one render.
const MAX_BATCH: usize = 8;

/// End of a sweep (or of its last frame): log its totals once it is over.
fn finish(app: &mut PhotocraftApp, gpu: &GpuCanvas, doc: DocId, step: Step) -> Step {
    if step != Step::Continuing
        && let Some(t) = app.canvas_states.get_mut(&doc).and_then(|s| s.totals.take())
        && t.blocks > 0
    {
        let gpu_rate = if t.gpu_px > 0 { format!("{:.2} ms/MP", t.gpu_ms / (t.gpu_px as f64 / 1_048_576.0)) } else { "not measured".into() };
        let mb = |b: u64| b / (1 << 20);
        let resident = gpu.compositor_bytes().map_or(0, |(pages, _)| pages);
        let budget = gpu.memory_budget().map_or_else(|| "?".into(), |b| mb(b).to_string());
        log::info!(
            "canvas: redrew {:.0} MP in {} blocks over {} frames, {:.0} ms ({:.0} ms encoding on the UI thread; GPU {gpu_rate}; uploaded {} layer tiles; layer pages {} of {budget} MB{}){}",
            t.px as f64 / 1_048_576.0,
            t.blocks,
            t.frames,
            crate::gpu_canvas::now_ms() - t.started_ms,
            t.cpu_ms,
            t.uploads,
            mb(resident),
            t.work.describe(t.blocks),
            if step == Step::Failed { "; stopped" } else { "" },
        );
    }
    step
}

/// Blocks covering `bounds`, in drawing order from the end (they are popped): the blocks the view
/// shows, nearest its centre first, then the rest by distance.
/// On one grid over `canvas`, so the same block of two regions is the same rectangle.
fn blocks(region: Rect, canvas: Rect, visible: Rect, (block_w, block_h): (i32, i32)) -> Vec<Rect> {
    let region = region.intersect(&canvas);
    let mut out = Vec::new();
    if region.is_empty() || block_w <= 0 || block_h <= 0 {
        return out;
    }
    let snap = |v: i32, origin: i32, step: i32| origin + (v - origin).div_euclid(step) * step;
    let mut y = snap(region.y0, canvas.y0, block_h);
    while y < region.y1 {
        let mut x = snap(region.x0, canvas.x0, block_w);
        while x < region.x1 {
            out.push(Rect::new(x, y, (x + block_w).min(canvas.x1), (y + block_h).min(canvas.y1)));
            x += block_w;
        }
        y += block_h;
    }
    let (cx, cy) = (i64::from(visible.x0 / 2 + visible.x1 / 2), i64::from(visible.y0 / 2 + visible.y1 / 2));
    let key = |b: &Rect| {
        let (bx, by) = (i64::from(b.x0 / 2 + b.x1 / 2), i64::from(b.y0 / 2 + b.y1 / 2));
        (!b.intersect(&visible).is_empty(), -((bx - cx).pow(2) + (by - cy).pow(2)))
    };
    // Ascending, so the last (drawn first) is in view and nearest the centre.
    out.sort_by_key(key);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_cover_the_canvas_once_and_draw_the_view_first() {
        let bounds = Rect::new(0, 0, 10_000, 5_000);
        let visible = Rect::new(6_000, 1_000, 8_000, 3_000);
        let b = blocks(bounds, bounds, visible, BLOCK_SIZES[0]);
        let area: u64 = b.iter().map(|r| r.size().area()).sum();
        assert_eq!(area, bounds.size().area(), "every pixel once");
        let first = *b.last().unwrap();
        assert!(!first.intersect(&visible).is_empty(), "the view's blocks come first: {first:?}");
        let last = b[0];
        assert!(last.intersect(&visible).is_empty(), "far blocks come last: {last:?}");
    }

    #[test]
    fn blocks_grow_only_once_they_are_cheap_and_stay_small_refreshes() {
        assert_eq!(block_size(None), BLOCK_SIZES[0], "unmeasured: the small size");
        assert_eq!(block_size(Some(10.0)), BLOCK_SIZES[0], "slow: the small size");
        assert_eq!(block_size(Some(0.5)), BLOCK_SIZES[1], "cheap: the larger size");
        for (w, h) in BLOCK_SIZES {
            assert!((w as u64) * (h as u64) < crate::gpu_canvas::DEFER_PX, "{w}x{h} stays a small refresh");
        }
    }

    #[test]
    fn blocks_of_overlapping_regions_are_the_same_rectangles() {
        let canvas = Rect::new(0, 0, 10_000, 5_000);
        let size = BLOCK_SIZES[0];
        let a = blocks(Rect::new(100, 100, 3_000, 1_500), canvas, canvas, size);
        let b = blocks(Rect::new(200, 50, 3_100, 1_400), canvas, canvas, size);
        let set = |v: &[Rect]| v.iter().map(|r| (r.x0, r.y0, r.x1, r.y1)).collect::<std::collections::HashSet<_>>();
        assert_eq!(set(&a), set(&b), "both regions fall on the same grid blocks");
        for r in &a {
            assert!(r.x0 % size.0 == 0 && r.y0 % size.1 == 0, "on the grid: {r:?}");
        }
        let covered = a.iter().fold(Rect::EMPTY, |u, r| if u.is_empty() { *r } else { u.union(r) });
        assert!(covered.contains_rect(&Rect::new(100, 100, 3_000, 1_500)), "the region is covered: {covered:?}");
    }
}
