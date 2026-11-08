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
}

impl Work {
    /// Plan `doc` once and time it: every block plans the whole document again.
    fn planned(doc: &Document) -> Self {
        let t0 = crate::gpu_canvas::now_ms();
        let (passes, fx_layers) = photocraft_gpu::plan(doc).map_or((0, 0), |p| (p.passes.len(), p.fx.len()));
        Work { plan_ms: crate::gpu_canvas::now_ms() - t0, passes, fx_layers, ..Work::default() }
    }

    fn add(&mut self, s: &photocraft_gpu::Stats) {
        self.chunks += s.chunks as u64;
        self.batched += s.layers_batched as u64;
        self.prefix_hits += s.prefix_hits as u64;
        self.fx_programs += s.fx_programs as u64;
        self.fx_px += s.fx_pixels;
        self.evicted += s.evicted as u64;
        self.flushes += s.flushes as u64;
    }

    /// "; plan 0.4 ms × 378 blocks, 12 passes (1 with effects), 4536 chunks, …"
    fn describe(&self, blocks: u32) -> String {
        format!(
            "; plan {:.1} ms x {blocks} blocks, {} passes ({} effect layers), {} chunks, {} layers batched, {} prefix reuses, {} effect maps ({:.0} MP), {} pages evicted, {} flushes",
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
        )
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

/// Whether a sweep of `doc` still has blocks to draw.
pub(crate) fn running(app: &PhotocraftApp, doc: DocId) -> bool {
    app.canvas_states.get(&doc).is_some_and(|s| !s.pending.is_empty())
}

/// (Re)start a sweep of all of `doc`'s canvas, the blocks nearest `visible` first.
pub(crate) fn start(app: &mut PhotocraftApp, doc: &Document, visible: Rect) {
    let state = app.canvas_states.entry(doc.id).or_default();
    state.pending = blocks(doc.bounds(), visible, block_size(state.gpu_ms_per_mp));
}

/// Add `region` of `doc`'s canvas to its sweep, drawn before what was already pending.
pub(crate) fn add(app: &mut PhotocraftApp, doc: &Document, region: Rect, visible: Rect) {
    let region = region.intersect(&doc.bounds());
    if region.is_empty() {
        return;
    }
    let state = app.canvas_states.entry(doc.id).or_default();
    // Popped from the end: the new blocks first. A block pending twice is drawn twice, which
    // is only wasted time, never a wrong pixel.
    state.pending.extend(blocks(region, visible, block_size(state.gpu_ms_per_mp)));
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
    loop {
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
            gpu_estimate += s.account(block, &r);
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
    /// Count a drawn block in the sweep's totals and learn the GPU's rate from `r`; returns the
    /// GPU time the block is estimated to take.
    fn account(&mut self, block: Rect, r: &crate::gpu_canvas::Refresh) -> f64 {
        // A GPU measurement comes back a frame or two late, for an earlier refresh: it carries
        // its own pixel count, so it still gives a rate.
        if let Some(t) = r.gpu.filter(|t| t.px > 0) {
            let rate = t.ms / (t.px as f64 / 1_048_576.0);
            self.gpu_ms_per_mp = Some(self.gpu_ms_per_mp.map_or(rate, |old| old * 0.7 + rate * 0.3));
        }
        if let Some(totals) = self.totals.as_mut() {
            totals.blocks += 1;
            totals.px += block.size().area();
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
        self.gpu_ms_per_mp.unwrap_or(0.0) * (block.size().area() as f64 / 1_048_576.0)
    }
}

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
fn blocks(bounds: Rect, visible: Rect, (block_w, block_h): (i32, i32)) -> Vec<Rect> {
    let mut out = Vec::new();
    let mut y = bounds.y0;
    while y < bounds.y1 {
        let mut x = bounds.x0;
        while x < bounds.x1 {
            out.push(Rect::new(x, y, (x + block_w).min(bounds.x1), (y + block_h).min(bounds.y1)));
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
        let b = blocks(bounds, visible, BLOCK_SIZES[0]);
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
}
