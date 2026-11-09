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
    /// Blocks were composited on the redraw thread ([`super::redraw_thread`]).
    threaded: bool,
    /// Time the sweep was under way while its document was shown (ms): a background tab's sweep
    /// waits until it is shown again, and its wall-clock time read as a 18 s redraw.
    active_ms: f64,
    last_frame_ms: f64,
}

impl Totals {
    /// Count a frame of the sweep at `now`; gaps longer than a few frames are time it waited.
    fn frame(&mut self, now: f64) {
        self.frames += 1;
        if self.last_frame_ms > 0.0 {
            self.active_ms += (now - self.last_frame_ms).clamp(0.0, MAX_FRAME_GAP_MS);
        }
        self.last_frame_ms = now;
    }
}

/// A gap between a sweep's frames longer than this (ms) counts as waiting, not redrawing.
const MAX_FRAME_GAP_MS: f64 = 250.0;

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
    /// Renders (one per block, or per batch of blocks) and their CPU stages, summed (ms).
    renders: u64,
    stage_ms: [f64; 5],
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
        self.renders += 1;
        for (sum, v) in self.stage_ms.iter_mut().zip([s.plan_ms, s.prepare_ms, s.setup_ms, s.cells_ms, s.tail_ms]) {
            *sum += v;
        }
    }

    /// "; per render (2.1 blocks): plan 3.9, cache check 2.0, setup 0.5, cells 4.1, clean-up 1.2, other 3.0 ms"
    fn stages(&self, blocks: u32, cpu_ms: f64) -> String {
        if self.renders == 0 {
            return String::new();
        }
        let n = self.renders as f64;
        let [plan, prepare, setup, cells, tail] = self.stage_ms.map(|v| v / n);
        let other = (cpu_ms / n - (plan + prepare + setup + cells + tail)).max(0.0);
        format!(
            "; per render ({:.1} blocks): plan {plan:.1}, cache check {prepare:.1}, setup {setup:.1}, cells {cells:.1}, clean-up {tail:.1}, other {other:.1} ms",
            f64::from(blocks) / n
        )
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
    /// Jobs on the redraw thread: their tickets and blocks.
    in_flight: Vec<(u64, Vec<Rect>)>,
    /// A job failed on the redraw thread: this sweep goes on on the UI thread.
    thread_failed: bool,
    /// No jobs before this time ([`crate::gpu_canvas::now_ms`]): the compositor wasn't ready.
    thread_wait_until: f64,
}

/// How long the redraw thread waits for a compositor that isn't ready (ms).
const THREAD_RETRY_MS: f64 = 100.0;

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
    // Blocks out on the redraw thread aren't drawn yet either.
    app.canvas_states.get(&doc).map(|s| s.pending.iter().chain(s.in_flight.iter().flat_map(|(_, b)| b)).copied().collect()).unwrap_or_default()
}

/// Whether a sweep of `doc` still has blocks to draw.
pub(crate) fn running(app: &PhotocraftApp, doc: DocId) -> bool {
    app.canvas_states.get(&doc).is_some_and(|s| !s.pending.is_empty() || !s.in_flight.is_empty())
}

/// (Re)start a sweep of all of `doc`'s canvas, the blocks nearest `visible` first.
pub(crate) fn start(app: &mut PhotocraftApp, doc: &Document, visible: Rect) {
    let state = app.canvas_states.entry(doc.id).or_default();
    state.pending = blocks(doc.bounds(), doc.bounds(), visible, block_size(state.gpu_ms_per_mp));
    state.thread_failed = false;
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

/// Add `region` of `doc`'s canvas to its sweep. Blocks of `region` still waiting from before
/// (what is being redrawn was touched again) go first; its other blocks take turns with the rest
/// of the sweep, a job of each at a time, so a change elsewhere (a storyboard hidden or shown) no
/// longer holds up the rest of the canvas until it is drawn.
pub(crate) fn add(app: &mut PhotocraftApp, doc: &Document, region: Rect, visible: Rect) {
    let region = region.intersect(&doc.bounds());
    if region.is_empty() {
        return;
    }
    let state = app.canvas_states.entry(doc.id).or_default();
    // A new redraw tries the redraw thread again, whatever happened to the last one.
    if state.pending.is_empty() && state.in_flight.is_empty() {
        state.thread_failed = false;
    }
    // Blocks sit on one grid over the canvas, so a block already pending is the same rectangle:
    // it moves to the front instead of being queued again. A Move drag of a large layer adds its
    // old and new place every frame; queued again each time, the sweep grew to several times
    // the canvas (6375 MP over a minute) and the blocks of earlier places waited behind the
    // newer ones. Popped from the end: the new blocks first.
    let new = blocks(region, doc.bounds(), visible, block_size(state.gpu_ms_per_mp));
    state.pending = queue_after_add(std::mem::take(&mut state.pending), new);
}

/// The sweep's queue (`pending`, popped from the end) once `new` (popped from the end too) joins
/// it: see [`add`].
fn queue_after_add(pending: Vec<Rect>, new: Vec<Rect>) -> Vec<Rect> {
    use super::redraw_thread::JOB_BLOCKS;
    let key = |b: &Rect| (b.x0, b.y0, b.x1, b.y1);
    let waiting: std::collections::HashSet<_> = pending.iter().map(key).collect();
    let queued: std::collections::HashSet<_> = new.iter().map(key).collect();
    // In the order they are drawn.
    let rest: Vec<Rect> = pending.into_iter().rev().filter(|b| !queued.contains(&key(b))).collect();
    let (again, fresh): (Vec<Rect>, Vec<Rect>) = new.into_iter().rev().partition(|b| waiting.contains(&key(b)));
    let mut order = again;
    let (mut a, mut b) = (fresh.chunks(JOB_BLOCKS), rest.chunks(JOB_BLOCKS));
    loop {
        match (a.next(), b.next()) {
            (None, None) => break,
            (x, y) => {
                order.extend(x.into_iter().flatten());
                order.extend(y.into_iter().flatten());
            }
        }
    }
    order.reverse();
    order
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
pub(crate) fn step(app: &mut PhotocraftApp, ctx: &egui::Context, gpu: &GpuCanvas, doc: &Arc<Document>, display: Option<&CanvasDisplay>) -> Step {
    land_finished(app);
    if app.canvas_states.get(&doc.id).is_some_and(|s| !s.thread_failed) && super::redraw_thread::thread(app, gpu, ctx).is_some() {
        return step_threaded(app, ctx, gpu, doc, display);
    }
    let started = crate::gpu_canvas::now_ms();
    let mut gpu_estimate = 0.0;
    if let Some(s) = app.canvas_states.get_mut(&doc.id) {
        let totals = s.totals.get_or_insert_with(|| Totals { started_ms: started, work: Work::planned(doc), ..Totals::default() });
        totals.frame(crate::gpu_canvas::now_ms());
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

/// Keep `doc`'s sweep going on the redraw thread on a frame that drew an edit of its own (a brush
/// step, a drag): [`step`] ran only on frames with nothing new, so the rest of the canvas waited
/// until the stroke or drag ended. Never draws on this thread; without the redraw thread the sweep
/// waits as before, rather than take frame time from the edit.
pub(crate) fn keep_fed(app: &mut PhotocraftApp, ctx: &egui::Context, gpu: &GpuCanvas, doc: &Arc<Document>, display: Option<&CanvasDisplay>) {
    if !running(app, doc.id) {
        return;
    }
    land_finished(app);
    if app.canvas_states.get(&doc.id).is_some_and(|s| !s.thread_failed) && super::redraw_thread::thread(app, gpu, ctx).is_some() {
        let _ = step_threaded(app, ctx, gpu, doc, display);
    }
}

/// [`step`] on the redraw thread: take the jobs it finished, keep it fed with [`IN_FLIGHT`] jobs
/// of the next blocks, and say whether the sweep is over. Never draws on this thread.
///
/// [`IN_FLIGHT`]: super::redraw_thread::IN_FLIGHT
fn step_threaded(app: &mut PhotocraftApp, ctx: &egui::Context, gpu: &GpuCanvas, doc: &Arc<Document>, display: Option<&CanvasDisplay>) -> Step {
    use super::redraw_thread::{IN_FLIGHT, JOB_BLOCKS};
    let now = crate::gpu_canvas::now_ms();
    super::redraw_thread::publish(app, doc, display.is_some_and(|d| d.encode_srgb));
    if let Some(s) = app.canvas_states.get_mut(&doc.id) {
        let totals = s.totals.get_or_insert_with(|| Totals { started_ms: now, work: Work::planned(doc), ..Totals::default() });
        totals.frame(crate::gpu_canvas::now_ms());
    }
    let failed = app.canvas_states.get(&doc.id).is_some_and(|s| s.thread_failed || s.thread_wait_until > now);

    if !failed {
        // Keep the thread fed: the next blocks (popped from the end) in jobs of a few.
        while let Some(s) = app.canvas_states.get_mut(&doc.id) {
            if s.in_flight.len() >= IN_FLIGHT || s.pending.is_empty() {
                break;
            }
            let keep = s.pending.len().saturating_sub(JOB_BLOCKS);
            let blocks: Vec<Rect> = s.pending.drain(keep..).rev().collect();
            let Some(ticket) = app.redraw_thread.as_mut().and_then(|t| t.send(doc.id, blocks.clone())) else {
                if let Some(s) = app.canvas_states.get_mut(&doc.id) {
                    s.pending.extend(blocks.into_iter().rev());
                    s.thread_failed = true;
                }
                break;
            };
            if let Some(s) = app.canvas_states.get_mut(&doc.id) {
                s.in_flight.push((ticket, blocks));
            }
        }
    }
    if !running(app, doc.id) {
        return finish(app, gpu, doc.id, Step::Done);
    }
    // The thread asks for a repaint as each job lands; this is a backstop.
    ctx.request_repaint_after(std::time::Duration::from_millis(100));
    Step::Continuing
}

/// Take the jobs the redraw thread finished, of any document: count the drawn blocks in their
/// sweep's totals, and give failed ones back to the UI thread's sweep.
fn land_finished(app: &mut PhotocraftApp) {
    let finished = app.redraw_thread.as_ref().map(|t| t.finished()).unwrap_or_default();
    for done in finished {
        let Some(s) = app.canvas_states.get_mut(&done.doc) else { continue };
        s.in_flight.retain(|(ticket, _)| *ticket != done.ticket);
        match done.result {
            Ok(stats) => {
                let px = done.blocks.iter().map(|b| b.size().area()).sum();
                let r = crate::gpu_canvas::Refresh {
                    kind: "gpu-rect",
                    px,
                    composite_ms: done.cpu_ms,
                    uploads: stats.tiles_uploaded as u64,
                    gpu: stats.gpu_time,
                    stats: Some(stats),
                    ..Default::default()
                };
                s.account(px, done.blocks.len() as u32, &r);
                if let Some(t) = s.totals.as_mut() {
                    t.threaded = true;
                }
            }
            // Not ready yet (the compositor still building at startup, or out on a UI-thread
            // render): the blocks wait a moment and go again.
            Err(e) if e.0 == crate::gpu_canvas::COMPOSITOR_BUILDING => {
                s.pending.extend(done.blocks);
                s.thread_wait_until = crate::gpu_canvas::now_ms() + THREAD_RETRY_MS;
            }
            Err(e) => {
                // Back to the UI thread's sweep, which redraws them (or falls back) as before.
                log::info!("canvas: the redraw thread couldn't draw {} block(s) ({}); going on on the UI thread", done.blocks.len(), e.0);
                s.pending.extend(done.blocks);
                s.thread_failed = true;
            }
        }
    }
}

impl CanvasState {
    /// Count `blocks` drawn blocks of `px` pixels in all in the sweep's totals and learn the GPU's
    /// rate from `r`; returns the GPU time they are estimated to take.
    fn account(&mut self, px: u64, blocks: u32, r: &crate::gpu_canvas::Refresh) -> f64 {
        // One render that held its thread for long: what it drew and uploaded, to find what made
        // a frame slow (painting a canvas-wide layer during a first draw ended with the window
        // dropped by the compositor).
        if r.composite_ms > SLOW_RENDER_MS {
            log::info!(
                "canvas: one render of {blocks} block(s) ({:.0} MP) took {:.0} ms; {} layer tiles uploaded{}",
                px as f64 / 1_048_576.0,
                r.composite_ms,
                r.uploads,
                r.stats
                    .map(|s| format!("; plan {:.1}, cache check {:.1}, setup {:.1}, cells {:.1} ms", s.plan_ms, s.prepare_ms, s.setup_ms, s.cells_ms))
                    .unwrap_or_default()
            );
        }
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
    // A render plans the document once, however many blocks it draws: that part is paid once.
    let w = &t.work;
    let plan = if w.renders > 0 { w.stage_ms[0] / w.renders as f64 } else { 0.0 };
    let cpu = ((t.cpu_ms - plan * w.renders as f64) / f64::from(t.blocks)).max(0.1);
    let gpu = s.gpu_ms_per_mp.unwrap_or(0.0) * (next.size().area() as f64 / 1_048_576.0);
    let left = FRAME_BUDGET_MS - (crate::gpu_canvas::now_ms() - started).max(gpu_estimate) - plan;
    let each = cpu.max(gpu).max(0.1);
    ((left / each).floor().max(1.0) as usize).min(MAX_BATCH)
}

/// A thin bar along the bottom of the canvas view with the share of a long sweep drawn so far
/// (shown once it has run for half a second). Numbers only, so it needs no translation.
pub(crate) fn draw_progress(app: &PhotocraftApp, painter: &egui::Painter, doc: DocId) {
    let Some(s) = app.canvas_states.get(&doc) else { return };
    let Some(t) = s.totals.as_ref() else { return };
    let left = s.pending.len() + s.in_flight.iter().map(|(_, b)| b.len()).sum::<usize>();
    if left == 0 || crate::gpu_canvas::now_ms() - t.started_ms < 500.0 {
        return;
    }
    let done = f64::from(t.blocks) / (f64::from(t.blocks) + left as f64);
    let view = painter.clip_rect();
    let bar = egui::Rect::from_min_max(egui::pos2(view.left(), view.bottom() - 4.0), view.right_bottom());
    let fill = egui::Rect::from_min_max(bar.min, egui::pos2(bar.left() + bar.width() * done as f32, bar.bottom()));
    painter.rect_filled(bar, 0.0, egui::Color32::from_black_alpha(140));
    painter.rect_filled(fill, 0.0, egui::Color32::from_rgb(70, 140, 230));
    let label = format!("{:.0}%", done * 100.0);
    let at = egui::pos2(view.left() + 8.0, bar.top() - 4.0);
    let galley = painter.layout_no_wrap(label, egui::FontId::proportional(12.0), egui::Color32::WHITE);
    let back = egui::Rect::from_min_size(egui::pos2(at.x - 4.0, at.y - galley.size().y - 2.0), galley.size() + egui::vec2(8.0, 4.0));
    painter.rect_filled(back, 3.0, egui::Color32::from_black_alpha(160));
    painter.galley(egui::pos2(at.x, at.y - galley.size().y), galley, egui::Color32::WHITE);
}

/// A render slower than this (ms) is logged on its own.
const SLOW_RENDER_MS: f64 = 150.0;

/// Most blocks drawn in one render.
const MAX_BATCH: usize = 4;

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
            "canvas: redrew {:.0} MP in {} blocks over {} frames, {:.0} ms ({:.0} ms encoding {}; GPU {gpu_rate}; uploaded {} layer tiles; layer pages {} of {budget} MB{}){}",
            t.px as f64 / 1_048_576.0,
            t.blocks,
            t.frames,
            {
                let now = crate::gpu_canvas::now_ms();
                t.active_ms + (now - t.last_frame_ms).clamp(0.0, MAX_FRAME_GAP_MS)
            },
            t.cpu_ms,
            if t.threaded { "on the redraw thread" } else { "on the UI thread" },
            t.uploads,
            mb(resident),
            t.work.describe(t.blocks) + &t.work.stages(t.blocks, t.cpu_ms),
            {
                let waited = (crate::gpu_canvas::now_ms() - t.started_ms) - t.active_ms;
                let waited = if waited > 1000.0 { format!("; waited {:.1} s while not shown", waited / 1000.0) } else { String::new() };
                waited + if step == Step::Failed { "; stopped" } else { "" }
            },
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
    fn a_new_region_takes_turns_with_the_sweep_and_blocks_waiting_again_go_first() {
        let r = |x: i32| Rect::new(x, 0, x + 1, 1);
        // Drawn in the order 0, 1, ... 7 (popped from the end).
        let pending: Vec<Rect> = (0..8).rev().map(r).collect();
        // A region of blocks 100..106 and block 5, which was still waiting.
        let new: Vec<Rect> = [100, 101, 102, 5, 103, 104, 105].into_iter().rev().map(r).collect();
        let mut q = queue_after_add(pending, new);
        let mut order = Vec::new();
        while let Some(b) = q.pop() {
            order.push(b.x0);
        }
        assert_eq!(order, [5, 100, 101, 102, 0, 1, 2, 103, 104, 105, 3, 4, 6, 7]);
        // Nothing running: the region alone, in its own order.
        let mut q = queue_after_add(Vec::new(), [1, 2].into_iter().rev().map(r).collect());
        assert_eq!((q.pop().map(|b| b.x0), q.pop().map(|b| b.x0)), (Some(1), Some(2)));
    }

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
