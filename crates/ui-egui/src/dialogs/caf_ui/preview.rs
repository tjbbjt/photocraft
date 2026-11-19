use super::{CMD, CafWorkspace, Color32, LayerId, MAX_CAF_POINTS, MAX_OUTLINE_SEGMENTS, PhotocraftApp, Rect, TextureHandle, Value, json, stroke_json};
use crate::outline::Segment;

/// What the plan (without the brush strokes) says: the window, the fill area and the sampling
/// option's area, over `rect`.
pub(super) struct Base {
    pub(super) window: Rect,
    pub(super) rect: Rect,
    pub(super) hole: Vec<bool>,
    pub(super) base: Vec<bool>,
}

impl Base {
    /// The index into `hole` and `base` of document point `(x, y)`, when it is inside `rect`.
    pub(super) fn index(&self, x: f64, y: f64) -> Option<usize> {
        let (xi, yi) = (x.floor() as i32, y.floor() as i32);
        self.rect.contains(xi, yi).then(|| (yi - self.rect.y0) as usize * self.rect.width() as usize + (xi - self.rect.x0) as usize)
    }
}

/// A computed preview: the filled rectangle as an image, or why it failed.
#[cfg(not(target_arch = "wasm32"))]
type PreviewResult = Result<(Rect, egui::ColorImage), String>;

/// The preview being computed or shown.
#[derive(Default)]
pub(super) struct Preview {
    /// Bumped whenever the state changes; a result for an older generation is dropped.
    pub(super) want: u64,
    #[cfg(not(target_arch = "wasm32"))]
    running: Option<(u64, std::sync::mpsc::Receiver<PreviewResult>, photocraft_engine::jobs::JobCtx)>,
    pub(super) shown: u64,
    /// A finished preview waiting to be uploaded (on the UI thread, which has the context).
    pub(super) pending_image: Option<(Rect, egui::ColorImage)>,
    pub(super) tex: Option<(TextureHandle, Rect)>,
    /// Screen px per document px (0 = frame the window on the next frame) and the view centre.
    pub(super) zoom: f32,
    pub(super) center: [f64; 2],
    pub(super) error: Option<String>,
    /// The base being worked out on a worker ([`start_base`]).
    #[cfg(not(target_arch = "wasm32"))]
    base_running: Option<BaseRun>,
    /// When the running preview started, for the timer beside the spinner.
    pub(super) started: Option<web_time::Instant>,
    /// How long the last preview and the last Apply / OK fill took, s (shown in the panel).
    pub(super) last_secs: Option<f32>,
    pub(super) last_fill_secs: Option<f32>,
}

/// A base being worked out on a worker: its answer, its job (to cancel it), and what to do with it.
#[cfg(not(target_arch = "wasm32"))]
struct BaseRun {
    rx: std::sync::mpsc::Receiver<Result<Base, String>>,
    job: photocraft_engine::jobs::JobCtx,
    then: BaseFor,
}

/// Why the base is being worked out, which decides what its answer does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BaseFor {
    /// The workspace just opened: the brush takes its default size from the fill area.
    Open,
    /// A lasso edit or Expand / Contract: one that empties the fill area is refused (undone).
    Edit,
    /// Anything else (an option, Undo / Redo, an Apply that changed the document).
    Change,
}

/// Whether the base is worked out on a worker, so the app keeps running meanwhile: the plan
/// reads and analyses the whole sampling window at full size (17 s after each Apply on an 84 MP
/// photo, all of it a frozen window). The web build has no threads, and the tests read the base
/// at once.
pub(super) const BASE_ON_A_WORKER: bool = cfg!(not(any(test, target_arch = "wasm32")));

impl Preview {
    /// Takes the result of the preview of state `generation`: an image to upload, or the error,
    /// which stands for that state until it changes (the same preview is not started again).
    fn finish(&mut self, generation: u64, r: Result<(Rect, egui::ColorImage), String>) {
        self.shown = generation;
        self.tex = None;
        match r {
            Ok(img) => {
                self.pending_image = Some(img);
                self.error = None;
            }
            Err(e) => {
                self.pending_image = None;
                self.error = Some(e);
            }
        }
    }
}

/// The preview's error when the sampling area has no pixels outside the fill area (the engine's
/// fill refuses that case with the same words).
const NOTHING_TO_SAMPLE: &str = "nothing to sample from: the sampling area has no pixels outside the fill area";

/// The plan of the current state (without strokes when `strokes` is false).
pub(super) fn plan_params(w: &CafWorkspace, strokes: bool) -> Value {
    let o = &w.opts;
    let mut p = json!({
        "sampling": o.sampling,
        "sampleAllLayers": o.sample_all,
        "colorAdaptation": o.color_adaptation,
        "rotationAdaptation": o.rotation,
        "scale": o.scale,
        "mirror": o.mirror,
        "output": o.output,
        "method": o.fill_method(),
        "detail": o.detail,
        "stamp": o.stamp,
        "edgeGrow": o.edge().0,
        "edgeFeather": o.edge().1,
        "layer": w.layer.0,
        "fillEdits": w.fill_edits,
    });
    if strokes && !w.strokes.is_empty() {
        p["samplingStrokes"] = Value::Array(w.strokes.iter().map(stroke_json).collect());
    }
    p
}

pub(super) fn points_limit() -> String {
    format!("at most {MAX_CAF_POINTS} points in all sampling strokes and lasso edits: undo some, or reset")
}

/// The plan without strokes: the window, fill area and the option's area. Custom starts empty
/// (the window and fill area are those of Rectangular).
pub(super) fn compute_base(app: &PhotocraftApp, w: &CafWorkspace) -> Result<Base, String> {
    let (doc, layer, p, custom) = base_inputs(app, w)?;
    base_from(&doc, layer, &p, custom, &photocraft_raster::Interrupt::NONE)
}

/// What the base is worked out from: the document, the layer, the plan's params and whether the
/// option is Custom (which starts with an empty area).
fn base_inputs(app: &PhotocraftApp, w: &CafWorkspace) -> Result<(std::sync::Arc<photocraft_doc::Document>, LayerId, Value, bool), String> {
    let st = app.session.active().ok_or("no document")?;
    let mut p = plan_params(w, false);
    let custom = w.opts.sampling == "custom";
    if custom {
        p["sampling"] = json!("rectangular");
    }
    Ok((st.doc.clone(), w.layer, p, custom))
}

/// The base of `doc`'s plan for `p`, asking `ctl` whether to stop ("cancelled").
fn base_from(doc: &photocraft_doc::Document, layer: LayerId, p: &Value, custom: bool, ctl: &photocraft_raster::Interrupt) -> Result<Base, String> {
    let plan = photocraft_engine::edit_menu_cmds::content_aware_plan_with(doc, layer, p, CMD, ctl).map_err(|e| match e {
        photocraft_engine::EngineError::Cancelled => "cancelled".to_string(),
        e => e.to_string(),
    })?;
    let base = if custom { vec![false; plan.base.len()] } else { plan.base };
    Ok(Base { window: plan.window, rect: plan.rect, hole: plan.hole, base })
}

/// The base of the workspace's current state, worked out now.
fn base_now(app: &PhotocraftApp) -> Result<Base, String> {
    app.distort.caf.as_ref().map_or_else(|| Err("Content-Aware Fill is not open".to_string()), |w| compute_base(app, w))
}

/// Takes a worked-out base (or why there is none) into the workspace, as `then` says.
pub(super) fn take_base(app: &mut PhotocraftApp, r: Result<Base, String>, then: BaseFor) {
    let Some(w) = app.distort.caf.as_mut() else { return };
    match r {
        // A newer one replaced it.
        Err(e) if e == "cancelled" => return,
        Ok(b) => {
            if then == BaseFor::Open {
                w.brush_size = photocraft_algo::content_aware::default_brush_size(super::lifecycle::bounds(&b));
            }
            w.set_base(b);
            w.error = None;
        }
        Err(e) => {
            w.error = Some(e);
            w.restate_all();
        }
    }
    w.stale();
    // An edit that empties the fill area is refused (the plan says so): undo it, if it is still
    // the last step (a stroke painted while the base was worked out is not undone for it).
    if then == BaseFor::Edit
        && app.distort.caf.as_ref().is_some_and(|w| matches!(w.undo.last(), Some(super::Step::Fill(_))))
        && let Some(msg) = app.distort.caf.as_ref().and_then(|w| w.error.clone())
    {
        super::lifecycle::undo(app);
        if let Some(w) = app.distort.caf.as_mut() {
            w.redo.clear();
            w.error = Some(msg);
        }
    }
}

/// Starts working out the base on a worker (cancelling one still running); [`collect_base`]
/// takes it when it arrives. Where no thread can start, it is worked out now.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn start_base(app: &mut PhotocraftApp, then: BaseFor) {
    let Some(w) = app.distort.caf.as_ref() else { return };
    let inputs = base_inputs(app, w);
    let Some(w) = app.distort.caf.as_mut() else { return };
    if let Some(old) = w.preview.base_running.take() {
        old.job.cancel();
    }
    let (doc, layer, p, custom) = match inputs {
        Ok(i) => i,
        Err(e) => return take_base(app, Err(e), then),
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let job = photocraft_engine::jobs::JobCtx::new();
    let work = job.clone();
    let spawned = std::thread::Builder::new().name("caf-base".into()).spawn(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work.stage(0.0, 1.0, "", |ctl| base_from(&doc, layer, &p, custom, ctl))))
            .unwrap_or_else(|_| Err("working out the fill area failed".into()));
        let _ = tx.send(r);
    });
    match spawned {
        Ok(_) => w.preview.base_running = Some(BaseRun { rx, job, then }),
        Err(_) => {
            let r = base_now(app);
            take_base(app, r, then);
        }
    }
}

/// The web build works the base out now.
#[cfg(target_arch = "wasm32")]
pub(super) fn start_base(app: &mut PhotocraftApp, then: BaseFor) {
    let r = base_now(app);
    take_base(app, r, then);
}

/// Takes the worker's base if it has arrived; whether one is still being worked out.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn collect_base(app: &mut PhotocraftApp) -> bool {
    let Some(w) = app.distort.caf.as_mut() else { return false };
    let Some(run) = &w.preview.base_running else { return false };
    match run.rx.try_recv() {
        Ok(r) => {
            let then = run.then;
            w.preview.base_running = None;
            take_base(app, r, then);
            false
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => true,
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
            w.preview.base_running = None;
            false
        }
    }
}

/// Waits for a base being worked out and takes it (the control channel reads the result).
#[cfg(not(target_arch = "wasm32"))]
fn settle_base(app: &mut PhotocraftApp) {
    let Some(run) = app.distort.caf.as_mut().and_then(|w| w.preview.base_running.take()) else { return };
    let r = run.rx.recv().unwrap_or_else(|_| Err("cancelled".into()));
    take_base(app, r, run.then);
}

#[cfg(target_arch = "wasm32")]
fn settle_base(_: &mut PhotocraftApp) {}

/// The fill area's edges merged into runs, as the canvas traces a selection, in document px;
/// `None` past [`MAX_OUTLINE_SEGMENTS`].
pub(super) fn outline(b: &Base) -> Option<Vec<Segment>> {
    let area = super::lifecycle::bounds(b);
    let w = b.rect.width() as usize;
    let at = |x: i32, y: i32| b.rect.contains(x, y) && b.hole.get((y - b.rect.y0) as usize * w + (x - b.rect.x0) as usize).copied().unwrap_or(false);
    let mut segs = Vec::new();
    // Between rows y - 1 and y along x, then between columns x - 1 and x along y.
    for horizontal in [true, false] {
        let (l0, l1, t0, t1) = if horizontal { (area.y0, area.y1, area.x0, area.x1) } else { (area.x0, area.x1, area.y0, area.y1) };
        for l in l0..=l1 {
            let mut run = None;
            for t in t0..=t1 {
                let (p, q) = if horizontal { ((t, l), (t, l.saturating_sub(1))) } else { ((l, t), (l.saturating_sub(1), t)) };
                let edge = t < t1 && at(p.0, p.1) != at(q.0, q.1);
                match (edge, run) {
                    (true, None) => run = Some(t),
                    (false, Some(s)) => {
                        segs.push(if horizontal { ([s, l], [t, l]) } else { ([l, s], [l, t]) });
                        if segs.len() > MAX_OUTLINE_SEGMENTS {
                            return None;
                        }
                        run = None;
                    }
                    _ => {}
                }
            }
        }
    }
    Some(segs)
}

/// Recompute the base after a change of the fill area, the sampling option or the document.
pub(super) fn rebase(app: &mut PhotocraftApp) {
    rebase_for(app, BaseFor::Change);
}

/// [`rebase`], saying why: on a worker in the app, at once in tests and the web build.
pub(super) fn rebase_for(app: &mut PhotocraftApp, then: BaseFor) {
    if BASE_ON_A_WORKER {
        start_base(app, then);
    } else {
        let r = base_now(app);
        take_base(app, r, then);
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn cancel_preview(w: CafWorkspace) {
    if let Some((_, _, job)) = w.preview.running {
        job.cancel();
    }
    if let Some(run) = w.preview.base_running {
        run.job.cancel();
    }
}

#[cfg(target_arch = "wasm32")]
pub(super) fn cancel_preview(_: CafWorkspace) {}

/// The document changed under the workspace (an Apply): new textures, a new base.
pub(super) fn refresh(app: &mut PhotocraftApp) {
    let rev = app.session.active().map(|s| s.revision);
    if let Some(w) = app.distort.caf.as_mut() {
        w.image_tex = None;
        w.revision = rev.unwrap_or(w.revision);
    }
    rebase(app);
}

/// Computes the preview now, on this thread (tests and the control channel).
pub(super) fn wait_preview(app: &mut PhotocraftApp) -> Result<(), String> {
    settle_base(app);
    let (doc, layer, p, want, limit) = {
        let w = app.distort.caf.as_ref().ok_or("Content-Aware Fill is not open")?;
        let st = app.session.active().ok_or("no document")?;
        (st.doc.clone(), w.layer, plan_params(w, true), w.preview.want, w.texture_limit)
    };
    let r = compute_preview(&doc, layer, &p, &photocraft_raster::Interrupt::NONE, limit);
    let w = app.distort.caf.as_mut().ok_or("Content-Aware Fill is not open")?;
    let err = r.as_ref().err().cloned();
    w.preview.finish(want, r);
    err.map_or(Ok(()), Err)
}

/// The filled rectangle as an image (what Apply would write there), as a texture no wider than
/// `max_side`: past the device's limit it is box-averaged to fit (the rectangle it is drawn over
/// stays the plan's). Past that limit the fill itself runs on the box-averaged window
/// ([`reduced_preview`]): the texture could not show more, and a full-size fill of a 156 MP
/// panorama's borders held many GB for a preview two thousand pixels wide.
pub(super) fn compute_preview(
    doc: &photocraft_doc::Document,
    layer: LayerId,
    p: &Value,
    ctl: &photocraft_raster::Interrupt,
    max_side: usize,
) -> Result<(Rect, egui::ColorImage), String> {
    let plan = photocraft_engine::edit_menu_cmds::content_aware_plan_with(doc, layer, p, CMD, ctl).map_err(|e| match e {
        photocraft_engine::EngineError::Cancelled => "cancelled".to_string(),
        e => e.to_string(),
    })?;
    // Patches come only from the sampling area outside the fill area. With none (the Sampling
    // Brush erased it all, a Custom area not painted yet) the fill would come out black, so the
    // preview says so instead of showing that.
    if !plan.source.iter().any(|s| *s) {
        return Err(NOTHING_TO_SAMPLE.into());
    }
    let (w, h) = (plan.rect.width() as usize, plan.rect.height() as usize);
    let n = plan.fmt.channels();
    // A fill model works on the whole window first (at its working size), so a preview of any
    // size and the fill after it get the same model fill.
    let guide = photocraft_engine::edit_menu_cmds::model_guide(&plan, ctl).map_err(|e| match e {
        photocraft_engine::EngineError::Cancelled => "cancelled".to_string(),
        e => e.to_string(),
    })?;
    let f = crate::filter_preview_ui::texture_factor(w as u32, h as u32, max_side) as usize;
    if f > 1 {
        return reduced_preview(plan, f, guide, ctl);
    }
    let with = plan.model.as_ref().zip(guide.as_ref());
    let filled = photocraft_engine::edit_menu_cmds::caf_fill(w, h, &plan.fmt, plan.img.clone(), &plan.hole, &plan.source, &plan.opts, with, ctl)
        .map_err(|_| "cancelled".to_string())?;
    let own = plan.layer_pixels();
    // Pixel `i`: the fill over the fill area (a partly selected pixel by its coverage), the
    // layer's own pixels elsewhere.
    let pixel = |i: usize| -> [f32; 4] {
        let src = if plan.hole[i] { &filled[i * n..(i + 1) * n] } else { &own[i * n..(i + 1) * n] };
        let mut c = photocraft_raster::to_rgba(&plan.fmt, src);
        if plan.hole[i] {
            let k = plan.cover[i];
            let under = photocraft_raster::to_rgba(&plan.fmt, &own[i * n..(i + 1) * n]);
            for j in 0..4 {
                c[j] = under[j] + (c[j] - under[j]) * k;
            }
        }
        c
    };
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let color = |c: [f32; 4]| Color32::from_rgba_unmultiplied(b(c[0]), b(c[1]), b(c[2]), b(c[3]));
    let f = crate::filter_preview_ui::texture_factor(w as u32, h as u32, max_side) as usize;
    if f == 1 {
        let px = (0..w * h).map(|i| color(pixel(i))).collect();
        return Ok((plan.rect, egui::ColorImage::new([w, h], px)));
    }
    let (tw, th) = (w.div_ceil(f), h.div_ceil(f));
    let mut px = Vec::with_capacity(tw * th);
    for ty in 0..th {
        for tx in 0..tw {
            // The block's premultiplied average.
            let (mut acc, mut count) = ([0.0f32; 4], 0.0f32);
            for y in ty * f..((ty + 1) * f).min(h) {
                for x in tx * f..((tx + 1) * f).min(w) {
                    let c = pixel(y * w + x);
                    for j in 0..3 {
                        acc[j] += c[j] * c[3];
                    }
                    acc[3] += c[3];
                    count += 1.0;
                }
            }
            let c = if acc[3] > 0.0 { [acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3], acc[3] / count] } else { [0.0; 4] };
            px.push(color(c));
        }
    }
    Ok((plan.rect, egui::ColorImage::new([tw, th], px)))
}

/// [`compute_preview`] past the texture limit: the window box-averaged `f`× (the colours, the
/// layer's own pixels and the fill coverage), the fill area where any of a block is in it, the
/// sampling area where most of a block is and none of it is to be filled; then the fill at that
/// size, one texel per block. The full-size plan is freed before the fill.
fn reduced_preview(
    plan: photocraft_engine::edit_menu_cmds::CafPlan,
    f: usize,
    guide: Option<photocraft_engine::edit_menu_cmds::ModelGuide>,
    ctl: &photocraft_raster::Interrupt,
) -> Result<(Rect, egui::ColorImage), String> {
    let (w, h) = (plan.rect.width() as usize, plan.rect.height() as usize);
    let n = plan.fmt.channels();
    let (rw, rh) = (w.div_ceil(f), h.div_ceil(f));
    struct Block {
        img: Vec<f32>,
        /// The layer's own pixels the fill leaves (each weighted by how little of it is filled),
        /// straight RGBA: a block on the fill area's edge blends the fill with the pixels beside
        /// the area, not with the ones under it (black in a hole, which left a dark line).
        own: [f32; 4],
        hole: bool,
        source: bool,
        cover: f32,
    }
    let own_px = plan.layer_pixels();
    let block = |j: usize| -> Block {
        let (bx, by) = (j % rw, j / rw);
        let mut img = vec![0.0f32; n];
        let (mut own, mut count, mut sources, mut cover, mut hole, mut kept) = ([0.0f32; 4], 0.0f32, 0usize, 0.0f32, false, 0.0f32);
        for y in by * f..((by + 1) * f).min(h) {
            for x in bx * f..((bx + 1) * f).min(w) {
                let i = y * w + x;
                for (a, v) in img.iter_mut().zip(&plan.img[i * n..(i + 1) * n]) {
                    *a += v;
                }
                let o = photocraft_raster::to_rgba(&plan.fmt, &own_px[i * n..(i + 1) * n]);
                let keep = 1.0 - plan.cover[i];
                for (a, v) in own.iter_mut().zip(&o).take(3) {
                    *a += v * o[3] * keep;
                }
                own[3] += o[3] * keep;
                kept += keep;
                hole |= plan.hole[i];
                sources += usize::from(plan.source[i]);
                cover += plan.cover[i];
                count += 1.0;
            }
        }
        for a in &mut img {
            *a /= count;
        }
        let own = if own[3] > 0.0 { [own[0] / own[3], own[1] / own[3], own[2] / own[3], own[3] / kept] } else { [0.0; 4] };
        Block { img, own, hole, source: !hole && sources as f32 * 2.0 >= count, cover: cover / count }
    };
    // The web build has no thread pool.
    #[cfg(not(target_arch = "wasm32"))]
    let blocks: Vec<Block> = {
        use rayon::prelude::*;
        (0..rw * rh).into_par_iter().map(block).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let blocks: Vec<Block> = (0..rw * rh).map(block).collect();
    let (rect, fmt, opts, model) = (plan.rect, plan.fmt, plan.opts.clone(), plan.model.clone());
    drop(plan);
    if ctl.cancelled() {
        return Err("cancelled".into());
    }
    let img: Vec<f32> = blocks.iter().flat_map(|b| b.img.iter().copied()).collect();
    let hole: Vec<bool> = blocks.iter().map(|b| b.hole).collect();
    let source: Vec<bool> = blocks.iter().map(|b| b.source).collect();
    // A sampling area thinner than a block leaves no whole block: the preview says so, as the
    // full-size one does when there is nothing to copy.
    if !source.iter().any(|s| *s) {
        return Err(NOTHING_TO_SAMPLE.into());
    }
    let with = model.as_ref().zip(guide.as_ref());
    let filled = photocraft_engine::edit_menu_cmds::caf_fill(rw, rh, &fmt, img, &hole, &source, &opts, with, ctl).map_err(|_| "cancelled".to_string())?;
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let px = blocks
        .iter()
        .enumerate()
        .map(|(j, block)| {
            let under = block.own;
            let c: [f32; 4] = if block.hole {
                let fill = photocraft_raster::to_rgba(&fmt, &filled[j * n..(j + 1) * n]);
                let k = block.cover;
                std::array::from_fn(|i| under[i] + (fill[i] - under[i]) * k)
            } else {
                under
            };
            Color32::from_rgba_unmultiplied(b(c[0]), b(c[1]), b(c[2]), b(c[3]))
        })
        .collect();
    Ok((rect, egui::ColorImage::new([rw, rh], px)))
}

/// Starts a background preview when the shown one is out of date (and the pointer is up), and
/// collects a finished one.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn drive_preview(app: &mut PhotocraftApp, ctx: &egui::Context, dragging: bool) {
    // The preview waits for the base: both read the whole window, and the base comes first.
    if collect_base(app) {
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
        return;
    }
    // Nor does it run beside an Apply or OK fill: that holds the full-size window itself. A
    // running preview stops; the next one starts once the fill is done (it changes the document).
    if let Some(w) = app.distort.caf.as_mut()
        && w.pending_fill.is_some()
    {
        if let Some((_, _, job)) = w.preview.running.take() {
            job.cancel();
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
        return;
    }
    let doc = app.session.active().map(|s| s.doc.clone());
    let Some(w) = app.distort.caf.as_mut() else { return };
    let pv = &mut w.preview;
    if let Some((generation, rx, _)) = &pv.running {
        match rx.try_recv() {
            Ok(r) => {
                let generation = *generation;
                pv.running = None;
                let secs = pv.started.take().map(|t| t.elapsed().as_secs_f32());
                if generation == pv.want {
                    match r {
                        Err(e) if e == "cancelled" => {}
                        r => {
                            if let Some(secs) = secs {
                                pv.last_secs = Some(secs);
                                let how = if r.is_ok() { "done" } else { "failed" };
                                log::info!("content-aware preview ({}): {how} in {secs:.1} s", w.opts.fill_method());
                            }
                            pv.finish(generation, r)
                        }
                    }
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Superseded: stop it, a new one starts when it has gone.
                if *generation != pv.want
                    && let Some((_, _, job)) = &pv.running
                {
                    job.cancel();
                }
                ctx.request_repaint_after(std::time::Duration::from_millis(50));
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => pv.running = None,
        }
    }
    if pv.running.is_some() || pv.shown == pv.want || dragging {
        return;
    }
    let Some(doc) = doc else { return };
    let (layer, p, generation, limit) = (w.layer, plan_params(w, true), w.preview.want, w.texture_limit);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let job = photocraft_engine::jobs::JobCtx::new();
    let work = job.clone();
    let repaint = ctx.clone();
    let spawned = std::thread::Builder::new().name("caf-preview".into()).spawn(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work.stage(0.0, 1.0, "", |ctl| compute_preview(&doc, layer, &p, ctl, limit))))
            .unwrap_or_else(|_| Err("the preview failed".into()));
        let _ = tx.send(r);
        repaint.request_repaint();
    });
    match spawned {
        Ok(_) => {
            w.preview.running = Some((generation, rx, job));
            w.preview.started = Some(web_time::Instant::now());
        }
        Err(e) => w.preview.error = Some(format!("could not start the preview: {e}")),
    }
}

/// The web build has no threads: the preview is computed when the pointer comes up.
#[cfg(target_arch = "wasm32")]
pub(super) fn drive_preview(app: &mut PhotocraftApp, _ctx: &egui::Context, dragging: bool) {
    let stale = app.distort.caf.as_ref().is_some_and(|w| w.preview.shown != w.preview.want);
    if stale && !dragging {
        let _ = wait_preview(app);
    }
}
