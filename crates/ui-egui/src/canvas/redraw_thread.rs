//! The redraw thread: a sweep's blocks ([`super::sweep`]) composited off the UI thread.
//!
//! On the UI thread a sweep drew only what fit a frame's budget (~24 ms) and the GPU waited
//! between frames: on a storyboard file the GPU was busy about 15% of a first draw. Here a thread
//! takes the blocks as fast as the GPU and the compositor allow, while the UI thread keeps to
//! input and painting, and shows each block as it lands.
//!
//! The UI thread [`publish`]es the document it shows (every frame); the thread reads it once it
//! holds the compositor ([`crate::gpu_canvas::GpuCanvas::composite_lent`]), so a block always
//! shows the newest version its render could see. A failed block goes back to the UI thread's
//! sweep, which redraws it as before. Off with `PHOTOCRAFT_SWEEP_ON_UI_THREAD` (to compare).

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

use photocraft_doc::{DocId, Document};
use photocraft_geom::Rect;

use crate::PhotocraftApp;
use crate::gpu_canvas::GpuCanvas;

/// Blocks per job: small enough that an edit drawn on the UI thread waits little for the
/// compositor, large enough that the document is planned once for several.
pub(crate) const JOB_BLOCKS: usize = 3;
/// Jobs queued per document at a time, so the thread always has the next one.
pub(crate) const IN_FLIGHT: usize = 2;

type Latest = Arc<Mutex<HashMap<u64, (Arc<Document>, bool)>>>;

struct Job {
    key: u64,
    ticket: u64,
    blocks: Vec<Rect>,
}

/// A finished job, back on the UI thread.
pub(crate) struct Done {
    pub(crate) doc: DocId,
    pub(crate) ticket: u64,
    pub(crate) blocks: Vec<Rect>,
    pub(crate) result: Result<photocraft_gpu::Stats, photocraft_gpu::Unsupported>,
    /// CPU time the render took on the thread.
    pub(crate) cpu_ms: f64,
}

pub(crate) struct RedrawThread {
    tx: Sender<Job>,
    rx: Receiver<Done>,
    latest: Latest,
    next: u64,
}

/// Whether sweeps run on the redraw thread (not on the web, not in unit tests, which step
/// sweeps frame by frame, and not when switched off).
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !cfg!(target_arch = "wasm32") && !cfg!(test) && std::env::var_os("PHOTOCRAFT_SWEEP_ON_UI_THREAD").is_none())
}

impl RedrawThread {
    #[cfg(target_arch = "wasm32")]
    fn spawn(_: GpuCanvas, _: egui::Context) -> Option<Self> {
        None
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn spawn(gpu: GpuCanvas, ctx: egui::Context) -> Option<Self> {
        let (tx, jobs) = std::sync::mpsc::channel::<Job>();
        let (done, rx) = std::sync::mpsc::channel::<Done>();
        let latest: Latest = Arc::default();
        let shared = latest.clone();
        let spawned = std::thread::Builder::new().name("photocraft-redraw".into()).spawn(move || {
            while let Ok(job) = jobs.recv() {
                let t0 = crate::gpu_canvas::now_ms();
                let pick = || shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(&job.key).cloned();
                let result = gpu.composite_lent(pick, &job.blocks);
                let out = Done { doc: DocId(job.key), ticket: job.ticket, blocks: job.blocks, result, cpu_ms: crate::gpu_canvas::now_ms() - t0 };
                if done.send(out).is_err() {
                    break;
                }
                ctx.request_repaint();
            }
        });
        match spawned {
            Ok(_) => {
                log::info!("canvas: redrawing in blocks on the redraw thread");
                Some(RedrawThread { tx, rx, latest, next: 0 })
            }
            Err(e) => {
                log::warn!("canvas: could not start the redraw thread ({e}); redrawing on the UI thread");
                None
            }
        }
    }

    /// Queue `blocks` of document `doc`; returns the job's ticket, `None` when the thread is gone.
    pub(crate) fn send(&mut self, doc: DocId, blocks: Vec<Rect>) -> Option<u64> {
        self.next += 1;
        self.tx.send(Job { key: doc.0, ticket: self.next, blocks }).ok()?;
        Some(self.next)
    }

    /// The jobs finished since the last call.
    pub(crate) fn finished(&self) -> Vec<Done> {
        self.rx.try_iter().collect()
    }
}

/// The redraw thread, started the first time a sweep needs it.
pub(crate) fn thread<'a>(app: &'a mut PhotocraftApp, gpu: &GpuCanvas, ctx: &egui::Context) -> Option<&'a mut RedrawThread> {
    if !enabled() || app.redraw_thread_failed {
        return None;
    }
    if app.redraw_thread.is_none() {
        app.redraw_thread = RedrawThread::spawn(gpu.clone(), ctx.clone());
        app.redraw_thread_failed = app.redraw_thread.is_none();
    }
    app.redraw_thread.as_mut()
}

/// Tell the redraw thread which version of document `doc` the canvas shows now. Called every
/// frame before the canvas draws anything itself.
pub(crate) fn publish(app: &PhotocraftApp, doc: &Arc<Document>, encode_srgb: bool) {
    let Some(t) = app.redraw_thread.as_ref() else { return };
    let mut latest = t.latest.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let live: Vec<u64> = app.session.documents().iter().map(|d| d.doc.id.0).collect();
    latest.retain(|k, _| live.contains(k));
    latest.insert(doc.id.0, (doc.clone(), encode_srgb));
}
