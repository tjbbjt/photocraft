//! A large PSD/PSB shows its stored composite the moment it opens (`photocraft_io::saved_composite`):
//! read reduced on a thread, it is drawn over the parts of the canvas the first draw hasn't
//! reached yet (the sweep's pending blocks), so the document is on screen at once instead of
//! filling in over seconds. It goes as soon as the first draw is done, the document changes, or
//! the file turns out to have no composite to show.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};

use photocraft_doc::DocId;
use photocraft_geom::Rect;

use crate::PhotocraftApp;
use crate::canvas::ViewXform;

/// Longest side of the reduced composite: enough for a fitted view on a large monitor.
const MAX_SIDE: u32 = 4096;

/// A document's stored composite: being read, ready to draw, or done with.
pub(crate) enum OpenPreview {
    Loading {
        rx: Receiver<Option<photocraft_io::saved_composite::SavedComposite>>,
        cancel: Arc<AtomicBool>,
        revision: u64,
    },
    Ready {
        texture: egui::TextureHandle,
        revision: u64,
    },
    /// None to show, or no longer needed: never read again for this document.
    Done,
}

impl Drop for OpenPreview {
    fn drop(&mut self) {
        if let OpenPreview::Loading { cancel, .. } = self {
            cancel.store(true, Ordering::Relaxed);
        }
    }
}

/// Whether document `id`'s canvas has been drawn whole once: a texture exists and no sweep is
/// still filling it in.
fn first_draw_done(app: &PhotocraftApp, id: DocId) -> bool {
    let drawn = app.canvases.iter().any(|((d, _), c)| *d == id && (c.on_gpu || c.texture.is_some()));
    drawn && !crate::canvas::sweep::running(app, id)
}

/// Whether `path` is a file whose stored composite can be read here.
fn readable(path: &str) -> bool {
    let ext = std::path::Path::new(path).extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    matches!(ext.as_deref(), Some("psd" | "psb"))
}

/// Starts reading the stored composite of the document at `idx` when it has just opened, takes a
/// finished read, and drops what is no longer needed. Called once per frame per shown document.
fn update(app: &mut PhotocraftApp, ctx: &egui::Context, idx: usize) -> Option<DocId> {
    let st = app.session.documents().get(idx)?;
    let (id, revision, path) = (st.doc.id, st.revision, st.path.clone());
    let done = first_draw_done(app, id);
    match app.open_previews.get_mut(&id) {
        None => {
            // Only a document opened from a PSD/PSB, before its canvas has ever been drawn.
            let path = path.filter(|p| readable(p) && !done)?;
            #[cfg(not(target_arch = "wasm32"))]
            {
                let (tx, rx) = std::sync::mpsc::channel();
                let cancel = Arc::new(AtomicBool::new(false));
                let flag = cancel.clone();
                let ctx2 = ctx.clone();
                let started = std::thread::Builder::new().name("photocraft-open-preview".into()).spawn(move || {
                    let t0 = std::time::Instant::now();
                    let read = photocraft_io::saved_composite::saved_composite_at(std::path::Path::new(&path), MAX_SIDE, &|| flag.load(Ordering::Relaxed));
                    match &read {
                        Ok(Some(c)) => {
                            log::info!("open preview: stored composite read at {}×{} (1/{}) in {} ms", c.width, c.height, c.factor, t0.elapsed().as_millis())
                        }
                        Ok(None) => log::info!("open preview: the file has no stored composite to show"),
                        Err(e) => log::info!("open preview: the stored composite couldn't be read: {e}"),
                    }
                    let _ = tx.send(read.ok().flatten());
                    ctx2.request_repaint();
                });
                let preview = if started.is_ok() { OpenPreview::Loading { rx, cancel, revision } } else { OpenPreview::Done };
                app.open_previews.insert(id, preview);
            }
            #[cfg(target_arch = "wasm32")]
            {
                let _ = (path, ctx);
                app.open_previews.insert(id, OpenPreview::Done);
            }
        }
        Some(p) => {
            let stale = match p {
                OpenPreview::Loading { revision: r, .. } | OpenPreview::Ready { revision: r, .. } => *r != revision || done,
                OpenPreview::Done => false,
            };
            if stale {
                *p = OpenPreview::Done;
            } else if let OpenPreview::Loading { rx, revision: r, .. } = p {
                match rx.try_recv() {
                    Ok(Some(c)) => {
                        let image = egui::ColorImage::from_rgba_unmultiplied([c.width as usize, c.height as usize], &c.rgba);
                        let texture = ctx.load_texture(format!("open-preview-{}", id.0), image, egui::TextureOptions::LINEAR);
                        *p = OpenPreview::Ready { texture, revision: *r };
                    }
                    Ok(None) | Err(TryRecvError::Disconnected) => *p = OpenPreview::Done,
                    Err(TryRecvError::Empty) => {}
                }
            }
        }
    }
    Some(id)
}

/// Draws the stored composite of the document at `idx` over the parts of its canvas not drawn
/// yet. Image content: called before the grid, guides and selection.
pub(crate) fn draw(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, idx: usize) {
    retain(app);
    let Some(id) = update(app, painter.ctx(), idx) else { return };
    let Some(OpenPreview::Ready { texture, .. }) = app.open_previews.get(&id) else { return };
    let Some(st) = app.session.documents().get(idx) else { return };
    let canvas = st.doc.bounds();
    if canvas.is_empty() {
        return;
    }
    let pending = crate::canvas::sweep::pending(app, id);
    let rects: Vec<Rect> = if pending.is_empty() { vec![canvas] } else { pending };
    let mut mesh = egui::Mesh::with_texture(texture.id());
    let (w, h) = (canvas.width() as f32, canvas.height() as f32);
    for r in rects {
        let r = r.intersect(&canvas);
        if r.is_empty() {
            continue;
        }
        let corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)];
        let base = mesh.vertices.len() as u32;
        for (x, y) in corners {
            let (x, y) = (x as f32, y as f32);
            let uv = egui::pos2((x - canvas.x0 as f32) / w, (y - canvas.y0 as f32) / h);
            mesh.vertices.push(egui::epaint::Vertex { pos: xf.to_screen(x, y), uv, color: egui::Color32::WHITE });
        }
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    if !mesh.indices.is_empty() {
        painter.add(egui::Shape::mesh(mesh));
    }
}

/// Forget closed documents' previews (and stop reading theirs).
fn retain(app: &mut PhotocraftApp) {
    let live: Vec<DocId> = app.session.documents().iter().map(|d| d.doc.id).collect();
    app.open_previews.retain(|id, _| live.contains(id));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_psd_and_psb_files_are_read() {
        assert!(readable("/a/board.psb"));
        assert!(readable("/a/board.PSD"));
        assert!(!readable("/a/board.png"));
        assert!(!readable("/a/board"));
    }
}
