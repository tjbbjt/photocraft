//! Raws of one File › Open that are still being developed. As in Photoshop's Camera Raw, the
//! open-time dialog's filmstrip has a tile for every raw of the set from the start, showing the
//! camera's embedded preview until the raw itself is developed. Picking a tile that isn't
//! developed yet moves its open to the front of the queue, and the dialog shows it when it lands.
//! Open opens the raws still on their way with their defaults; Cancel and Done stop them.

use super::PhotocraftApp;
use super::open::Target;
use photocraft_doc::DocId;
use std::sync::{Arc, Mutex};

/// Raw file name extensions (File › Open's "RAW images" filter): only these get a tile before
/// they are decoded. The open itself still decides by the file's bytes.
const RAW_EXTENSIONS: &[&str] = &["dng", "cr2", "cr3", "nef", "nrw", "arw", "pef", "orf", "rw2", "raf"];

/// Long side of a tile's embedded preview, in pixels (the filmstrip's thumbnail size).
const PREVIEW_SIDE: u32 = 240;

/// A filmstrip tile for a raw of the set.
pub struct RawTile {
    pub(super) name: String,
    pub(super) path: String,
    /// The raw's document once it has been developed and joined the filmstrip's raws.
    pub(super) document: Option<DocId>,
    /// The open ended without a raw for the dialog (it failed, or opened as its preview).
    pub(super) gone: bool,
    /// The embedded preview, filled in by a worker thread.
    preview: Arc<Mutex<Option<egui::ColorImage>>>,
    pub(super) tex: Option<egui::TextureHandle>,
}

/// What happens to a raw of the set that lands after the dialog closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AfterDialog {
    /// Open was clicked: it opens developed with its defaults (sidecar, Auto), without the dialog.
    OpenPlain,
    /// Cancel or Done: it closes again.
    Close,
}

/// The tiles of the set being opened, and what the dialog asked of them.
#[derive(Default)]
pub struct RawComing {
    pub(super) tiles: Vec<RawTile>,
    /// The path of the tile picked before its raw was developed: shown as soon as it is.
    pub(super) wanted: Option<String>,
    /// Raws of a set whose dialog has closed, by path.
    after: Vec<(String, AfterDialog)>,
}

/// Whether the open-time Camera Raw dialog is up, or about to be (a raw of the set is waiting
/// for it or still being developed): the other files' open prompts wait until it has closed.
pub(crate) fn holds_prompts(app: &PhotocraftApp) -> bool {
    app.raw.as_ref().is_some_and(|s| s.target == Target::OpenRaw) || !app.raw_shelf.is_empty() || app.raw_coming.tiles.iter().any(|t| t.document.is_none() && !t.gone)
}

/// Is `path` named like a raw file?
pub(crate) fn is_raw_name(path: &str) -> bool {
    std::path::Path::new(path).extension().and_then(|e| e.to_str()).is_some_and(|e| RAW_EXTENSIONS.iter().any(|r| e.eq_ignore_ascii_case(r)))
}

/// A raw at `path` starts opening in the background: it gets a filmstrip tile now, with its
/// embedded preview read on a worker, so the dialog shows the whole set when it first opens. A
/// raw opened while the dialog is already open joins it the old way (when it lands, in the
/// raws Open develops, without a tile).
pub(crate) fn expect_raw(app: &mut PhotocraftApp, name: &str, path: &str) {
    // Opened again: whatever the last dialog decided for this file no longer applies.
    app.raw_coming.after.retain(|(p, _)| p != path);
    if !app.background_jobs || !app.session.prefs().raw_defaults.open_in_camera_raw || !is_raw_name(path) {
        return;
    }
    if app.raw.as_ref().is_some_and(|s| s.target == Target::OpenRaw) {
        return;
    }
    // The tiles of an earlier set that never reached the dialog (every open failed) go.
    if app.raw.is_none() && app.raw_shelf.is_empty() && still_coming(app).is_empty() {
        forget_tiles(app);
    }
    let coming = &mut app.raw_coming;
    if coming.tiles.iter().any(|t| t.path == path && !t.gone) {
        return;
    }
    let preview = Arc::new(Mutex::new(None));
    read_preview(path.to_string(), Arc::clone(&preview));
    coming.tiles.push(RawTile { name: name.to_string(), path: path.to_string(), document: None, gone: false, preview, tex: None });
}

/// Reads the embedded preview of the raw at `path` into `into`, off the UI thread.
#[cfg(not(target_arch = "wasm32"))]
fn read_preview(path: String, into: Arc<Mutex<Option<egui::ColorImage>>>) {
    let _ = std::thread::Builder::new().name("raw tile preview".into()).spawn(move || {
        let Ok(bytes) = std::fs::read(&path) else { return };
        let Some((w, h, rgba)) = photocraft_io::raw::tile_preview(&bytes, PREVIEW_SIDE) else { return };
        let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
        if let Ok(mut slot) = into.lock() {
            *slot = Some(img);
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn read_preview(_path: String, _into: Arc<Mutex<Option<egui::ColorImage>>>) {}

/// The raw of tile `path` was developed into `document` and joined the filmstrip's raws.
pub(super) fn landed(app: &mut PhotocraftApp, path: &str, document: DocId) {
    if let Some(tile) = app.raw_coming.tiles.iter_mut().find(|t| t.path == path && t.document.is_none() && !t.gone) {
        tile.document = Some(document);
    }
}

/// The open of `path` ended: a tile still without its raw won't get one.
pub(crate) fn open_ended(app: &mut PhotocraftApp, path: &str) {
    let mut gone = false;
    for tile in app.raw_coming.tiles.iter_mut().filter(|t| t.path == path && t.document.is_none()) {
        tile.gone = true;
        gone = true;
    }
    // A wanted raw that landed is shown by the dialog's next frame; one that won't land isn't.
    if gone && app.raw_coming.wanted.as_deref() == Some(path) {
        app.raw_coming.wanted = None;
    }
    // An open stopped after its dialog closed has nothing left to close.
    app.raw_coming.after.retain(|(p, _)| p != path);
}

/// What to do with the raw of `path` that just opened, when its set's dialog has already closed
/// (consumed: a later open of the same file is an open like any other).
pub(crate) fn after_dialog(app: &mut PhotocraftApp, path: Option<&str>) -> Option<AfterDialog> {
    let path = path?;
    let at = app.raw_coming.after.iter().position(|(p, _)| p == path)?;
    Some(app.raw_coming.after.remove(at).1)
}

/// The paths of the tiles whose raws are still being developed.
fn still_coming(app: &PhotocraftApp) -> Vec<String> {
    app.raw_coming.tiles.iter().filter(|t| t.document.is_none() && !t.gone).map(|t| t.path.clone()).collect()
}

/// The dialog closed with Open: the raws still on their way open with their defaults.
pub(super) fn open_the_rest_plain(app: &mut PhotocraftApp) {
    for path in still_coming(app) {
        app.raw_coming.after.push((path, AfterDialog::OpenPlain));
    }
    forget_tiles(app);
}

/// The dialog closed with Cancel or Done: the raws still on their way stop (an open that ends
/// anyway closes again).
pub(super) fn stop_the_rest(app: &mut PhotocraftApp) {
    for path in still_coming(app) {
        if !crate::jobs_ui::cancel_open(app, &path) {
            app.raw_coming.after.push((path, AfterDialog::Close));
        }
    }
    forget_tiles(app);
}

fn forget_tiles(app: &mut PhotocraftApp) {
    app.raw_coming.tiles.clear();
    app.raw_coming.wanted = None;
}

/// Where the open-time dialog's filmstrip tiles come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TileRef {
    /// A developed raw: its place in `app.raw_shelf`.
    Ready(usize),
    /// A raw still being developed: its place in the set's tiles.
    Coming(usize),
}

/// The filmstrip in order: the set's tiles as they were opened (developed or not), then any
/// developed raw that isn't one of them.
pub(super) fn tile_order(app: &PhotocraftApp) -> Vec<TileRef> {
    let shelf_index = |doc: DocId| app.raw_shelf.iter().position(|h| h.raw.document == doc);
    let mut order: Vec<TileRef> = Vec::new();
    for (i, tile) in app.raw_coming.tiles.iter().enumerate() {
        // A tile whose raw was developed and then left the shelf (opened or closed) stays, empty.
        order.push(match tile.document.and_then(shelf_index) {
            Some(s) => TileRef::Ready(s),
            None => TileRef::Coming(i),
        });
    }
    for s in 0..app.raw_shelf.len() {
        if !order.contains(&TileRef::Ready(s)) {
            order.push(TileRef::Ready(s));
        }
    }
    order
}

/// Tile `i`'s embedded preview as a texture, once the worker has read it.
pub(super) fn preview_texture(app: &mut PhotocraftApp, ctx: &egui::Context, i: usize) -> Option<egui::load::SizedTexture> {
    let tile = app.raw_coming.tiles.get_mut(i)?;
    if tile.tex.is_none() {
        let img = tile.preview.lock().ok()?.take()?;
        tile.tex = Some(ctx.load_texture(format!("camera-raw-tile-{i}-{}", tile.path), img, egui::TextureOptions::LINEAR));
    }
    tile.tex.as_ref().map(egui::load::SizedTexture::from_handle)
}

/// Tile `i` was picked before its raw was developed: its open goes first, and the dialog shows
/// it when it lands.
pub(super) fn want(app: &mut PhotocraftApp, i: usize) -> Result<(), String> {
    let Some(tile) = app.raw_coming.tiles.get(i) else { return Ok(()) };
    let (name, path) = (tile.name.clone(), tile.path.clone());
    if tile.gone || tile.document.is_some() {
        return Err(crate::i18n::fmt(tl!("{name} can't be shown here: it didn't open as a raw"), &[("name", &name)]));
    }
    crate::jobs_ui::hurry_open(app, &path);
    app.raw_coming.wanted = Some(path);
    app.ui.status = crate::i18n::fmt(tl!("Developing {name}…"), &[("name", &name)]);
    app.ui.status_error = false;
    Ok(())
}

/// The filmstrip place of the wanted raw, once it has been developed (and forgets the wish).
pub(super) fn take_wanted(app: &mut PhotocraftApp) -> Option<usize> {
    let path = app.raw_coming.wanted.clone()?;
    let i = app.raw_coming.tiles.iter().position(|t| t.path == path)?;
    app.raw_coming.tiles[i].document?;
    let at = tile_order(app).iter().position(|r| matches!(r, TileRef::Ready(s) if Some(app.raw_shelf[*s].raw.document) == app.raw_coming.tiles[i].document))?;
    app.raw_coming.wanted = None;
    Some(at)
}
