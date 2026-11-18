//! Fill models for Content-Aware Fill: LaMa (Apache-2.0) and MI-GAN (MIT), run by rten, a pure
//! Rust engine that reads the publishers' ONNX files as they are, on the processor.
//!
//! A model fills a picture's hole at a working size (the caller shrinks the picture first): in
//! 512-pixel tiles that overlap by a quarter, each keeping only what it filled in its middle (or
//! out to the picture's edge), so every tile sees the fills before it. [`Model::Mix`] runs both and keeps,
//! block by block, whichever fill matches the picture around it better.
//!
//! The model files are not part of PhotoCraft: they are read from [`models_dir`]. Loaded models
//! stay in memory, and the last few fills of each model are kept, and the tiles of the last
//! fills, so a preview and the fill that follows it on the same picture run the model once, and
//! a fill that follows a preview stopped part-way reuses the tiles it got through.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use photocraft_raster::Interrupt;
use rten::Model as Graph;
use rten_tensor::{AsView, NdTensor};

/// The models' side.
pub const TILE: usize = 512;
/// How far apart tiles start: they overlap by a quarter. Half (256) ran half as many tiles
/// again for a guide whose detail comes from the photo anyway.
const STEP: usize = TILE * 3 / 4;
/// Fills kept per model, and tiles kept in all.
const FILLS_KEPT: usize = 4;
const TILES_KEPT: usize = 96;
/// The longest side a picture is shrunk to before a model fills it.
pub const WORKING_SIDE: usize = 2048;
/// LaMa's file (Carve/LaMa-ONNX, `lama_fp32.onnx`, 208 044 816 bytes).
pub const LAMA_FILE: &str = "lama_fp32.onnx";
/// MI-GAN's file (andraniksargsyan/migan, `migan_pipeline_v2.onnx`).
pub const MIGAN_FILE: &str = "migan_pipeline_v2.onnx";

/// Which model fills.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Model {
    Lama,
    Migan,
    /// Both, combined block by block.
    Mix,
}

impl Model {
    /// From Content-Aware Fill's `method` parameter (`lama`, `migan`, `mix`).
    pub fn parse(s: &str) -> Option<Model> {
        match s {
            "lama" => Some(Model::Lama),
            "migan" => Some(Model::Migan),
            "mix" => Some(Model::Mix),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Model::Lama => "lama",
            Model::Migan => "migan",
            Model::Mix => "mix",
        }
    }
}

/// Why a model could not fill.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Its file is not in the models folder.
    Missing(PathBuf),
    /// The file is there but is not the model expected, or the run failed.
    Failed(String),
    Cancelled,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Missing(p) => write!(f, "the model file {} is missing: put it in {}", file_name(p), p.parent().unwrap_or(Path::new("")).display()),
            Error::Failed(why) => write!(f, "the fill model failed: {why}"),
            Error::Cancelled => f.write_str("cancelled"),
        }
    }
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Where the model files are read from: `PHOTOCRAFT_MODELS` when set, else
/// `$XDG_DATA_HOME/photocraft/models` (in the Flatpak `~/.var/app/ai.storyteller.photocraft/data/photocraft/models`),
/// else `~/.local/share/photocraft/models`.
pub fn models_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("PHOTOCRAFT_MODELS").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    let data = std::env::var_os("XDG_DATA_HOME")
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_default();
    data.join("photocraft").join("models")
}

/// A picture at its working size: RGB bytes and its hole (true = fill).
pub struct Picture<'a> {
    pub w: usize,
    pub h: usize,
    pub rgb: &'a [u8],
    pub hole: &'a [bool],
}

/// `pic` with its hole filled by `model` (RGB bytes, the picture's size). `ctl` is asked between
/// tiles whether to stop, and told the share of tiles done.
pub fn fill(model: Model, pic: &Picture, ctl: &Interrupt) -> Result<Vec<u8>, Error> {
    assert_eq!(pic.rgb.len(), pic.w * pic.h * 3);
    assert_eq!(pic.hole.len(), pic.w * pic.h);
    match model {
        Model::Lama | Model::Migan => fill_one(model, pic, ctl, (0.0, 1.0)),
        Model::Mix => {
            let a = fill_one(Model::Lama, pic, ctl, (0.0, 0.8))?;
            let b = fill_one(Model::Migan, pic, ctl, (0.8, 1.0))?;
            Ok(mix(pic, &a, &b))
        }
    }
}

/// The last fills of each model ([`FILLS_KEPT`]), by what they filled.
static FILLS: Mutex<Vec<Fill>> = Mutex::new(Vec::new());

/// A model's fill and the key of what it filled ([`key`]).
type Fill = (Model, u64, Arc<Vec<u8>>);

/// A key for what `pic` asks a model to fill.
fn key(pic: &Picture) -> u64 {
    let mut h = DefaultHasher::new();
    (pic.w, pic.h).hash(&mut h);
    pic.rgb.hash(&mut h);
    pic.hole.hash(&mut h);
    h.finish()
}

/// One model's fill of `pic`, reporting progress across `span` of `ctl`'s range.
fn fill_one(model: Model, pic: &Picture, ctl: &Interrupt, span: (f32, f32)) -> Result<Vec<u8>, Error> {
    let k = key(pic);
    if let Some(done) = FILLS.lock().ok().and_then(|f| f.iter().find(|(m, kk, _)| *m == model && *kk == k).map(|(_, _, v)| v.clone())) {
        ctl.progress(span.1);
        return Ok(done.to_vec());
    }
    let graph = load(model)?;
    let out = tiles(pic, ctl, span, |rgb, hole| {
        run_tile(model, rgb, hole, || match model {
            Model::Migan => run_migan(&graph, rgb, hole),
            _ => run_lama(&graph, rgb, hole),
        })
    })?;
    if let Ok(mut f) = FILLS.lock() {
        if f.iter().filter(|(m, _, _)| *m == model).count() >= FILLS_KEPT
            && let Some(oldest) = f.iter().position(|(m, _, _)| *m == model)
        {
            f.remove(oldest);
        }
        f.push((model, k, Arc::new(out.clone())));
    }
    Ok(out)
}

/// The tiles run lately ([`TILES_KEPT`]), by model and what the tile held.
static TILE_FILLS: Mutex<Vec<Fill>> = Mutex::new(Vec::new());

/// One tile's fill by `model`: kept from an earlier run that gave it the same pixels and hole,
/// else `run`, and kept.
fn run_tile(model: Model, rgb: &[u8], hole: &[bool], run: impl FnOnce() -> Result<Vec<u8>, Error>) -> Result<Vec<u8>, Error> {
    let mut h = DefaultHasher::new();
    rgb.hash(&mut h);
    hole.hash(&mut h);
    let k = h.finish();
    if let Some(done) = TILE_FILLS.lock().ok().and_then(|t| t.iter().find(|(m, kk, _)| *m == model && *kk == k).map(|(_, _, v)| v.clone())) {
        return Ok(done.to_vec());
    }
    let out = run()?;
    if let Ok(mut t) = TILE_FILLS.lock() {
        if t.len() >= TILES_KEPT {
            t.remove(0);
        }
        t.push((model, k, Arc::new(out.clone())));
    }
    Ok(out)
}

/// The loaded graphs, by file.
static GRAPHS: Mutex<Vec<(&'static str, Arc<Graph>)>> = Mutex::new(Vec::new());

/// `model`'s graph, loaded from [`models_dir`] the first time.
fn load(model: Model) -> Result<Arc<Graph>, Error> {
    let file = if model == Model::Migan { MIGAN_FILE } else { LAMA_FILE };
    if let Some(g) = GRAPHS.lock().ok().and_then(|g| g.iter().find(|(f, _)| *f == file).map(|(_, g)| g.clone())) {
        return Ok(g);
    }
    let path = models_dir().join(file);
    if !path.is_file() {
        return Err(Error::Missing(path));
    }
    let graph = Arc::new(Graph::load_file(&path).map_err(|e| Error::Failed(format!("{} could not be loaded: {e}", path.display())))?);
    if let Ok(mut g) = GRAPHS.lock() {
        g.push((file, graph.clone()));
    }
    Ok(graph)
}

/// The tiles along a side of `n`: each tile's start (every [`STEP`], the last one flush with the
/// end) and its middle `lo..hi`, the part it keeps. The middles split the side between them: two
/// neighbours meet halfway through their overlap, the first and last reach the ends.
fn spans(n: usize) -> Vec<(usize, usize, usize)> {
    if n <= TILE {
        return vec![(0, 0, n)];
    }
    let mut starts: Vec<usize> = (0..n - TILE).step_by(STEP).collect();
    starts.push(n - TILE);
    starts.dedup();
    let meet = |a: usize, b: usize| (a + TILE + b) / 2;
    (0..starts.len())
        .map(|i| {
            let lo = if i == 0 { 0 } else { meet(starts[i - 1], starts[i]) };
            let hi = if i + 1 == starts.len() { n } else { meet(starts[i], starts[i + 1]) };
            (starts[i], lo, hi)
        })
        .collect()
}

/// A tile to fill: its top-left corner and the middle it keeps (`x0..x1`, `y0..y1`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Tile {
    left: usize,
    top: usize,
    keep: (usize, usize, usize, usize),
}

/// The tiles whose middle holds some of `hole`, row by row.
fn tiles_to_fill(pic: &Picture) -> Vec<Tile> {
    let (w, h) = (pic.w, pic.h);
    let mut out = Vec::new();
    for &(top, y0, y1) in &spans(h) {
        for &(left, x0, x1) in &spans(w) {
            if (y0..y1).any(|y| pic.hole[y * w + x0..y * w + x1].iter().any(|v| *v)) {
                out.push(Tile { left, top, keep: (x0, y0, x1, y1) });
            }
        }
    }
    out
}

/// The tile at `(x0, y0)` of `rgb` and `left` (pixels past the picture repeat its edge and count
/// as known).
fn cut(w: usize, h: usize, rgb: &[u8], left: &[bool], x0: usize, y0: usize) -> (Vec<u8>, Vec<bool>) {
    let mut trgb = vec![0u8; TILE * TILE * 3];
    let mut thole = vec![false; TILE * TILE];
    for ty in 0..TILE {
        for tx in 0..TILE {
            let (x, y) = ((x0 + tx).min(w - 1), (y0 + ty).min(h - 1));
            let (i, t) = (y * w + x, ty * TILE + tx);
            thole[t] = x0 + tx < w && y0 + ty < h && left[i];
            trgb[t * 3..t * 3 + 3].copy_from_slice(&rgb[i * 3..i * 3 + 3]);
        }
    }
    (trgb, thole)
}

/// `pic` filled tile by tile by `run`: each tile keeps what it filled in its middle.
fn tiles(pic: &Picture, ctl: &Interrupt, span: (f32, f32), run: impl Fn(&[u8], &[bool]) -> Result<Vec<u8>, Error>) -> Result<Vec<u8>, Error> {
    let (w, h) = (pic.w, pic.h);
    let mut rgb = pic.rgb.to_vec();
    let mut left = pic.hole.to_vec();
    let todo = tiles_to_fill(pic);
    for (n, tile) in todo.iter().enumerate() {
        if ctl.cancelled() {
            return Err(Error::Cancelled);
        }
        let (trgb, thole) = cut(w, h, &rgb, &left, tile.left, tile.top);
        let out = run(&trgb, &thole)?;
        let (x0, y0, x1, y1) = tile.keep;
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y * w + x;
                if left[i] {
                    let t = (y - tile.top) * TILE + (x - tile.left);
                    rgb[i * 3..i * 3 + 3].copy_from_slice(&out[t * 3..t * 3 + 3]);
                    left[i] = false;
                }
            }
        }
        ctl.progress(span.0 + (span.1 - span.0) * (n + 1) as f32 / todo.len().max(1) as f32);
    }
    Ok(rgb)
}

fn failed(e: impl std::fmt::Display) -> Error {
    Error::Failed(e.to_string())
}

/// LaMa on one tile: `image` in 0..1 with the hole zeroed, `mask` 1 = fill; `output` in 0..255
/// (scaled up when it comes in 0..1).
fn run_lama(g: &Graph, rgb: &[u8], hole: &[bool]) -> Result<Vec<u8>, Error> {
    let n = TILE * TILE;
    let mut img = vec![0.0f32; 3 * n];
    let mut mask = vec![0.0f32; n];
    for i in 0..n {
        mask[i] = if hole[i] { 1.0 } else { 0.0 };
        for c in 0..3 {
            img[c * n + i] = if hole[i] { 0.0 } else { f32::from(rgb[i * 3 + c]) / 255.0 };
        }
    }
    let (image_id, mask_id, out_id) = (g.node_id("image").map_err(failed)?, g.node_id("mask").map_err(failed)?, g.node_id("output").map_err(failed)?);
    let image = NdTensor::from_data([1, 3, TILE, TILE], img);
    let mask = NdTensor::from_data([1, 1, TILE, TILE], mask);
    let [out] = g.run_n(vec![(image_id, image.view().into()), (mask_id, mask.view().into())], [out_id], None).map_err(failed)?;
    let out: NdTensor<f32, 4> = out.try_into().map_err(|_| failed("LaMa's output is not a 4-d f32 tensor"))?;
    let v = out.to_vec();
    if v.len() != 3 * n {
        return Err(failed(format!("LaMa's output has {} values, not {}", v.len(), 3 * n)));
    }
    let scale = if v.iter().fold(0.0f32, |a, b| a.max(*b)) <= 1.5 { 255.0 } else { 1.0 };
    Ok((0..n).flat_map(|i| (0..3).map(move |c| (i, c))).map(|(i, c)| (v[c * n + i] * scale).round().clamp(0.0, 255.0) as u8).collect())
}

/// MI-GAN's published pipeline on one tile: `image` RGB bytes, `mask` 255 = known and 0 = fill;
/// `result` RGB bytes, blended across the hole's edge, so the hole first takes its nearest known
/// colours (blended with an empty border's black it left dark lines where tiles meet).
fn run_migan(g: &Graph, rgb: &[u8], hole: &[bool]) -> Result<Vec<u8>, Error> {
    let n = TILE * TILE;
    let rgb = nearest_known(rgb, hole);
    let mut img = vec![0u8; 3 * n];
    let mut mask = vec![0u8; n];
    for i in 0..n {
        mask[i] = if hole[i] { 0 } else { 255 };
        for c in 0..3 {
            img[c * n + i] = rgb[i * 3 + c];
        }
    }
    let (image_id, mask_id, out_id) = (g.node_id("image").map_err(failed)?, g.node_id("mask").map_err(failed)?, g.node_id("result").map_err(failed)?);
    let image = NdTensor::from_data([1, 3, TILE, TILE], img);
    let mask = NdTensor::from_data([1, 1, TILE, TILE], mask);
    let [out] = g.run_n(vec![(image_id, image.view().into()), (mask_id, mask.view().into())], [out_id], None).map_err(failed)?;
    let out: NdTensor<u8, 4> = out.try_into().map_err(|_| failed("MI-GAN's output is not a 4-d u8 tensor"))?;
    let v = out.to_vec();
    if v.len() != 3 * n {
        return Err(failed(format!("MI-GAN's output has {} values, not {}", v.len(), 3 * n)));
    }
    let v = &v;
    Ok((0..n).flat_map(|i| (0..3).map(move |c| v[c * n + i])).collect())
}

/// A tile's RGB bytes with each hole pixel given the colour of its nearest known pixel (breadth
/// first, 4-connected); unchanged when the tile has no known pixel.
pub fn nearest_known(rgb: &[u8], hole: &[bool]) -> Vec<u8> {
    let mut out = rgb.to_vec();
    let mut done: Vec<bool> = hole.iter().map(|h| !h).collect();
    let mut front: Vec<usize> = (0..hole.len()).filter(|&i| done[i]).collect();
    while !front.is_empty() {
        let mut next = Vec::new();
        for &i in &front {
            for j in neighbours(i, TILE, hole.len() / TILE) {
                if !done[j] {
                    done[j] = true;
                    out.copy_within(i * 3..i * 3 + 3, j * 3);
                    next.push(j);
                }
            }
        }
        front = next;
    }
    out
}

/// The 4-connected neighbours of cell `i` of a `w × h` grid.
fn neighbours(i: usize, w: usize, h: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (i % w, i / w);
    [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)].into_iter().flatten()
}

/// Side of [`mix`]'s blocks.
const BLOCK: usize = 16;

/// Mean luma and mean step between neighbouring pixels over the pixels of a box that `keep`
/// allows; `None` when it allows none.
fn stats(w: usize, v: &[u8], (x0, y0, x1, y1): (usize, usize, usize, usize), keep: &dyn Fn(usize) -> bool) -> Option<(f32, f32)> {
    let luma = |i: usize| (0.299 * f32::from(v[i * 3]) + 0.587 * f32::from(v[i * 3 + 1]) + 0.114 * f32::from(v[i * 3 + 2])) / 255.0;
    let (mut sum, mut step, mut n, mut m) = (0.0f32, 0.0f32, 0usize, 0usize);
    for y in y0..y1 {
        for x in x0..x1 {
            let i = y * w + x;
            if !keep(i) {
                continue;
            }
            let l = luma(i);
            sum += l;
            n += 1;
            for (ok, j) in [(x + 1 < x1, i + 1), (y + 1 < y1, i + w)] {
                if ok && keep(j) {
                    step += (l - luma(j)).abs();
                    m += 1;
                }
            }
        }
    }
    (n > 0).then(|| (sum / n as f32, step / m.max(1) as f32))
}

/// Per block: 1 where `b` matches the known picture around the block better, 0 where `a` does,
/// 0.5 where there is nothing to fill or compare. A fill matches when its mean luma and mean step
/// (its amount of detail) are close to those of the known pixels within 48 pixels (128 where none
/// is that close).
fn picks(pic: &Picture, a: &[u8], b: &[u8]) -> (usize, usize, Vec<f32>) {
    let (w, h) = (pic.w, pic.h);
    let (bw, bh) = (w.div_ceil(BLOCK), h.div_ceil(BLOCK));
    let hole = |i: usize| pic.hole[i];
    let known = |i: usize| !pic.hole[i];
    let mut pick = vec![0.5f32; bw * bh];
    for by in 0..bh {
        for bx in 0..bw {
            let r = (bx * BLOCK, by * BLOCK, ((bx + 1) * BLOCK).min(w), ((by + 1) * BLOCK).min(h));
            let (Some(fa), Some(fb)) = (stats(w, a, r, &hole), stats(w, b, r, &hole)) else { continue };
            let around = |m: usize| stats(w, pic.rgb, (r.0.saturating_sub(m), r.1.saturating_sub(m), (r.2 + m).min(w), (r.3 + m).min(h)), &known);
            let Some((cl, cd)) = around(48).or_else(|| around(128)) else { continue };
            let score = |(l, d): (f32, f32)| (d - cd).abs() / (cd + 0.01) + 4.0 * (l - cl).abs();
            pick[by * bw + bx] = if score(fb) < score(fa) { 1.0 } else { 0.0 };
        }
    }
    (bw, bh, pick)
}

/// Two fills of `pic` combined block by block ([`picks`]), blended between block centres.
pub fn mix(pic: &Picture, a: &[u8], b: &[u8]) -> Vec<u8> {
    let (bw, bh, pick) = picks(pic, a, b);
    let mut out = pic.rgb.to_vec();
    for y in 0..pic.h {
        for x in 0..pic.w {
            let i = y * pic.w + x;
            if !pic.hole[i] {
                continue;
            }
            let fx = ((x as f32 + 0.5) / BLOCK as f32 - 0.5).clamp(0.0, (bw - 1) as f32);
            let fy = ((y as f32 + 0.5) / BLOCK as f32 - 0.5).clamp(0.0, (bh - 1) as f32);
            let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(bw - 1), (y0 + 1).min(bh - 1));
            let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
            let at = |bx: usize, by: usize| pick[by * bw + bx];
            let t = (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty) + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty;
            for c in 0..3 {
                out[i * 3 + c] = (f32::from(a[i * 3 + c]) * (1.0 - t) + f32::from(b[i * 3 + c]) * t).round() as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
