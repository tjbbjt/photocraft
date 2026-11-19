//! Hole filling for Content-Aware Fill and Spot Healing: patch-based completion, proximity match
//! and texture synthesis.
//!
//! * [`complete`], content-aware fill: the multi-scale "space-time completion" EM scheme of
//!   Y. Wexler, E. Shechtman, M. Irani, *Space-Time Completion of Video*, IEEE TPAMI 2007, with the
//!   nearest-neighbour field computed by C. Barnes, E. Shechtman, A. Finkelstein, D. B. Goldman,
//!   *PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing*, SIGGRAPH 2009
//!   (random init, propagation, exponentially shrinking random search). Each EM iteration finds, for
//!   every patch overlapping the hole, its most similar source patch, then re-estimates hole pixels
//!   by a similarity-weighted vote of the overlapping matches. Coarse to fine over an image pyramid
//!   (down to where the hole is about four patches across): the coarsest level starts by filling
//!   the hole from its edge inwards, ring by ring, each pixel from the patch that best matches what
//!   is known around it (rather than from a smooth membrane, which draws the fill towards flat
//!   areas); every finer level starts from the coarser level's matches scaled up and voted with the
//!   finer level's own pixels, so the texture stays sharp. Where neighbouring patches agree on their
//!   source (the usual case), the vote copies the source pixels exactly; only along the seams
//!   between copied pieces does it blend.
//!
//!   A thin hole (a selected wire, string or hair: long, but only a few patches thick) is
//!   measured by its thickness, not its bounds (#1366): the pyramid stops where it is still about
//!   two patches thick. That level is big, so the onion peel starts it only where its exhaustive
//!   search stays affordable; otherwise a membrane across the hole does, which over so short a
//!   distance is the surroundings themselves. Shrunk to where its bounds are four patches across,
//!   a long thin hole swallows its surroundings, the matches found there copy whatever happens to
//!   look like the blur (a dark blotch, or a piece of another string), and the finer levels,
//!   which only refine those matches, keep them: the removed object's outline stays visible.
//!
//!   This is what Photoshop's Content-Aware Fill does, measured as a black box on Photoshop 25.4
//!   (see `content_aware`): with colour adaptation off its fill is pieces of the image copied pixel
//!   for pixel, typically 15–40 px across, joined by seams a few pixels wide; with colour
//!   adaptation the copies come with a gain and bias. That is the patch transform of S. Darabi,
//!   E. Shechtman, C. Barnes, D. B. Goldman, P. Sen, *Image Melding: Combining Inconsistent Images
//!   using Patch-based Synthesis*, SIGGRAPH 2012: each source patch may be scaled and offset per
//!   channel to match its target, within limits ([`Adapt`]), and is voted in adjusted.
//! * [`best_offset`], proximity match: the single displacement whose surrounding ring best matches the
//!   ring around the hole (sum of squared differences), i.e. an automatic Healing Brush source.
//! * [`synthesize`], create texture: patch-based texture synthesis in the style of A. Efros,
//!   W. Freeman, *Image Quilting for Texture Synthesis and Transfer*, SIGGRAPH 2001 (blocks picked from
//!   random candidates by overlap error, joined along minimum-error boundary cuts, so the texture is
//!   copied, never averaged).
//!
//! All functions take interleaved `w × h × ch` normalised floats (any colour model/depth) and a
//! `w × h` hole mask, and return a full buffer equal to the input outside the hole. They are
//! deterministic (fixed-seed hash RNG, results independent of thread count).

use std::borrow::Cow;

use crate::par;
use crate::poisson::membrane_fill;

mod proximity_integral;
use proximity_integral::HoleIntegral;

mod cells;
mod common;
mod solver;
mod sources;
mod synthesis;

use cells::Cells;
#[cfg(test)]
use common::Bounds;
pub(crate) use common::Rng;
pub use common::{Adapt, Px};
use common::{Level, SOURCE_MOMENTS_MAX};
use solver::Solver;
pub(crate) use solver::fit;
#[cfg(test)]
use solver::least_sum;
use sources::{Coarser, Sources, THIN_RATIO, downsample, hole_bounds, hole_thickness, peel_affordable, valid_sources, window_count};
#[cfg(test)]
use sources::{hole_bbox, integral};
use synthesis::onion_peel;
pub use synthesis::{best_offset, synthesize};

/// How strongly a guided fill favours coherent matches ([`Solver::coherence`]).
const GUIDED_COHERENCE: f32 = 6.0;
/// The least distance a nearby fill copies from ([`CompleteParams::nearby`]), full-size px.
const NEARBY_MIN: usize = 64;

/// Parameters for [`complete`].
#[derive(Clone, Debug, PartialEq)]
pub struct CompleteParams {
    /// Patch radius (patches are `2r+1` square).
    pub patch_radius: usize,
    /// EM iterations at the coarsest level (finer levels use fewer, down to `min_em_iters`).
    pub em_iters: usize,
    pub min_em_iters: usize,
    /// PatchMatch passes per EM iteration.
    pub pm_iters: usize,
    /// Levels where the hole is wider than this (in that level's pixels) only *refine* the
    /// scaled-up match field (one local PatchMatch pass and one vote), which keeps big holes
    /// tractable: the structure is decided at coarse scale, fine levels just add detail.
    pub refine_extent: usize,
    /// Colour adaptation of the copies.
    pub adapt: Adapt,
    pub seed: u64,
    /// Copy only from near each hole pixel: no farther than the hole is thick (at least 64 px).
    /// Classic copied whatever matched best from anywhere, a far, smaller-scale stretch of
    /// ground into a near one; nearby, the pieces are from the surroundings and keep their size.
    pub nearby: bool,
    /// A guided fill whose guide only lays the hole out: the hole starts as the guide, and from
    /// then on the fill matches, votes and keeps Classic's way. The guide's colours never steer a
    /// match, so the pieces are chosen and blended as Classic chooses them, only around a start
    /// that already has the rocks low and the grass high.
    pub layout_only: bool,
}

impl Default for CompleteParams {
    fn default() -> Self {
        Self { patch_radius: 3, em_iters: 10, min_em_iters: 3, pm_iters: 2, refine_extent: 192, adapt: Adapt::NONE, seed: 1, nearby: false, layout_only: false }
    }
}

/// Content-aware completion of `hole` (see module docs), copying from any pixel outside it. Returns
/// `None` when there is no fully-known patch to copy from (the caller can fall back to
/// [`membrane_fill`]).
pub fn complete(w: usize, h: usize, ch: usize, img: &[f32], hole: &[bool], p: &CompleteParams) -> Option<Vec<f32>> {
    // Never cancelled, so never `Err`.
    complete_with(w, h, ch, img, hole, p, &photocraft_raster::Interrupt::NONE).unwrap_or(None)
}

/// [`complete`] that can be cancelled (checked per EM iteration, PatchMatch pass and row band)
/// and reports progress (by pyramid level). `Err(Cancelled)` when `ctl` was cancelled.
pub fn complete_with(
    w: usize,
    h: usize,
    ch: usize,
    img: &[f32],
    hole: &[bool],
    p: &CompleteParams,
    ctl: &photocraft_raster::Interrupt,
) -> Result<Option<Vec<f32>>, photocraft_raster::Cancelled> {
    assert_eq!(w.checked_mul(h), Some(hole.len()));
    let px: Vec<Px> = hole.iter().map(|h| if *h { Px::Hole } else { Px::Source }).collect();
    complete_masked(w, h, ch, img, &px, p, ctl)
}

/// [`complete_with`] with every pixel's role given ([`Px`]): the hole, the context, what may be
/// copied, and what isn't image at all.
pub fn complete_masked(
    w: usize,
    h: usize,
    ch: usize,
    img: &[f32],
    px: &[Px],
    p: &CompleteParams,
    ctl: &photocraft_raster::Interrupt,
) -> Result<Option<Vec<f32>>, photocraft_raster::Cancelled> {
    Ok(complete_cow(w, h, ch, Cow::Borrowed(img), px, None, p, ctl)?.ok())
}

/// [`complete_masked`] that fills `img` itself instead of a copy of it (only the hole changes):
/// `Err(img)`, untouched, when there is nothing to copy from.
pub fn complete_masked_owned(
    w: usize,
    h: usize,
    ch: usize,
    img: Vec<f32>,
    px: &[Px],
    p: &CompleteParams,
    ctl: &photocraft_raster::Interrupt,
) -> Result<Result<Vec<f32>, Vec<f32>>, photocraft_raster::Cancelled> {
    Ok(complete_cow(w, h, ch, Cow::Owned(img), px, None, p, ctl)?.map_err(Cow::into_owned))
}

/// [`complete_masked_owned`] guided by `guide`, a picture the size of `img` of what the hole
/// should look like: a model's fill at a reduced size, scaled up `scale` times (1 for a guide at
/// full sharpness). The guide decides the layout (where the sky ends, which way a ridge runs) and
/// the picture the detail: at every level the hole's patches are matched as the guide shows them,
/// against the picture softened the same way, and the fill stamps the matched pieces of the
/// picture at their full sharpness, neighbours favouring one piece. The guide is never
/// overwritten, so the matching cannot drift away from it the way a fill that only starts from
/// it does.
#[allow(clippy::too_many_arguments)]
pub fn complete_guided_owned(
    w: usize,
    h: usize,
    ch: usize,
    img: Vec<f32>,
    px: &[Px],
    guide: &[f32],
    scale: f32,
    p: &CompleteParams,
    ctl: &photocraft_raster::Interrupt,
) -> Result<Result<Vec<f32>, Vec<f32>>, photocraft_raster::Cancelled> {
    assert_eq!(guide.len(), img.len(), "the guide is the picture's size");
    Ok(complete_cow(w, h, ch, Cow::Owned(img), px, Some((guide, scale.max(1.0))), p, ctl)?.map_err(Cow::into_owned))
}

/// A model's fill brought to the tones of the picture around it, before it guides a fill. In
/// each region of `tile` px (and its eight neighbours), each of the first `colour` channels of
/// `guide`'s hole pixels gets a gain and an offset so their mean and spread match the known
/// pixels there; between regions the two are interpolated. A model's fill tends to be lighter
/// and flatter than the photo deep inside a large hole (it settles on an average); matched
/// against such a guide, the fill picked light, smooth pieces of the photo and came out hazy.
/// `guide` holds the picture outside the hole; only its hole pixels change.
pub fn match_guide_tones(w: usize, h: usize, ch: usize, colour: usize, guide: &mut [f32], hole: &[bool], tile: usize) {
    let (cc, tile) = (colour.min(ch), tile.max(8));
    if w == 0 || h == 0 || cc == 0 || guide.len() != w * h * ch || hole.len() != w * h {
        return;
    }
    let (tw, th) = (w.div_ceil(tile), h.div_ceil(tile));
    // Per region and channel: hole pixels' count, sum and sum of squares, then the known ones'.
    let rows: Vec<Vec<[f64; 6]>> = par::map(th, w * h >= common::PAR_MIN, |ty| {
        let mut out = vec![[0.0f64; 6]; tw * cc];
        for y in ty * tile..((ty + 1) * tile).min(h) {
            for x in 0..w {
                let i = y * w + x;
                let k = if hole[i] { 0 } else { 3 };
                for c in 0..cc {
                    let v = f64::from(guide[i * ch + c]);
                    let s = &mut out[(x / tile) * cc + c];
                    s[k] += 1.0;
                    s[k + 1] += v;
                    s[k + 2] += v * v;
                }
            }
        }
        out
    });
    let at = |tx: usize, ty: usize, c: usize| rows[ty][tx * cc + c];
    let add = |a: [f64; 6], b: [f64; 6]| std::array::from_fn::<f64, 6, _>(|i| a[i] + b[i]);
    let least = (tile * tile / 16).max(16) as f64;
    let fit = |s: [f64; 6]| -> Option<(f32, f32)> {
        if s[0] < least || s[3] < least {
            return None;
        }
        let (mh, mk) = (s[1] / s[0], s[4] / s[3]);
        let (sh, sk) = ((s[2] / s[0] - mh * mh).max(0.0).sqrt(), (s[5] / s[3] - mk * mk).max(0.0).sqrt());
        let gain = ((sk + 1e-3) / (sh + 1e-3)).clamp(0.67, 1.5);
        Some((gain as f32, (mk - gain * mh) as f32))
    };
    let global: Vec<(f32, f32)> = (0..cc)
        .map(|c| fit((0..th).flat_map(|ty| (0..tw).map(move |tx| (tx, ty))).fold([0.0; 6], |a, (tx, ty)| add(a, at(tx, ty, c)))).unwrap_or((1.0, 0.0)))
        .collect();
    let params: Vec<(f32, f32)> = (0..th)
        .flat_map(|ty| (0..tw).map(move |tx| (tx, ty)))
        .flat_map(|(tx, ty)| {
            (0..cc).map(move |c| {
                let mut s = [0.0f64; 6];
                for ny in ty.saturating_sub(1)..(ty + 2).min(th) {
                    for nx in tx.saturating_sub(1)..(tx + 2).min(tw) {
                        s = add(s, at(nx, ny, c));
                    }
                }
                (s, c)
            })
        })
        .map(|(s, c)| fit(s).unwrap_or(global[c]))
        .collect();
    let ps = |tx: usize, ty: usize, c: usize| params[(ty * tw + tx) * cc + c];
    par::chunks_mut(guide, w * ch, w * h >= common::PAR_MIN, |y, row| {
        let fy = ((y as f32 + 0.5) / tile as f32 - 0.5).clamp(0.0, (th - 1) as f32);
        let (y0, ay) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(th - 1);
        for x in 0..w {
            if !hole[y * w + x] {
                continue;
            }
            let fx = ((x as f32 + 0.5) / tile as f32 - 0.5).clamp(0.0, (tw - 1) as f32);
            let (x0, ax) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(tw - 1);
            for c in 0..cc {
                let lerp = |a: (f32, f32), b: (f32, f32), t: f32| (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
                let (g, b) = lerp(lerp(ps(x0, y0, c), ps(x1, y0, c), ax), lerp(ps(x0, y1, c), ps(x1, y1, c), ax), ay);
                let v = &mut row[x * ch + c];
                *v = *v * g + b;
            }
        }
    });
}

/// The completion, on the caller's pixels or a copy of them, guided or not: `Err` gives them
/// back when there is nothing to copy from.
#[allow(clippy::too_many_arguments)]
fn complete_cow<'a>(
    w: usize,
    h: usize,
    ch: usize,
    img: Cow<'a, [f32]>,
    px: &'a [Px],
    guide: Option<(&[f32], f32)>,
    p: &CompleteParams,
    ctl: &photocraft_raster::Interrupt,
) -> Result<Result<Vec<f32>, Cow<'a, [f32]>>, photocraft_raster::Cancelled> {
    // Checked: wrapped products must not let tiny buffers pass for a huge `h`.
    assert_eq!(w.checked_mul(h), Some(px.len()));
    assert_eq!(w.checked_mul(h).and_then(|n| n.checked_mul(ch)), Some(img.len()));
    if w == 0 || h == 0 {
        // Nothing to fill; no per-row tables for a zero-width image of any height.
        return Ok(Ok(img.into_owned()));
    }
    let Some(hb0) = hole_bounds(w, h, px) else { return Ok(Ok(img.into_owned())) };
    if ch == 0 {
        return Ok(Ok(Vec::new()));
    }
    let r = p.patch_radius.max(1);
    let psz = 2 * r + 1;
    let Pyramid { mut levels, boxes, exts, thin, thickness } = build_pyramid(Level { w, h, img, px: Cow::Borrowed(px) }, ch, hb0, psz, ctl)?;
    let nlev = levels.len();
    // The guide at each level's size, finest first, and how much softer than that level it is.
    let guides: Option<(Vec<Vec<f32>>, f32)> = guide.map(|(g, scale)| (guide_pyramid(g, w, h, ch, nlev), scale));
    // The coarser level's result.
    let mut prev: Option<Coarser> = None;
    let total_px: usize = levels.iter().map(|l| l.w * l.h).sum();
    let mut done_px = 0usize;
    let ri = r as i32;
    for li in (0..nlev).rev() {
        ctl.check()?;
        let Level { w: lw, h: lh, img: limg, px: lpx } = std::mem::replace(&mut levels[li], Level::EMPTY);
        let hb = boxes[li];
        let tb = hb.grow(r, lw, lh);
        let valid = valid_sources(lw, lh, r, &lpx, ctl).ok_or(photocraft_raster::Cancelled)?;
        let sources = Sources::new(&valid, lw);
        // Nothing to copy from at the finest level: the caller falls back (and gets its pixels
        // back as they were).
        if li == 0 && sources.len() == 0 {
            return Ok(Err(limg));
        }
        // The level's own pixels to fill (at level 0 the caller's, or a copy made in parallel).
        let mut limg = into_owned_pixels(limg);
        let cells = Cells::around(lw, r, &lpx, hb, tb);
        ctl.check()?;
        // Initialise the hole: at the coarsest level peeled from its edge inwards (or a membrane
        // where nothing can be copied), else the coarser level's pixels, refined below by a vote of
        // its scaled-up matches.
        let coarser = prev.take();
        match &coarser {
            Some(c) => copy_up_coarser(&mut limg, &lpx, lw, ch, hb, c),
            // Guided: the hole starts as the guide.
            None if guides.as_ref().is_some_and(|(g, _)| g[li].len() == limg.len()) => {
                let g = guides.as_ref().map_or(&[][..], |(g, _)| &g[li][..]);
                for (i, k) in lpx.iter().enumerate() {
                    if *k == Px::Hole {
                        limg[i * ch..(i + 1) * ch].copy_from_slice(&g[i * ch..(i + 1) * ch]);
                    }
                }
            }
            // A thin hole, when peeling it would take too long, takes the membrane below: a few
            // patches thick here, a membrane across it is its surroundings.
            None if sources.len() > 0 && (!thin || peel_affordable(lw, &lpx, hb, sources.len())) => {
                onion_peel(lw, lh, ch, ri, &mut limg, &lpx, hb, &sources.all(&valid, lw), ctl)?
            }
            None => fill_hole_with_membrane(&mut limg, &lpx, (lw, lh, ch), r, thin),
        }
        // A guided fill's guide at this level, and how many of this level's pixels soft it is.
        let steer = guides.is_some() && !p.layout_only;
        let guide_here: Option<(&[f32], usize)> = guides
            .as_ref()
            .filter(|_| steer)
            .filter(|(g, _)| g[li].len() == limg.len())
            .map(|(g, scale)| (&g[li][..], (scale / (1u64 << li.min(63)) as f32).ceil().max(1.0) as usize));
        if sources.len() == 0 {
            prev = Some(Coarser { w: lw, h: lh, img: limg, field: Vec::new(), cells });
            continue;
        }
        let refine_only = li + 1 < nlev && exts[li] > p.refine_extent;
        let n_targets = cells.len();
        // Source moments are worth keeping when there are few sources for the matches tried.
        let keep_source_moments = lw * lh <= SOURCE_MOMENTS_MAX && sources.len() <= n_targets.saturating_mul(if refine_only { 4 } else { 64 });
        let mut solver = Solver {
            w: lw,
            h: lh,
            ch,
            r: ri,
            img: &mut limg,
            guide: guide_here,
            mimg: None,
            coherence: if steer { GUIDED_COHERENCE } else { 0.0 },
            stamp: steer,
            reach: if p.nearby { (thickness.max(NEARBY_MIN) >> li).max(2 * psz) as i32 } else { 0 },
            px: &lpx,
            valid,
            cells,
            full: Vec::new(),
            tmom: Vec::new(),
            smom: Vec::new(),
            adapt: p.adapt,
            ctl: *ctl,
        };
        solver.refresh_match();
        solver.prepare(keep_source_moments);
        ctl.check()?;
        // NNF init: the coarser level's matches scaled up where they land on a valid source,
        // random otherwise.
        let mut rng = Rng::new(p.seed ^ (li as u64) << 20);
        let (mut nnf, upsampled) = seed_matches(&solver, coarser.as_ref(), &sources, &mut rng, lw, ctl)?;
        // The coarser level has given all it had.
        drop(coarser);
        let mut cost = vec![f32::INFINITY; n_targets];
        // A finer level starts from the coarse matches voted with its own (sharp) pixels.
        if upsampled {
            solver.costs(&nnf, &mut cost)?;
            solver.vote(&nnf, &cost);
        }
        let em = if li + 1 == nlev {
            p.em_iters
        } else if refine_only {
            1
        } else {
            (p.em_iters >> (nlev - 1 - li)).max(p.min_em_iters)
        };
        let pm_iters = if refine_only { 1 } else { p.pm_iters };
        let full_search = (lw.max(lh)) as i32;
        for it in 0..em {
            solver.costs(&nnf, &mut cost)?;
            // Wide random search on the coarsest level and on each level's first iteration,
            // local refinement afterwards.
            let radius = if refine_only {
                2
            } else if li + 1 == nlev || it == 0 {
                full_search
            } else {
                (4 * psz as i32).min(full_search)
            };
            // Nearby: nothing farther is taken, so nothing farther is searched.
            let radius = if solver.reach > 0 { radius.min(solver.reach) } else { radius };
            for pass in 0..pm_iters {
                ctl.check()?;
                solver.patchmatch(&mut nnf, &mut cost, pass + it * pm_iters, p.seed.wrapping_add((li * 1000 + it) as u64), radius);
            }
            ctl.check()?;
            solver.vote(&nnf, &cost);
            ctl.check()?;
        }
        // A vote that saw the cancellation stopped part-way: never hand its pixels on.
        ctl.check()?;
        // Keep the match field for the next level (none after the last).
        let cells = solver.cells;
        drop(cost);
        let field = if li == 0 { Vec::new() } else { nnf };
        // Finer levels cost ~4× the previous one: weight progress by pixel count.
        done_px += lw * lh;
        ctl.progress(done_px as f32 / total_px.max(1) as f32);
        prev = Some(Coarser { w: lw, h: lh, img: limg, field, cells });
    }
    Ok(prev.map(|c| c.img).ok_or(Cow::Owned(Vec::new())))
}

/// The levels of the completion, finest first, with the hole's bounds and extent at each.
struct Pyramid<'a> {
    levels: Vec<Level<'a>>,
    boxes: Vec<common::Bounds>,
    exts: Vec<usize>,
    /// The hole is a thin one: its thickness, not its length, decides how deep to go.
    thin: bool,
    /// The hole's thickness at full size.
    thickness: usize,
}

/// Shrinks the image until the hole is about four patches across (a deeper pyramid decides the
/// structure where the texture has averaged out, and fills flat). A thin hole stops while it is
/// still about two patches thick: through it, not along it, is what counts (#1366).
fn build_pyramid<'a>(
    first: Level<'a>,
    ch: usize,
    hb0: common::Bounds,
    psz: usize,
    ctl: &photocraft_raster::Interrupt,
) -> Result<Pyramid<'a>, photocraft_raster::Cancelled> {
    let thickness = hole_thickness(first.w, hb0, &first.px);
    let mut levels = vec![first];
    let mut boxes = vec![hb0];
    let mut ext = hb0.width().max(hb0.height());
    let thin = thickness.saturating_mul(THIN_RATIO) <= ext;
    if thin {
        ext = thickness;
    }
    let mut exts = vec![ext];
    while let Some(l) = levels.last() {
        ctl.check()?;
        if ext <= if thin { 2 * psz } else { 4 * psz } || l.w / 2 < 3 * psz || l.h / 2 < 3 * psz {
            break;
        }
        let d = downsample(l, ch);
        levels.push(d);
        boxes.push(boxes[boxes.len() - 1].half());
        ext = ext.div_ceil(2);
        exts.push(ext);
    }
    Ok(Pyramid { levels, boxes, exts, thin, thickness })
}

/// A `w × h × ch` guide at each of `levels` pyramid levels, finest first: halved as the pyramid
/// halves the picture (each cell the mean of its 2 × 2 pixels within the grid).
fn guide_pyramid(g: &[f32], w: usize, h: usize, ch: usize, levels: usize) -> Vec<Vec<f32>> {
    let mut out = vec![g.to_vec()];
    let (mut cw, mut chh) = (w, h);
    while out.len() < levels {
        let cur = &out[out.len() - 1];
        let (nw, nh) = (cw.div_ceil(2), chh.div_ceil(2));
        let mut next = vec![0.0f32; nw * nh * ch];
        par::chunks_mut(&mut next, nw * ch, nw * nh >= common::PAR_MIN, |y, row| {
            for x in 0..nw {
                let mut n = 0.0f32;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (sx, sy) = (2 * x + dx, 2 * y + dy);
                    if sx < cw && sy < chh {
                        for c in 0..ch {
                            row[x * ch + c] += cur[(sy * cw + sx) * ch + c];
                        }
                        n += 1.0;
                    }
                }
                for v in &mut row[x * ch..(x + 1) * ch] {
                    *v /= n;
                }
            }
        });
        out.push(next);
        (cw, chh) = (nw, nh);
    }
    out
}

/// The level's own pixels to fill: the caller's, or a copy made in parallel.
fn into_owned_pixels(img: Cow<'_, [f32]>) -> Vec<f32> {
    match img {
        Cow::Owned(v) => v,
        Cow::Borrowed(b) => {
            const CHUNK: usize = 1 << 16;
            let mut v = vec![0.0f32; b.len()];
            par::chunks_mut(&mut v, CHUNK, true, |i, d| d.copy_from_slice(&b[i * CHUNK..i * CHUNK + d.len()]));
            v
        }
    }
}

/// Starts the hole from the coarser level's pixels, scaled up.
fn copy_up_coarser(img: &mut [f32], px: &[Px], w: usize, ch: usize, hb: common::Bounds, c: &Coarser) {
    let (pw, ph) = (c.w, c.h);
    for y in hb.y0..hb.y1 {
        for x in hb.x0..hb.x1 {
            if px[y * w + x] == Px::Hole {
                let (sx, sy) = ((x / 2).min(pw - 1), (y / 2).min(ph - 1));
                let (d, s) = ((y * w + x) * ch, (sy * pw + sx) * ch);
                img[d..d + ch].copy_from_slice(&c.img[s..s + ch]);
            }
        }
    }
}

/// Starts the hole as a membrane over its surroundings, for where nothing can be copied. `dims`
/// is the level's width, height and channel count.
fn fill_hole_with_membrane(img: &mut [f32], px: &[Px], dims: (usize, usize, usize), r: usize, thin: bool) {
    let (w, h, ch) = dims;
    let mut unknown: Vec<bool> = px.iter().map(|k| matches!(k, Px::Hole | Px::Void)).collect();
    // Across a thin hole the membrane starts a patch radius out, past the
    // anti-aliased fringe or halo that a selection drawn close to the object leaves.
    if thin {
        let hole: Vec<bool> = px.iter().map(|k| *k == Px::Hole).collect();
        for (u, d) in unknown.iter_mut().zip(crate::selection::edt(&hole, w, h)) {
            *u |= d <= r as f32;
        }
    }
    let m = membrane_fill(w, h, ch, img, &unknown);
    for (i, k) in px.iter().enumerate() {
        if *k == Px::Hole {
            img[i * ch..(i + 1) * ch].copy_from_slice(&m[i * ch..(i + 1) * ch]);
        }
    }
}

/// The coarser level's match for `(x, y)`, scaled up, where it lands on a valid source.
fn upscaled_match(solver: &Solver, c: &Coarser, x: usize, y: usize) -> Option<(i32, i32)> {
    let (cx, cy) = ((x / 2).min(c.w - 1), (y / 2).min(c.h - 1));
    let (sx, sy) = c.field_at(cx, cy);
    if sx == i32::MIN {
        return None;
    }
    // The same displacement, twice as long.
    let m = (x as i32 + 2 * (sx - cx as i32), y as i32 + 2 * (sy - cy as i32));
    solver.is_valid(m.0, m.1).then_some(m)
}

/// NNF init: the coarser level's matches scaled up where they land on a valid source, random
/// otherwise. Also says whether any came from the coarser level.
fn seed_matches(
    solver: &Solver,
    coarser: Option<&Coarser>,
    sources: &Sources,
    rng: &mut Rng,
    lw: usize,
    ctl: &photocraft_raster::Interrupt,
) -> Result<(Vec<(i32, i32)>, bool), photocraft_raster::Cancelled> {
    let cells = &solver.cells;
    let mut nnf = vec![(0i32, 0i32); cells.len()];
    let mut upsampled = false;
    for y in cells.tbox.y0..cells.tbox.y1 {
        if y % 32 == 0 {
            ctl.check()?;
        }
        for s in cells.row(y) {
            for x in s.x0..s.x1 {
                let cand = coarser.filter(|c| !c.field.is_empty()).and_then(|c| upscaled_match(solver, c, x, y));
                upsampled |= cand.is_some();
                nnf[s.at(x)] = cand.unwrap_or_else(|| sources.nth((rng.next() % sources.len() as u64) as usize, &solver.valid, lw));
            }
        }
    }
    Ok((nnf, upsampled))
}

#[cfg(test)]
mod tests;
