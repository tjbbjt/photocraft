//! Seam cutting for a stamped fill: where two copied pieces of the picture meet, the seam is
//! moved to where they agree.
//!
//! A stamped fill takes each pixel from one copied patch, so neighbouring pixels from different
//! copies meet along the patch grid: straight seams across clouds and ridges. Here each pixel's
//! copy is a label (the shift of the picture it copies), and the labels are relaxed so that two
//! neighbours with different copies cost what the copies disagree there (in the style of
//! graph-cut textures, Kwatra et al. 2003): a seam then runs where both copies show nearly the
//! same, along a cloud's edge or through even sky, not along the grid. The relaxation is iterated
//! conditional modes on a checkerboard (each pixel takes its own or a neighbour's copy, whichever
//! costs least against its four neighbours), a few sweeps: it moves seams, it doesn't invent
//! copies.

use super::{Px, par};

/// Sweeps of the relaxation (each updates both colours of the checkerboard).
const SWEEPS: usize = 8;
/// Where a copy has no pixel (off the picture, or in the hole), what a disagreement costs: about
/// a full step in one channel, so such a copy is only taken where nothing else fits.
const MISSING: f32 = 1.0;
/// Pixels from which the rows are relaxed in parallel.
const PAR_MIN: usize = 1 << 16;

/// The most channels a feathered pixel has (CMYK with alpha is 5).
const MAX_CH: usize = 8;

/// The picture as the copies see it.
#[derive(Clone, Copy)]
struct View<'a> {
    img: &'a [f32],
    px: &'a [Px],
    w: usize,
    h: usize,
    ch: usize,
}

impl<'a> View<'a> {
    /// The picture's pixel at `(x, y)` moved by `s`, when it is one a copy can show (known, not
    /// hole).
    #[inline]
    fn at(&self, x: usize, y: usize, s: (i32, i32)) -> Option<&'a [f32]> {
        let (sx, sy) = (x as i64 + i64::from(s.0), y as i64 + i64::from(s.1));
        if sx < 0 || sy < 0 || sx >= self.w as i64 || sy >= self.h as i64 {
            return None;
        }
        let j = sy as usize * self.w + sx as usize;
        matches!(self.px[j], Px::Source | Px::Known).then(|| &self.img[j * self.ch..(j + 1) * self.ch])
    }
}

#[inline]
fn diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum()
}

/// Relaxes the copies of the `active` pixels (`labels`: the shift each copies) and writes each
/// active pixel from its copy. Other pixels (the picture, and fill pixels near it, which a
/// stamped fill blends) are fixed: an active pixel next to one pays what its copy shows there
/// against what is there.
pub(super) fn cut(img: &mut [f32], px: &[Px], w: usize, h: usize, ch: usize, active: &[bool], labels: &mut [(i32, i32)]) {
    let par = w * h >= PAR_MIN;
    for _ in 0..SWEEPS {
        let mut moved = 0usize;
        for parity in 0..2 {
            let (src, lab) = (&*img, &*labels);
            let view = View { img: src, px, w, h, ch };
            // What one neighbour costs pixel (x, y) with copy `a`.
            let edge = |x: usize, y: usize, a: (i32, i32), nx: usize, ny: usize| -> f32 {
                let q = ny * w + nx;
                if px[q] == Px::Void {
                    return 0.0;
                }
                if active[q] {
                    let b = lab[q];
                    if a == b {
                        return 0.0;
                    }
                    let at_p = match (view.at(x, y, a), view.at(x, y, b)) {
                        (Some(u), Some(v)) => diff(u, v),
                        _ => MISSING,
                    };
                    let at_q = match (view.at(nx, ny, a), view.at(nx, ny, b)) {
                        (Some(u), Some(v)) => diff(u, v),
                        _ => MISSING,
                    };
                    return at_p + at_q;
                }
                // A fixed neighbour: what copy `a` shows there against what is there, or failing
                // that, this pixel's copy against it.
                let there = &src[q * ch..(q + 1) * ch];
                match view.at(nx, ny, a).or_else(|| view.at(x, y, a)) {
                    Some(u) => 2.0 * diff(u, there),
                    None => 2.0 * MISSING,
                }
            };
            let rows = par::map(h, par, |y| {
                let mut out = Vec::new();
                for x in ((y + parity) % 2..w).step_by(2) {
                    let p = y * w + x;
                    if !active[p] {
                        continue;
                    }
                    let mut nbrs = [(0usize, 0usize); 4];
                    let mut n = 0;
                    if x > 0 {
                        nbrs[n] = (x - 1, y);
                        n += 1;
                    }
                    if x + 1 < w {
                        nbrs[n] = (x + 1, y);
                        n += 1;
                    }
                    if y > 0 {
                        nbrs[n] = (x, y - 1);
                        n += 1;
                    }
                    if y + 1 < h {
                        nbrs[n] = (x, y + 1);
                        n += 1;
                    }
                    let cost = |a: (i32, i32)| -> f32 { nbrs[..n].iter().map(|&(nx, ny)| edge(x, y, a, nx, ny)).sum() };
                    let cur = lab[p];
                    let mut best = (cur, if view.at(x, y, cur).is_some() { cost(cur) } else { f32::INFINITY });
                    let neighbours = nbrs[..n].iter().filter(|&&(nx, ny)| active[ny * w + nx]).map(|&(nx, ny)| lab[ny * w + nx]);
                    for cand in neighbours {
                        if cand == best.0 || view.at(x, y, cand).is_none() {
                            continue;
                        }
                        let c = cost(cand);
                        if c < best.1 {
                            best = (cand, c);
                        }
                    }
                    if best.0 != cur {
                        out.push((p, best.0));
                    }
                }
                out
            });
            for (p, l) in rows.into_iter().flatten() {
                labels[p] = l;
                moved += 1;
            }
        }
        if moved == 0 {
            break;
        }
    }
    for (p, _) in active.iter().enumerate().filter(|(_, a)| **a) {
        let (x, y) = (p % w, p / w);
        let view = View { img: &*img, px, w, h, ch };
        if let Some(v) = view.at(x, y, labels[p]).map(<[f32]>::to_vec) {
            img[p * ch..(p + 1) * ch].copy_from_slice(&v);
        }
    }
}

/// Feathers the seams [`cut`] left: each active pixel within `f` pixels of a seam (an active
/// neighbour from another copy) becomes the mix of the copies around it, each weighted by how
/// many of its pixels are near and how near (a tent falling to nothing `f + 1` pixels away), so
/// two copies cross-fade over about `2f` pixels instead of meeting at a hard edge. Pixels
/// farther from any seam keep their copy as it is.
pub(super) fn feather(img: &mut [f32], px: &[Px], dims: (usize, usize, usize), active: &[bool], labels: &[(i32, i32)], f: usize) {
    let (w, h, ch) = dims;
    if f == 0 || ch > MAX_CH {
        return;
    }
    let seam: Vec<bool> = (0..w * h)
        .map(|p| {
            let (x, y) = (p % w, p / w);
            let differs = |q: usize| active[q] && labels[q] != labels[p];
            active[p] && ((x > 0 && differs(p - 1)) || (x + 1 < w && differs(p + 1)) || (y > 0 && differs(p - w)) || (y + 1 < h && differs(p + w)))
        })
        .collect();
    let near = dilate_box(&seam, w, h, f);
    let par = w * h >= PAR_MIN;
    let view = View { img: &*img, px, w, h, ch };
    // The neighbourhood is sampled every `step` pixels: a wide feather reads about the same
    // 9 × 9 samples as a narrow one, not (2f + 1)², and gives the same blend (the copies change
    // over whole patches, not single pixels). At f = 8 that was 289 reads per pixel near a seam.
    let step = f.div_ceil(4).max(1);
    let rows = par::map(h, par, |y| {
        let mut out: Vec<(usize, [f32; MAX_CH])> = Vec::new();
        // Each copy near: its weight, and what [`fuse`] added to it.
        let mut copies: Vec<((i32, i32), f32, [f32; MAX_CH])> = Vec::new();
        for x in 0..w {
            let p = y * w + x;
            if !active[p] || !near[p] {
                continue;
            }
            copies.clear();
            let (ylo, xlo) = (y - (y.min(f) / step) * step, x - (x.min(f) / step) * step);
            for qy in (ylo..(y + f + 1).min(h)).step_by(step) {
                for qx in (xlo..(x + f + 1).min(w)).step_by(step) {
                    let q = qy * w + qx;
                    if !active[q] {
                        continue;
                    }
                    let d = qx.abs_diff(x).max(qy.abs_diff(y));
                    let wt = (f + 1 - d) as f32;
                    let i = match copies.iter().position(|c| c.0 == labels[q]) {
                        Some(i) => i,
                        None => {
                            // What [`fuse`] added to this copy, read where it was first met (it
                            // varies slowly within a copy).
                            let mut add = [0.0f32; MAX_CH];
                            if let Some(own) = view.at(qx, qy, labels[q]) {
                                add.iter_mut().zip(&view.img[q * ch..(q + 1) * ch]).zip(own).for_each(|((a, now), o)| *a = now - o);
                            }
                            copies.push((labels[q], 0.0, add));
                            copies.len() - 1
                        }
                    };
                    copies[i].1 += wt;
                }
            }
            let mut acc = [0.0f32; MAX_CH];
            let mut sum = 0.0f32;
            for (l, wt, cs) in &copies {
                if let Some(v) = view.at(x, y, *l) {
                    for ((a, v), s) in acc.iter_mut().zip(v).zip(cs) {
                        *a += (v + s) * wt;
                    }
                    sum += wt;
                }
            }
            if sum > 0.0 {
                acc.iter_mut().for_each(|a| *a /= sum);
                out.push((p, acc));
            }
        }
        out
    });
    for (p, v) in rows.into_iter().flatten() {
        img[p * ch..(p + 1) * ch].copy_from_slice(&v[..ch]);
    }
}

/// `mask` grown by `f` pixels in each direction (a `(2f + 1)²` box).
fn dilate_box(mask: &[bool], w: usize, h: usize, f: usize) -> Vec<bool> {
    let grow = |get: &dyn Fn(usize) -> bool, len: usize, out: &mut dyn FnMut(usize, bool)| {
        // The distance back to the last set pixel, then forward to the next.
        let mut last: Option<usize> = None;
        let mut near = vec![false; len];
        for (i, n) in near.iter_mut().enumerate() {
            if get(i) {
                last = Some(i);
            }
            *n = last.is_some_and(|l| i - l <= f);
        }
        last = None;
        for i in (0..len).rev() {
            if get(i) {
                last = Some(i);
            }
            out(i, near[i] || last.is_some_and(|l| l - i <= f));
        }
    };
    let mut rows = vec![false; w * h];
    for y in 0..h {
        let line = &mask[y * w..(y + 1) * w];
        grow(&|i| line[i], w, &mut |c, v| rows[y * w + c] = v);
    }
    let mut out = vec![false; w * h];
    for x in 0..w {
        grow(&|i| rows[i * w + x], h, &mut |c, v| out[c * w + x] = v);
    }
    out
}

/// Brings the cut fill to the picture where they meet, so no edge is left where the hole was:
/// next to the picture, each fill pixel's copy is compared with the picture beside it (what the
/// copy shows there against what is there), and that difference is added to the fill, fading
/// to nothing `fade` pixels into the hole. Like seamless cloning, but local: a copy that is a
/// little brighter or a little cloudier than the picture it continues blends into it instead of
/// meeting it at a line, and farther in the copy keeps its own texture.
pub(super) fn meet_picture(img: &mut [f32], px: &[Px], dims: (usize, usize, usize), active: &[bool], labels: &[(i32, i32)], fade: usize) {
    let (w, h, ch) = dims;
    if fade == 0 {
        return;
    }
    let picture = |q: usize| matches!(px[q], Px::Source | Px::Known);
    let nbrs = |p: usize| {
        let (x, y) = (p % w, p / w);
        [(x > 0).then(|| p - 1), (x + 1 < w).then(|| p + 1), (y > 0).then(|| p - w), (y + 1 < h).then(|| p + w)].into_iter().flatten()
    };
    // The first layer: fill pixels beside the picture, each with the picture minus its copy there.
    let mut corr = vec![0.0f32; w * h * ch];
    let mut layer: Vec<usize> = Vec::new();
    let mut depth = vec![0usize; w * h];
    {
        let view = View { img: &*img, px, w, h, ch };
        for p in (0..w * h).filter(|p| active[*p]) {
            let (mut acc, mut n) = (vec![0.0f32; ch], 0usize);
            for q in nbrs(p).filter(|q| picture(*q)) {
                let (qx, qy) = (q % w, q / w);
                let there = &img[q * ch..(q + 1) * ch];
                let copy = view.at(qx, qy, labels[p]).unwrap_or(&img[p * ch..(p + 1) * ch]);
                for ((a, t), c) in acc.iter_mut().zip(there).zip(copy) {
                    *a += t - c;
                }
                n += 1;
            }
            if n > 0 {
                for (o, a) in corr[p * ch..(p + 1) * ch].iter_mut().zip(&acc) {
                    *o = a / n as f32;
                }
                depth[p] = 1;
                layer.push(p);
            }
        }
    }
    // Inwards, layer by layer: the mean of the layer before, shrinking so it is gone at `fade`.
    let mut all = layer.clone();
    for k in 2..=fade {
        let keep = (fade + 1 - k) as f32 / (fade + 2 - k) as f32;
        let mut next = Vec::new();
        for &p in &layer {
            for q in nbrs(p) {
                if active[q] && depth[q] == 0 {
                    depth[q] = k;
                    next.push(q);
                }
            }
        }
        for &q in &next {
            let (mut acc, mut n) = (vec![0.0f32; ch], 0usize);
            for r in nbrs(q).filter(|r| depth[*r] == k - 1) {
                for (a, c) in acc.iter_mut().zip(&corr[r * ch..(r + 1) * ch]) {
                    *a += c;
                }
                n += 1;
            }
            for (o, a) in corr[q * ch..(q + 1) * ch].iter_mut().zip(&acc) {
                *o = keep * a / n.max(1) as f32;
            }
        }
        all.extend_from_slice(&next);
        layer = next;
        if layer.is_empty() {
            break;
        }
    }
    for p in all {
        for (o, c) in img[p * ch..(p + 1) * ch].iter_mut().zip(&corr[p * ch..(p + 1) * ch]) {
            *o += c;
        }
    }
}

/// Grid cells of the [`fuse`] solve: left out (no equation, a free edge), solved, or held at 0.
const OUT: u8 = 0;
const SOLVE: u8 = 1;
const HELD: u8 = 2;
/// V-cycles per channel of the [`fuse`] solve, and smoothing sweeps before and after each
/// coarser solve.
const CYCLES: usize = 12;
/// The largest residual (in 0..1 of a channel, summed over a pixel's links) a solve stops at.
const TOL: f32 = 1.0 / 400.0;
const SMOOTH: usize = 3;
/// The most [`fuse`] changes one step between neighbours (in 0..1 of a channel): it evens out
/// a copy's slight shading, while two copies that differ more where they meet (a cloud's edge
/// against sky) keep most of their step, instead of lifting or sinking a whole region to hide it.
const FUSE_LINK: f32 = 8.0 / 255.0;
/// At or below this many unknowns a level is solved by sweeps alone.
const COARSEST: usize = 2048;

/// One level of the [`fuse`] solve: which cells are solved, each solved cell's diagonal, and the
/// weights of the links to its right and lower neighbours. The first level is the pixel grid (links
/// of 1 between solved pixels, a held neighbour counted in the diagonal only); each level above
/// joins 2×2 cells into one and adds up their equations (a Galerkin coarsening by aggregation), so
/// a coarse correction never makes the error worse.
struct Level {
    w: usize,
    h: usize,
    solve: Vec<bool>,
    diag: Vec<f32>,
    right: Vec<f32>,
    down: Vec<f32>,
    unknowns: usize,
    /// Per row, the runs of solved cells (start, end): the sweeps skip the rest, which on a
    /// panorama's border fill is most of the box.
    spans: Vec<Vec<(usize, usize)>>,
}

/// Per row of `solve` (`w` × `h`), its runs of set cells as (start, end).
fn spans_of(solve: &[bool], w: usize, h: usize) -> Vec<Vec<(usize, usize)>> {
    (0..h)
        .map(|y| {
            let row = &solve[y * w..(y + 1) * w];
            let mut runs = Vec::new();
            let mut x = 0;
            while x < w {
                if row[x] {
                    let s = x;
                    while x < w && row[x] {
                        x += 1;
                    }
                    runs.push((s, x));
                } else {
                    x += 1;
                }
            }
            runs
        })
        .collect()
}

impl Level {
    fn pixels(w: usize, h: usize, kind: &[u8]) -> Level {
        let solve: Vec<bool> = kind.iter().map(|k| *k == SOLVE).collect();
        let (mut diag, mut right, mut down) = (vec![0.0f32; w * h], vec![0.0f32; w * h], vec![0.0f32; w * h]);
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                if !solve[p] {
                    continue;
                }
                let n = [(x > 0).then(|| p - 1), (x + 1 < w).then(|| p + 1), (y > 0).then(|| p - w), (y + 1 < h).then(|| p + w)]
                    .into_iter()
                    .flatten()
                    .filter(|q| kind[*q] != OUT)
                    .count();
                // A little pull to 0, so a region that touches nothing held still has one answer.
                diag[p] = n as f32 + 1e-3;
                if x + 1 < w && solve[p + 1] {
                    right[p] = 1.0;
                }
                if y + 1 < h && solve[p + w] {
                    down[p] = 1.0;
                }
            }
        }
        let unknowns = solve.iter().filter(|s| **s).count();
        let spans = spans_of(&solve, w, h);
        Level { w, h, solve, diag, right, down, unknowns, spans }
    }

    fn coarser(&self) -> Level {
        let (w, h) = (self.w.div_ceil(2), self.h.div_ceil(2));
        let (mut diag, mut right, mut down) = (vec![0.0f32; w * h], vec![0.0f32; w * h], vec![0.0f32; w * h]);
        let mut solve = vec![false; w * h];
        for y in 0..self.h {
            for x in 0..self.w {
                let p = y * self.w + x;
                if !self.solve[p] {
                    continue;
                }
                let c = (y / 2) * w + x / 2;
                solve[c] = true;
                diag[c] += self.diag[p];
                // A link inside the block cancels out of its sum; one out of it links two blocks.
                if self.right[p] != 0.0 {
                    if x % 2 == 0 {
                        diag[c] -= 2.0 * self.right[p];
                    } else {
                        right[c] += self.right[p];
                    }
                }
                if self.down[p] != 0.0 {
                    if y % 2 == 0 {
                        diag[c] -= 2.0 * self.down[p];
                    } else {
                        down[c] += self.down[p];
                    }
                }
            }
        }
        let unknowns = solve.iter().filter(|s| **s).count();
        let spans = spans_of(&solve, w, h);
        Level { w, h, solve, diag, right, down, unknowns, spans }
    }

    fn nbr_sum(&self, x: usize, y: usize, v: &[f32]) -> f32 {
        let (w, p) = (self.w, y * self.w + x);
        let mut s = 0.0;
        if x > 0 {
            s += self.right[p - 1] * v[p - 1];
        }
        if x + 1 < w {
            s += self.right[p] * v[p + 1];
        }
        if y > 0 {
            s += self.down[p - w] * v[p - w];
        }
        if y + 1 < self.h {
            s += self.down[p] * v[p + w];
        }
        s
    }

    /// The largest residual of `diag·v − Σ links·neighbours = b`.
    fn residual(&self, v: &[f32], b: &[f32]) -> f32 {
        let par = self.w * self.h >= PAR_MIN;
        par::map(self.h, par, |y| {
            self.spans[y]
                .iter()
                .flat_map(|&(a, b)| a..b)
                .map(|x| {
                    let p = y * self.w + x;
                    (b[p] - (self.diag[p] * v[p] - self.nbr_sum(x, y, v))).abs()
                })
                .fold(0.0f32, f32::max)
        })
        .into_iter()
        .fold(0.0f32, f32::max)
    }

    /// Red-black Gauss–Seidel sweeps of `diag·v − Σ links·neighbours = b`, each half written
    /// into the other of two buffers (so rows run in parallel without sharing what they write).
    /// Only solved cells are visited: `v` is 0 everywhere else, in both buffers.
    fn smooth(&self, v: &mut [f32], b: &[f32], sweeps: usize) {
        let par = self.w * self.h >= PAR_MIN;
        let mut other = vec![0.0f32; v.len()];
        let half = |src: &[f32], dst: &mut [f32], parity: usize| {
            par::chunks_mut(dst, self.w, par, |y, row| {
                for &(a, e) in &self.spans[y] {
                    for x in a..e {
                        let p = y * self.w + x;
                        row[x] = if (x + y) % 2 == parity { (b[p] + self.nbr_sum(x, y, src)) / self.diag[p] } else { src[p] };
                    }
                }
            });
        };
        for _ in 0..sweeps {
            half(v, &mut other, 0);
            half(&other, v, 1);
        }
    }
}

/// One multigrid V-cycle at `levels[0]`.
fn v_cycle(levels: &[Level], v: &mut [f32], b: &[f32]) {
    let l = &levels[0];
    if levels.len() == 1 || l.unknowns <= COARSEST {
        l.smooth(v, b, if l.unknowns <= COARSEST { 4 * l.w.max(l.h) } else { 64 });
        return;
    }
    l.smooth(v, b, SMOOTH);
    let c = &levels[1];
    let par = l.w * l.h >= PAR_MIN;
    // The residual, summed into the coarser cells (two fine rows to each coarse one).
    let cb: Vec<f32> = par::map(c.h, par, |cy| {
        let mut row = vec![0.0f32; c.w];
        for y in (2 * cy)..(2 * cy + 2).min(l.h) {
            for x in l.spans[y].iter().flat_map(|&(a, e)| a..e) {
                let p = y * l.w + x;
                row[x / 2] += b[p] - (l.diag[p] * v[p] - l.nbr_sum(x, y, v));
            }
        }
        row
    })
    .concat();
    let mut cv = vec![0.0f32; c.w * c.h];
    v_cycle(&levels[1..], &mut cv, &cb);
    par::chunks_mut(v, l.w, par, |y, row| {
        for x in l.spans[y].iter().flat_map(|&(a, e)| a..e) {
            row[x] += cv[(y / 2) * c.w + x / 2];
        }
    });
    l.smooth(v, b, SMOOTH);
}

/// Fuses the cut fill in the gradient domain, so no step is left at any size: a correction is
/// added to the fill that keeps every copy's own texture (its steps between neighbours) but, where
/// two copies meet, takes the step between them to be what both copies show there, and where the
/// fill meets the picture, what its copy shows there (Pérez et al. 2003; Agarwala et al. 2004's
/// gradient-domain fusion). A copy a little brighter, bluer or cloudier than its neighbours is
/// evened out over its whole size, however big, instead of over a fixed few pixels.
pub(super) fn fuse(img: &mut [f32], px: &[Px], dims: (usize, usize, usize), active: &[bool], labels: &[(i32, i32)]) {
    let (w, h, ch) = dims;
    let picture = |q: usize| matches!(px[q], Px::Source | Px::Known);
    // The active pixels' box, one pixel wider for the picture around them.
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for p in (0..w * h).filter(|p| active[*p]) {
        let (x, y) = (p % w, p / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    if x0 > x1 {
        return;
    }
    let (x0, y0, x1, y1) = (x0.saturating_sub(1), y0.saturating_sub(1), (x1 + 1).min(w - 1), (y1 + 1).min(h - 1));
    let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
    let at = |bx: usize, by: usize| (y0 + by) * w + x0 + bx;
    let kind: Vec<u8> = (0..bw * bh)
        .map(|i| {
            let p = at(i % bw, i / bw);
            if active[p] {
                SOLVE
            } else if picture(p) {
                HELD
            } else {
                OUT
            }
        })
        .collect();
    let mut levels = vec![Level::pixels(bw, bh, &kind)];
    while levels.last().is_some_and(|l| l.unknowns > COARSEST && l.w > 2 && l.h > 2) {
        let next = levels.last().map(Level::coarser);
        levels.extend(next);
    }
    let par = bw * bh >= PAR_MIN;
    // The colour channels: a last alpha channel (RGBA, gray + alpha) is left as the copies have it.
    let colours = if ch == 2 || ch == 4 { ch - 1 } else { ch };
    for c in 0..colours {
        // Each channel reads only its own values, and the picture (never written).
        let view = View { img: &*img, px, w, h, ch };
        let img_r = &*img;
        // Each solved cell's right-hand side: the steps to its neighbours the fill should take,
        // less the steps it takes.
        let rows = par::map(bh, par, |by| {
            let mut row = vec![0.0f32; bw];
            for (bx, out) in row.iter_mut().enumerate() {
                let p = at(bx, by);
                if !active[p] {
                    continue;
                }
                let (x, y) = (p % w, p / w);
                let f = img_r[p * ch + c];
                let nbrs = [(x > 0).then(|| (x - 1, y)), (x + 1 < w).then(|| (x + 1, y)), (y > 0).then(|| (x, y - 1)), (y + 1 < h).then(|| (x, y + 1))];
                for (nx, ny) in nbrs.into_iter().flatten() {
                    let q = ny * w + nx;
                    let g = img_r[q * ch + c];
                    let step = |s: (i32, i32)| Some(view.at(x, y, s)?[c] - view.at(nx, ny, s)?[c]);
                    let want = if active[q] {
                        if labels[q] == labels[p] {
                            continue;
                        }
                        match (step(labels[p]), step(labels[q])) {
                            (Some(a), Some(b)) => 0.5 * (a + b),
                            (Some(a), None) | (None, Some(a)) => a,
                            (None, None) => 0.0,
                        }
                    } else if picture(q) {
                        step(labels[p]).unwrap_or(0.0)
                    } else {
                        continue;
                    };
                    *out += (want - (f - g)).clamp(-FUSE_LINK, FUSE_LINK);
                }
            }
            row
        });
        let b: Vec<f32> = rows.into_iter().flatten().collect();
        let mut v = vec![0.0f32; bw * bh];
        for _ in 0..CYCLES {
            v_cycle(&levels, &mut v, &b);
            if levels[0].residual(&v, &b) < TOL {
                break;
            }
        }
        drop(b);
        for (i, x) in v.iter().enumerate() {
            let p = at(i % bw, i / bw);
            if active[p] {
                img[p * ch + c] += x;
            }
        }
    }
}

/// `v` (`ch` channels) averaged over a `(2r + 1)²` box, counting only pixels where `wt` is set
/// (where none is, 0).
fn box_mean(v: &[f32], wt: &[bool], w: usize, h: usize, ch: usize, r: usize) -> Vec<f32> {
    // Running sums along rows, then along columns; the last channel slot holds the count.
    let n = ch + 1;
    let mut rows = vec![0.0f32; w * h * n];
    for y in 0..h {
        let mut acc = vec![0.0f64; n];
        let add = |acc: &mut [f64], x: usize, s: f64| {
            let p = y * w + x;
            if wt[p] {
                for c in 0..ch {
                    acc[c] += s * f64::from(v[p * ch + c]);
                }
                acc[ch] += s;
            }
        };
        for x in 0..r.min(w) {
            add(&mut acc, x, 1.0);
        }
        for x in 0..w {
            if x + r < w {
                add(&mut acc, x + r, 1.0);
            }
            if x > r {
                add(&mut acc, x - r - 1, -1.0);
            }
            for c in 0..n {
                rows[(y * w + x) * n + c] = acc[c] as f32;
            }
        }
    }
    let mut out = vec![0.0f32; w * h * ch];
    for x in 0..w {
        let mut acc = vec![0.0f64; n];
        let add = |acc: &mut [f64], y: usize, s: f64| {
            for c in 0..n {
                acc[c] += s * f64::from(rows[(y * w + x) * n + c]);
            }
        };
        for y in 0..r.min(h) {
            add(&mut acc, y, 1.0);
        }
        for y in 0..h {
            if y + r < h {
                add(&mut acc, y + r, 1.0);
            }
            if y > r {
                add(&mut acc, y - r - 1, -1.0);
            }
            if acc[ch] > 0.5 {
                for c in 0..ch {
                    out[(y * w + x) * ch + c] = (acc[c] / acc[ch]) as f32;
                }
            }
        }
    }
    out
}

/// Distance (chamfer 3-4, in pixels) from each pixel to the nearest pixel of `from`.
fn chamfer(from: &[bool], w: usize, h: usize) -> Vec<f32> {
    let mut d = vec![f32::INFINITY; w * h];
    for (p, f) in from.iter().enumerate() {
        if *f {
            d[p] = 0.0;
        }
    }
    let (a, b) = (1.0f32, 1.4f32);
    for y in 0..h {
        for x in 0..w {
            let p = y * w + x;
            let mut v = d[p];
            if x > 0 {
                v = v.min(d[p - 1] + a);
            }
            if y > 0 {
                v = v.min(d[p - w] + a);
                if x > 0 {
                    v = v.min(d[p - w - 1] + b);
                }
                if x + 1 < w {
                    v = v.min(d[p - w + 1] + b);
                }
            }
            d[p] = v;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let p = y * w + x;
            let mut v = d[p];
            if x + 1 < w {
                v = v.min(d[p + 1] + a);
            }
            if y + 1 < h {
                v = v.min(d[p + w] + a);
                if x + 1 < w {
                    v = v.min(d[p + w + 1] + b);
                }
                if x > 0 {
                    v = v.min(d[p + w - 1] + b);
                }
            }
            d[p] = v;
        }
    }
    d
}

/// Relights in one of two ways, from the lighting (brightness averaged over `radius` px).
///
/// With no `active` pixels (before a fill): the picture's even areas (sky, haze: not texture or
/// a shadow's edge) within `band` px of the hole take the lighting the picture has past the band,
/// carried smoothly across it, fully at the hole and less and less to nothing `band` px out, so
/// the fill doesn't copy a panorama's darker stitched edge or a lens's vignetting into the hole.
/// The caller puts those pixels back afterwards (the picture is never changed): it returns them
/// as they were.
///
/// With `active` pixels (after a fill): the fill takes the lighting of the picture all round it,
/// carried smoothly into the hole, where the picture around it is even.
///
/// Each pull is at most about `cap`. Lighting is smooth, so it is worked out on a grid of blocks
/// about half the radius across and spread back over the pixels: cheap at any size.
pub(super) fn relight(img: &mut [f32], px: &[Px], dims: (usize, usize, usize), active: &[bool], radius: usize, band: usize, cap: f32) -> Relit {
    let (w, h, ch) = dims;
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for p in (0..w * h).filter(|p| px[*p] == Px::Hole) {
        let (x, y) = (p % w, p / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    if x0 > x1 || cap <= 0.0 || ch == 0 {
        return Relit::default();
    }
    // The box around the hole, wide enough for the band, the picture past it and the blur.
    let m = band + 2 * radius + 2;
    let (x0, y0, x1, y1) = (x0.saturating_sub(m), y0.saturating_sub(m), (x1 + m).min(w - 1), (y1 + m).min(h - 1));
    let s = (radius / 2).max(1);
    let (bw, bh) = ((x1 - x0 + 1).div_ceil(s), (y1 - y0 + 1).div_ceil(s));
    let n = bw * bh;
    // Per block: the picture's sum, sum of squares and count; the fill's (with the picture) sum
    // and count; whether any of it is hole. Each row of blocks on its own thread.
    struct Sums {
        psum: Vec<f64>,
        psq: Vec<f64>,
        pcount: Vec<u32>,
        asum: Vec<f64>,
        acount: Vec<u32>,
        holes: Vec<bool>,
    }
    let par = (x1 - x0 + 1) * (y1 - y0 + 1) >= PAR_MIN;
    let rows: Vec<Sums> = par::map(bh, par, |by| {
        let mut r =
            Sums { psum: vec![0.0; bw * ch], psq: vec![0.0; bw], pcount: vec![0; bw], asum: vec![0.0; bw * ch], acount: vec![0; bw], holes: vec![false; bw] };
        for y in (y0 + by * s)..=(y0 + by * s + s - 1).min(y1) {
            for x in x0..=x1 {
                let (p, b) = (y * w + x, (x - x0) / s);
                let v = &img[p * ch..(p + 1) * ch];
                let pic = matches!(px[p], Px::Source | Px::Known);
                if px[p] == Px::Hole {
                    r.holes[b] = true;
                }
                if pic {
                    for c in 0..ch {
                        r.psum[b * ch + c] += f64::from(v[c]);
                        r.psq[b] += f64::from(v[c]) * f64::from(v[c]);
                    }
                    r.pcount[b] += 1;
                }
                if pic || active[p] {
                    for c in 0..ch {
                        r.asum[b * ch + c] += f64::from(v[c]);
                    }
                    r.acount[b] += 1;
                }
            }
        }
        r
    });
    let (mut psum, mut psq, mut pcount) = (Vec::with_capacity(n * ch), Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut asum, mut acount, mut holes) = (Vec::with_capacity(n * ch), Vec::with_capacity(n), Vec::with_capacity(n));
    for r in rows {
        psum.extend(r.psum);
        psq.extend(r.psq);
        pcount.extend(r.pcount);
        asum.extend(r.asum);
        acount.extend(r.acount);
        holes.extend(r.holes);
    }
    let mean = |sum: &[f64], count: &[u32]| -> Vec<f32> {
        (0..n * ch).map(|i| if count[i / ch] > 0 { (sum[i] / f64::from(count[i / ch])) as f32 } else { 0.0 }).collect()
    };
    let (pmean, amean) = (mean(&psum, &pcount), mean(&asum, &acount));
    let has_pic: Vec<bool> = pcount.iter().map(|c| *c > 0).collect();
    let has_all: Vec<bool> = acount.iter().map(|c| *c > 0).collect();
    // How even each block of picture is: its spread about its own mean, per channel.
    let spread: Vec<f32> = (0..n)
        .map(|b| {
            if pcount[b] == 0 {
                return f32::INFINITY;
            }
            let k = f64::from(pcount[b]) * ch as f64;
            let mu2: f64 = (0..ch).map(|c| (psum[b * ch + c] / f64::from(pcount[b])).powi(2)).sum::<f64>() / ch as f64;
            ((psq[b] / k - mu2).max(0.0).sqrt()) as f32
        })
        .collect();
    let rl = radius.div_ceil(s).max(1);
    let pic_light = box_mean(&pmean, &has_pic, bw, bh, ch, rl);
    let fill_light = box_mean(&amean, &has_all, bw, bh, ch, rl);
    let dist: Vec<f32> = chamfer(&holes, bw, bh).iter().map(|d| d * s as f32).collect();
    // The lighting carried in from the band's far side.
    // A fill takes the lighting of the picture all round it (so it meets the picture it sits
    // in); the band, before a fill, the lighting from past it.
    let fill = acount.iter().zip(&pcount).any(|(a, p)| a > p);
    let far = if fill { 0.0 } else { band as f32 };
    let unknown: Vec<bool> = (0..n).map(|b| !(has_pic[b] && dist[b] >= far)).collect();
    let carried = crate::poisson::membrane_fill(bw, bh, ch, &pic_light, &unknown);
    // How even the picture is, block by block, and carried into the hole: the fill is relit
    // only where the picture around it is even too (a hole across a shadow's edge or in a
    // textured area keeps the lighting its copies bring).
    // (Its spread within each block, and of the blocks about their lighting nearby.)
    let dev: Vec<f32> = (0..n).map(|b| (0..ch).map(|c| (pmean[b * ch + c] - pic_light[b * ch + c]).powi(2)).sum::<f32>() / ch as f32).collect();
    let dev = box_mean(&dev, &has_pic, bw, bh, 1, rl);
    let even: Vec<f32> = (0..n).map(|b| (1.0 - ((spread[b].powi(2) + dev[b]).sqrt() - EVEN) / EVEN).clamp(0.0, 1.0)).collect();
    let no_pic: Vec<bool> = has_pic.iter().map(|p| !p).collect();
    let even_in = crate::poisson::membrane_fill(bw, bh, 1, &even, &no_pic);
    let soft = |d: f32| cap * (d / cap).tanh();
    let even_at = |b: usize| if has_pic[b] { even[b] } else { 0.0 };
    let mut fill_corr = vec![0.0f32; n * ch];
    let mut band_corr = vec![0.0f32; n * ch];
    for b in 0..n {
        for c in 0..ch {
            let i = b * ch + c;
            if has_all[b] {
                fill_corr[i] = even_in[b] * soft(carried[i] - fill_light[i]);
            }
            if !fill && has_pic[b] && dist[b] < band as f32 {
                let t = (dist[b] / band as f32).clamp(0.0, 1.0);
                let fade = 1.0 - t * t * (3.0 - 2.0 * t);
                band_corr[i] = fade * even_at(b) * soft(carried[i] - pic_light[i]);
            }
        }
    }
    // Spread back over the pixels, bilinearly between block centres.
    let sample = |corr: &[f32], x: usize, y: usize, c: usize| -> f32 {
        let fx = ((x - x0) as f32 + 0.5) / s as f32 - 0.5;
        let fy = ((y - y0) as f32 + 0.5) / s as f32 - 0.5;
        let (ix, iy) = (fx.floor().max(0.0) as usize, fy.floor().max(0.0) as usize);
        let (ix, iy) = (ix.min(bw - 1), iy.min(bh - 1));
        let (jx, jy) = ((ix + 1).min(bw - 1), (iy + 1).min(bh - 1));
        let (tx, ty) = ((fx - ix as f32).clamp(0.0, 1.0), (fy - iy as f32).clamp(0.0, 1.0));
        let g = |bx: usize, by: usize| corr[(by * bw + bx) * ch + c];
        (g(ix, iy) * (1.0 - tx) + g(jx, iy) * tx) * (1.0 - ty) + (g(ix, jy) * (1.0 - tx) + g(jx, jy) * tx) * ty
    };
    // Which correction a pixel takes, if any.
    let which = |x: usize, y: usize| -> Option<&[f32]> {
        let p = y * w + x;
        if active[p] {
            Some(&fill_corr[..])
        } else if !fill && matches!(px[p], Px::Source | Px::Known) && dist[((y - y0) / s) * bw + (x - x0) / s] < (band + s) as f32 {
            Some(&band_corr[..])
        } else {
            None
        }
    };
    // The picture pixels about to change, as they were (row by row, in parallel).
    let mut changed = Relit::default();
    if !fill {
        let img_r = &*img;
        let rows = par::map(y1 - y0 + 1, par, |dy| {
            let y = y0 + dy;
            let mut r = Relit::default();
            for x in x0..=x1 {
                let p = y * w + x;
                if !active[p] && which(x, y).is_some() {
                    r.pixels.push(p);
                    r.was.extend_from_slice(&img_r[p * ch..(p + 1) * ch]);
                }
            }
            r
        });
        for r in rows {
            changed.pixels.extend(r.pixels);
            changed.was.extend(r.was);
        }
    }
    par::chunks_mut(img, w * ch, par, |y, row| {
        if y < y0 || y > y1 {
            return;
        }
        for x in x0..=x1 {
            if let Some(corr) = which(x, y) {
                for c in 0..ch {
                    row[x * ch + c] += sample(corr, x, y, c);
                }
            }
        }
    });
    changed
}

/// Picture pixels [`relight`] changed, and their values before.
#[derive(Default)]
pub(super) struct Relit {
    pub(super) pixels: Vec<usize>,
    pub(super) was: Vec<f32>,
}

impl Relit {
    /// Puts the pixels back in `img` (`ch` channels) as they were.
    pub(super) fn put_back(&self, img: &mut [f32], ch: usize) {
        for (i, p) in self.pixels.iter().enumerate() {
            img[p * ch..(p + 1) * ch].copy_from_slice(&self.was[i * ch..(i + 1) * ch]);
        }
    }
}

/// How far the picture strays from its own lighting where it counts as even (in 0..1 of a
/// channel); twice that is texture, which [`relight`] leaves alone.
const EVEN: f32 = 3.0 / 255.0;

/// `px` with the picture within `m` px of the hole matched against but not copied from
/// ([`Px::Known`]).
pub(super) fn keep_off_edge(px: &[Px], w: usize, h: usize, m: usize) -> Vec<Px> {
    let hole: Vec<bool> = px.iter().map(|p| *p == Px::Hole).collect();
    let near = dilate_box(&hole, w, h, m);
    px.iter().zip(&near).map(|(p, n)| if *n && *p == Px::Source { Px::Known } else { *p }).collect()
}
