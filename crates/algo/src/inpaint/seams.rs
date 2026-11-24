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
    if f == 0 {
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
    let rows = par::map(h, par, |y| {
        let mut out: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut copies: Vec<((i32, i32), f32)> = Vec::new();
        for x in 0..w {
            let p = y * w + x;
            if !active[p] || !near[p] {
                continue;
            }
            copies.clear();
            for qy in y.saturating_sub(f)..(y + f + 1).min(h) {
                for qx in x.saturating_sub(f)..(x + f + 1).min(w) {
                    let q = qy * w + qx;
                    if !active[q] {
                        continue;
                    }
                    let d = qx.abs_diff(x).max(qy.abs_diff(y));
                    let wt = (f + 1 - d) as f32;
                    match copies.iter_mut().find(|c| c.0 == labels[q]) {
                        Some(c) => c.1 += wt,
                        None => copies.push((labels[q], wt)),
                    }
                }
            }
            let mut acc = vec![0.0f32; ch];
            let mut sum = 0.0f32;
            for &(l, wt) in &copies {
                if let Some(v) = view.at(x, y, l) {
                    for (a, v) in acc.iter_mut().zip(v) {
                        *a += v * wt;
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
        img[p * ch..(p + 1) * ch].copy_from_slice(&v);
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
