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
                    for &(nx, ny) in &nbrs[..n] {
                        let q = ny * w + nx;
                        if !active[q] {
                            continue;
                        }
                        let cand = lab[q];
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

/// The hole pixels whose whole `(2r + 1)²` box is hole: the ones a stamped fill takes from one
/// copy (nearer the picture it blends).
pub(super) fn deep_in_hole(px: &[Px], w: usize, h: usize, r: usize) -> Vec<bool> {
    let hole: Vec<bool> = px.iter().map(|p| *p == Px::Hole).collect();
    // Along rows, then down columns: all of the run of 2r + 1 centred here is hole (off the
    // picture counts as not).
    let run = |get: &dyn Fn(usize) -> bool, len: usize, out: &mut dyn FnMut(usize, bool)| {
        let mut count = 0usize;
        for i in 0..len + r {
            if i < len && get(i) {
                count += 1;
            }
            if i > 2 * r && get(i - 2 * r - 1) {
                count -= 1;
            }
            if i >= r {
                let c = i - r;
                let lo = c.saturating_sub(r);
                let hi = (c + r).min(len - 1);
                let span = hi - lo + 1;
                out(c, span == 2 * r + 1 && count == span);
            }
        }
    };
    let mut rows = vec![false; w * h];
    for y in 0..h {
        let line = &hole[y * w..(y + 1) * w];
        run(&|i| line[i], w, &mut |c, v| rows[y * w + c] = v);
    }
    let mut out = vec![false; w * h];
    for x in 0..w {
        run(&|i| rows[i * w + x], h, &mut |c, v| out[c * w + x] = v);
    }
    out
}
