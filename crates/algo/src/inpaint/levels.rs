//! A fill's sources and last steps: the fringe of a transparent edge kept out of its sources,
//! and its levels brought to the picture's around it.

use super::Px;
use crate::par;
use crate::poisson::membrane_fill;

/// The block [`seamless_levels`] measures and corrects in, px.
const LEVELS_BLOCK: usize = 16;
/// A step between the picture and the fill larger than this, in any colour channel, is not a
/// level but a different thing on each side (a cloud's edge by sky): not measured there.
const LEVELS_STEP_MAX: f32 = 0.06;

/// Brings a fill's levels to the picture's where they meet, smoothly over the hole. Pieces
/// copied from further in keep their own brightness, and a photo's levels drift across it (a
/// panorama's frames, vignetting), so the fill meets the picture at a step. Per block of
/// [`LEVELS_BLOCK`] px along the hole's edge, the step between the picture's pixels and the
/// fill's in it (with its neighbours'); a step above [`LEVELS_STEP_MAX`] is left out. The steps
/// are spread over the hole as a membrane over the blocks and added to the fill, so its texture
/// is kept and its levels bend to meet the picture all round. `cc`: the colour channels, the
/// first of `ch`.
pub fn seamless_levels(img: &mut [f32], hole: &[bool], w: usize, h: usize, ch: usize, cc: usize) {
    const B: usize = LEVELS_BLOCK;
    let cc = cc.clamp(1, ch.min(4));
    let (bw, bh) = (w.div_ceil(B), h.div_ceil(B));
    let (mut known, mut fill) = (vec![0f64; bw * bh * cc], vec![0f64; bw * bh * cc]);
    let (mut kn, mut fnum) = (vec![0u32; bw * bh], vec![0u32; bw * bh]);
    for (i, hh) in hole.iter().enumerate().take(w * h) {
        let k = (i / w / B) * bw + (i % w) / B;
        let (sum, n) = if *hh { (&mut fill, &mut fnum) } else { (&mut known, &mut kn) };
        n[k] += 1;
        for c in 0..cc {
            sum[k * cc + c] += f64::from(img[i * ch + c]);
        }
    }
    // The step at each edge block, over it and its neighbours (an edge along a block's side
    // has the picture in one and the fill in the next).
    let least = (B * B / 8) as u32;
    let mut step = vec![0.0f32; bw * bh * cc];
    let mut unknown = vec![true; bw * bh];
    for by in 0..bh {
        for bx in 0..bw {
            let k = by * bw + bx;
            // A block of the fill along the edge (beside the picture, in it or in a neighbour).
            if fnum[k] == 0 {
                continue;
            }
            let (mut ks, mut fs, mut kc, mut fc) = ([0f64; 4], [0f64; 4], 0u32, 0u32);
            for ny in by.saturating_sub(1)..(by + 2).min(bh) {
                for nx in bx.saturating_sub(1)..(bx + 2).min(bw) {
                    let j = ny * bw + nx;
                    (kc, fc) = (kc + kn[j], fc + fnum[j]);
                    for c in 0..cc {
                        ks[c] += known[j * cc + c];
                        fs[c] += fill[j * cc + c];
                    }
                }
            }
            if kc < least || fc < least {
                continue;
            }
            let d: Vec<f32> = (0..cc).map(|c| (ks[c] / f64::from(kc) - fs[c] / f64::from(fc)) as f32).collect();
            if d.iter().all(|d| d.abs() <= LEVELS_STEP_MAX) {
                unknown[k] = false;
                step[k * cc..(k + 1) * cc].copy_from_slice(&d);
            }
        }
    }
    if unknown.iter().all(|u| *u) {
        return;
    }
    let step = membrane_fill(bw, bh, cc, &step, &unknown);
    let at = |bx: usize, by: usize, c: usize| step[(by.min(bh - 1) * bw + bx.min(bw - 1)) * cc + c];
    for (i, hh) in hole.iter().enumerate().take(w * h) {
        if !*hh {
            continue;
        }
        let fx = ((i % w) as f32 + 0.5) / B as f32 - 0.5;
        let fy = ((i / w) as f32 + 0.5) / B as f32 - 0.5;
        let (x0, y0) = (fx.max(0.0) as usize, fy.max(0.0) as usize);
        let (tx, ty) = ((fx - x0 as f32).clamp(0.0, 1.0), (fy - y0 as f32).clamp(0.0, 1.0));
        for c in 0..cc {
            let top = at(x0, y0, c) * (1.0 - tx) + at(x0 + 1, y0, c) * tx;
            let bottom = at(x0, y0 + 1, c) * (1.0 - tx) + at(x0 + 1, y0 + 1, c) * tx;
            img[i * ch + c] = (img[i * ch + c] + top * (1.0 - ty) + bottom * ty).max(0.0);
        }
    }
}

/// The horizon of an RGB landscape as a row per column (the sky's blue meeting the ground's
/// red, per strip of columns), from the pixels outside `hole`; `None` without one.
pub fn horizon_line(img: &[f32], hole: &[bool], w: usize, h: usize, ch: usize) -> Option<Vec<f32>> {
    let px: Vec<Px> = hole.iter().map(|h| if *h { Px::Hole } else { Px::Source }).collect();
    super::solver::horizon_line(img, &px, w, h, ch)
}

/// `mask` grown by `r` px each way (a square): true where any of `mask` is within `r` across
/// and `r` down. Two running counts, so it costs the same for any `r`.
pub fn dilate_mask(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    if r == 0 || w == 0 || h == 0 {
        return mask.to_vec();
    }
    // Across: per row, a prefix count of set pixels.
    let mut across = vec![false; w * h];
    par::chunks_mut(&mut across, w, w * h >= PAR_PIXELS, |y, row| {
        let src = &mask[y * w..(y + 1) * w];
        let mut pre = vec![0u32; w + 1];
        for (x, s) in src.iter().enumerate() {
            pre[x + 1] = pre[x] + u32::from(*s);
        }
        for (x, o) in row.iter_mut().enumerate() {
            *o = pre[(x + r + 1).min(w)] > pre[x.saturating_sub(r)];
        }
    });
    // Down: a count per column over the rows within `r`, slid down the picture.
    let mut out = vec![false; w * h];
    let mut count = vec![0u32; w];
    let add = |count: &mut [u32], y: usize, by: i32| {
        for (c, a) in count.iter_mut().zip(&across[y * w..(y + 1) * w]) {
            if *a {
                *c = c.wrapping_add_signed(by);
            }
        }
    };
    for y in 0..r.min(h) {
        add(&mut count, y, 1);
    }
    for y in 0..h {
        if y + r < h {
            add(&mut count, y + r, 1);
        }
        if y > r {
            add(&mut count, y - r - 1, -1);
        }
        for (o, c) in out[y * w..(y + 1) * w].iter_mut().zip(&count) {
            *o = *c > 0;
        }
    }
    out
}

/// Above this many pixels [`dilate_mask`] works its rows in parallel.
const PAR_PIXELS: usize = 1 << 16;
/// An alpha below this is not solid ([`drop_fringe`]).
const ALPHA_SOLID: f32 = 0.999;

/// Takes the fringe of a transparent edge out of `source`: pixels that are not quite opaque,
/// and those within `fringe` px of one. A stitched panorama's edge fades out over a few pixels
/// whose colours are left over from the stitch (barely seen, at their alpha), and copied
/// opaque they came out as coloured noise; the last pixels before it are often frayed too.
/// The alpha is the last of `ch` channels. Left as it is if nothing would be left.
pub fn drop_fringe(source: &mut [bool], img: &[f32], w: usize, h: usize, ch: usize, fringe: usize) {
    if ch == 0 {
        return;
    }
    let soft: Vec<bool> = par::map(w * h, w * h >= PAR_PIXELS, |i| img[i * ch + ch - 1] < ALPHA_SOLID);
    let near = dilate_mask(&soft, w, h, fringe);
    if source.iter().zip(&near).any(|(s, n)| *s && !*n) {
        source.iter_mut().zip(&near).for_each(|(s, n)| *s &= !*n);
    }
}
