//! A filled picture's last steps: its levels brought to the picture's around it, the horizon of a
//! landscape, and a sky drawn by a model with the picture's own fine texture.

use super::Px;
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

/// The horizon of an RGB landscape as a row per column ([`super::solver::horizon_line`]), from
/// the pixels outside `hole`; `None` without a clear one.
pub fn horizon_line(img: &[f32], hole: &[bool], w: usize, h: usize, ch: usize) -> Option<Vec<f32>> {
    let px: Vec<Px> = hole.iter().map(|hh| if *hh { Px::Hole } else { Px::Source }).collect();
    super::solver::horizon_line(img, &px, w, h, ch)
}

/// Rows above the horizon where [`sky_from_guide`] starts, and over how many more it takes over.
const SKY_START: f32 = 16.0;
const SKY_RAMP: f32 = 32.0;
/// The most fine texture [`sky_from_guide`] adds, per channel.
const SKY_DETAIL_MAX: f32 = 0.03;

/// The sky of a stamped fill from the model's drawing: the model lays clouds out and finishes
/// them better than copied pieces, which repeat whole clouds, but draws them soft. Above the
/// `horizon` (a row per column), each hole pixel becomes the model's (`guide`, in the picture's
/// tones) plus the fill's own fine texture (it less its box mean of `radius`, at most
/// [`SKY_DETAIL_MAX`]), taking over from
/// the fill over [`SKY_RAMP`] rows from [`SKY_START`] above the horizon. Worked in bands of rows,
/// so only a band's box means are held. `cc`: the colour channels, the first of `ch`.
#[allow(clippy::too_many_arguments)]
pub fn sky_from_guide(img: &mut [f32], guide: &[f32], hole: &[bool], w: usize, h: usize, ch: usize, cc: usize, radius: usize, horizon: &[f32]) {
    let cc = cc.clamp(1, ch.min(4));
    let r = radius.max(1);
    let weight = |x: usize, y: usize| ((horizon.get(x).copied().unwrap_or(0.0) - SKY_START - y as f32) / SKY_RAMP).clamp(0.0, 1.0);
    let top = horizon.iter().copied().fold(0.0f32, f32::max).max(0.0).ceil() as usize;
    const BAND: usize = 256;
    for y0 in (0..top.min(h)).step_by(BAND) {
        let y1 = (y0 + BAND).min(h);
        if !(y0..y1).any(|y| (0..w).any(|x| hole[y * w + x] && weight(x, y) > 0.0)) {
            continue;
        }
        // Box means of the fill over the band (with `r` rows of margin), per colour channel.
        let (a, b) = (y0.saturating_sub(r), (y1 + r).min(h));
        let bh = b - a;
        let mut mean = vec![0f32; w * (y1 - y0) * cc];
        for c in 0..cc {
            // Across: prefix sums per row.
            let mut across = vec![0f32; w * bh];
            let mut pre = vec![0f64; w + 1];
            for y in a..b {
                for x in 0..w {
                    pre[x + 1] = pre[x] + f64::from(img[(y * w + x) * ch + c]);
                }
                for x in 0..w {
                    let (l, rr) = (x.saturating_sub(r), (x + r + 1).min(w));
                    across[(y - a) * w + x] = ((pre[rr] - pre[l]) / (rr - l) as f64) as f32;
                }
            }
            // Down.
            for x in 0..w {
                for y in y0..y1 {
                    let (l, rr) = (y.saturating_sub(r).max(a), (y + r + 1).min(b));
                    let s: f32 = (l..rr).map(|yy| across[(yy - a) * w + x]).sum();
                    mean[((y - y0) * w + x) * cc + c] = s / (rr - l) as f32;
                }
            }
        }
        for y in y0..y1 {
            for x in 0..w {
                let i = y * w + x;
                let t = weight(x, y);
                if !hole[i] || t == 0.0 {
                    continue;
                }
                for c in 0..cc {
                    let v = img[i * ch + c];
                    // Fine texture only: a copied cloud's edge would etch a line into the sky.
                    let detail = (v - mean[((y - y0) * w + x) * cc + c]).clamp(-SKY_DETAIL_MAX, SKY_DETAIL_MAX);
                    let sky = guide[i * ch + c] + detail;
                    img[i * ch + c] = (v * (1.0 - t) + sky * t).max(0.0);
                }
            }
        }
    }
}
