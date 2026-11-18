use super::cells::{Cells, Span};
use super::common::{Adapt, PAR_MIN_PATCHES, Px, Rng};
use super::par;

/// Most channels whose moments the colour adaptation fits (the rest are copied as they are).
const MOM_CH: usize = 8;

/// Per-level PatchMatch/EM state.
pub(super) struct Solver<'a> {
    pub(super) w: usize,
    pub(super) h: usize,
    pub(super) ch: usize,
    pub(super) r: i32,
    pub(super) img: &'a mut Vec<f32>,
    /// A guided fill's guide at this level and how many pixels soft it is here
    /// ([`Solver::refresh_match`]); `None` unguided.
    pub(super) guide: Option<(&'a [f32], usize)>,
    /// What patches are compared on when guided ([`Solver::m`]): the picture, with the hole as
    /// the guide plus the current fill's own detail.
    pub(super) mimg: Option<Vec<f32>>,
    /// How much PatchMatch favours coherent matches: a match its left or upper neighbour's
    /// shifted by one is taken over a better one unless that is better by this share (0: the best
    /// wins). A guided fill sets it, so neighbouring pixels copy one piece of the picture and the
    /// vote stamps it rather than averaging many pieces that match a soft guide about equally.
    pub(super) coherence: f32,
    pub(super) px: &'a [Px],
    /// Valid source centres: the whole patch is inside the grid and may be copied from.
    pub(super) valid: Vec<bool>,
    /// Target centres (the patch overlaps the hole), within the hole's bounds grown by the patch
    /// radius: nothing outside them changes or is matched. The match field, its costs and these
    /// per-target tables hold one entry per target, in the cells' order.
    pub(super) cells: Cells,
    /// Per target: the whole patch is inside the grid and not void (so its distances need no
    /// bounds checks and its own moments can be kept).
    pub(super) full: Vec<bool>,
    /// With colour adaptation, per target and channel (at most [`MOM_CH`]): `Σt, Σt²` of the
    /// patch, for full patches, as of the last [`Solver::costs`].
    pub(super) tmom: Vec<[f32; 2]>,
    /// With colour adaptation where it pays: per source centre and channel, `Σs, Σs²` of its
    /// patch (source pixels never change within a level). Empty otherwise.
    pub(super) smom: Vec<[f32; 2]>,
    pub(super) adapt: Adapt,
    /// Cancellation: checked per row band; a cancelled solve's output is discarded.
    pub(super) ctl: photocraft_raster::Interrupt<'a>,
}

/// A channel deviation under which a patch counts as flat for [`fit`] (about 5 levels of 255).
const FLAT: f32 = 0.02;

/// Gain and bias of one channel for target moments `(Σt, Σt²)` and source moments `(Σs, Σs²)`
/// over `n` samples, within the limits.
#[inline]
pub(crate) fn fit(a: Adapt, n: f32, st: f32, stt: f32, ss: f32, sss: f32) -> (f32, f32) {
    let (mt, ms) = (st / n, ss / n);
    let vt = (stt / n - mt * mt).max(0.0);
    let vs = (sss / n - ms * ms).max(0.0);
    let g = if vs > 1e-10 { (vt / vs).sqrt() } else { 1.0 };
    let (mut lo, mut hi) = a.gain;
    // Nearly flat patches of about the same colour (their means within the bias of each other):
    // the ratio of their deviations is noise, so match the mean first, keeping the gain where the
    // bias can still bring the source's mean to the target's. Gained about zero, a gain at its
    // limit moved bright means further than the bias could bring back, and each channel by a
    // different amount: a tinted blotch in a smooth gradient. Textured patches, and flat ones
    // whose colours differ more (Content-Aware Move's Color), are fitted as before.
    if vt.max(vs) < FLAT * FLAT && (mt - ms).abs() <= a.bias && ms > 1e-6 {
        let (glo, ghi) = ((mt - a.bias) / ms, (mt + a.bias) / ms);
        if glo.max(lo) <= ghi.min(hi) {
            (lo, hi) = (glo.max(lo), ghi.min(hi));
        }
    }
    let g = g.clamp(lo, hi);
    let b = (mt - g * ms).clamp(-a.bias, a.bias);
    (g, b)
}

impl Solver<'_> {
    /// The pixels patches are compared on: the match image of a guided fill, else the picture.
    #[inline]
    pub(super) fn m(&self) -> &[f32] {
        self.mimg.as_deref().unwrap_or(&self.img[..])
    }

    /// A guided fill's match image, from the current fill: outside the hole the picture, in it the
    /// guide plus the fill's detail finer than the guide (the fill less its mean over the guide's
    /// softness). The guide decides each patch's layout and colour and never drifts; the detail,
    /// which a soft guide lacks, comes from the copies voted in so far, so neighbouring matches
    /// agree and the copies stay sharp. Nothing unguided.
    pub(super) fn refresh_match(&mut self) {
        let Some((guide, soft)) = self.guide else { return };
        let (w, h, ch) = (self.w, self.h, self.ch);
        let r = soft / 2;
        let mut m = self.mimg.take().unwrap_or_else(|| self.img.to_vec());
        let (img, px) = (&self.img[..], self.px);
        par::chunks_mut(&mut m, w * ch, w * h >= PAR_MIN_PATCHES, |y, row| {
            let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
            for x in 0..w {
                if px[y * w + x] != Px::Hole {
                    continue;
                }
                let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
                let n = ((y1 - y0) * (x1 - x0)) as f32;
                for c in 0..ch {
                    let i = (y * w + x) * ch + c;
                    let mean =
                        if r == 0 { img[i] } else { (y0..y1).flat_map(|yy| (x0..x1).map(move |xx| (yy * w + xx) * ch + c)).map(|j| img[j]).sum::<f32>() / n };
                    row[x * ch + c] = guide[i] + img[i] - mean;
                }
            }
        });
        self.mimg = Some(m);
    }

    #[inline]
    pub(super) fn is_valid(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h && self.valid[y as usize * self.w + x as usize]
    }

    /// Mark the full target patches (`full`), and keep the source moments where they pay off.
    pub(super) fn prepare(&mut self, keep_source_moments: bool) {
        let (w, h, r) = (self.w, self.h, self.r as usize);
        let side = 2 * r + 1;
        let cells = &self.cells;
        let mut full = vec![false; cells.len()];
        let px = self.px;
        cells.rows_mut(&mut full, 1, cells.len() >= PAR_MIN_PATCHES, |y, runs, out| {
            if y < r || y + r >= h {
                return;
            }
            for (x, f) in row_cells(runs).zip(out.iter_mut()) {
                if x < r || x + r >= w {
                    continue;
                }
                *f = (y - r..=y + r).all(|yy| px[yy * w + x - r..yy * w + x - r + side].iter().all(|k| *k != Px::Void));
            }
        });
        self.full = full;
        if self.adapt.is_none() {
            return;
        }
        let c8 = self.ch.min(MOM_CH);
        self.tmom = vec![[0.0; 2]; self.cells.len() * c8];
        if !keep_source_moments {
            return;
        }
        let (ch, img, valid, ctl) = (self.ch, self.m(), &self.valid, &self.ctl);
        let mut smom = vec![[0.0f32; 2]; w * h * c8];
        par::chunks_mut(&mut smom, w * c8, w * h >= PAR_MIN_PATCHES, |y, out| {
            if ctl.cancelled() {
                return;
            }
            for x in 0..w {
                if valid[y * w + x] {
                    patch_moments(img, w, ch, c8, side, x - r, y - r, &mut out[x * c8..(x + 1) * c8]);
                }
            }
        });
        self.smom = smom;
    }

    /// Distance between the target patch at `t` (its pixels inside the grid and not void; `k` its
    /// number among the targets) and the source patch at `s`, per sample; with colour adaptation
    /// the source is fitted first.
    #[inline]
    fn dist(&self, t: (i32, i32), k: usize, s: (i32, i32), cutoff: f32) -> f32 {
        let full = self.full[k];
        if !self.adapt.is_none() {
            return if full { self.dist_adapted_full(k, t, s) } else { self.dist_adapted(t, s) };
        }
        if full { self.dist_full(t, s, cutoff) } else { self.dist_clipped(t, s, cutoff) }
    }

    /// [`Solver::dist`] without adaptation, of a patch that may be clipped.
    pub(super) fn dist_clipped(&self, t: (i32, i32), s: (i32, i32), cutoff: f32) -> f32 {
        let img = self.m();
        let (w, h, ch, r) = (self.w as i32, self.h as i32, self.ch, self.r);
        let full = ((2 * r + 1) * (2 * r + 1)) as f32 * ch as f32;
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for dy in -r..=r {
            let ty = t.1 + dy;
            if ty < 0 || ty >= h {
                continue;
            }
            let sy = s.1 + dy;
            for dx in -r..=r {
                let tx = t.0 + dx;
                if tx < 0 || tx >= w || self.px[ty as usize * self.w + tx as usize] == Px::Void {
                    continue;
                }
                let ti = (ty as usize * self.w + tx as usize) * ch;
                let si = (sy as usize * self.w + (s.0 + dx) as usize) * ch;
                for c in 0..ch {
                    let d = img[ti + c] - img[si + c];
                    sum += d * d;
                }
                n += ch;
            }
            // Early out: the mean can only be ≥ sum / (full patch size).
            if sum > cutoff * full {
                return f32::INFINITY;
            }
        }
        if n == 0 { f32::INFINITY } else { sum / n as f32 }
    }

    /// [`Solver::dist_clipped`] of a full patch: the same sums, in the same order. It gives up
    /// (infinity) as soon as the sum shows the patch is worse than `cutoff`, which the sum, only
    /// growing, can show after any pixel: the caller only asks whether it is better.
    pub(super) fn dist_full(&self, t: (i32, i32), s: (i32, i32), cutoff: f32) -> f32 {
        let img = self.m();
        let (w, ch, r) = (self.w, self.ch, self.r);
        let side = (2 * r + 1) as usize;
        let full = ((2 * r + 1) * (2 * r + 1)) as f32 * ch as f32;
        let (span, limit) = (side * ch, cutoff * full);
        let mut sum = 0.0f32;
        for dy in -r..=r {
            let ti = ((t.1 + dy) as usize * w + (t.0 - r) as usize) * ch;
            let si = ((s.1 + dy) as usize * w + (s.0 - r) as usize) * ch;
            for (tp, sp) in img[ti..ti + span].chunks_exact(ch).zip(img[si..si + span].chunks_exact(ch)) {
                for (a, b) in tp.iter().zip(sp) {
                    let d = a - b;
                    sum += d * d;
                }
                if sum > limit {
                    return f32::INFINITY;
                }
            }
        }
        sum / (side * side * ch) as f32
    }

    /// Moments of the target patch at `t` and the source patch at `s` per channel:
    /// `[Σt, Σt², Σs, Σs², Σts]`, and the sample count.
    pub(super) fn moments(&self, t: (i32, i32), s: (i32, i32), m: &mut [[f32; 5]]) -> usize {
        let img = self.m();
        let (w, h, ch, r) = (self.w as i32, self.h as i32, self.ch, self.r);
        m.iter_mut().for_each(|v| *v = [0.0; 5]);
        let mut n = 0usize;
        for dy in -r..=r {
            let ty = t.1 + dy;
            if ty < 0 || ty >= h {
                continue;
            }
            let sy = s.1 + dy;
            for dx in -r..=r {
                let tx = t.0 + dx;
                if tx < 0 || tx >= w || self.px[ty as usize * self.w + tx as usize] == Px::Void {
                    continue;
                }
                let ti = (ty as usize * self.w + tx as usize) * ch;
                let si = (sy as usize * self.w + (s.0 + dx) as usize) * ch;
                for (c, mc) in m.iter_mut().enumerate().take(ch) {
                    let (a, b) = (img[ti + c], img[si + c]);
                    mc[0] += a;
                    mc[1] += a * a;
                    mc[2] += b;
                    mc[3] += b * b;
                    mc[4] += a * b;
                }
                n += 1;
            }
        }
        n
    }

    /// [`Solver::moments`] of a full target patch (`k` its number among the targets): the target's own
    /// sums kept from [`Solver::costs`], the source's from [`Solver::prepare`] when kept, each
    /// summed in the same order as there.
    pub(super) fn moments_full(&self, k: usize, t: (i32, i32), s: (i32, i32), m: &mut [[f32; 5]]) {
        let img = self.m();
        let (w, ch, r) = (self.w, self.ch, self.r);
        let c8 = m.len();
        let side = (2 * r + 1) as usize;
        let tm = &self.tmom[k * c8..(k + 1) * c8];
        let sm = self.smom.get((s.1 as usize * w + s.0 as usize) * c8..(s.1 as usize * w + s.0 as usize + 1) * c8);
        let mut acc = [[0.0f32; 3]; MOM_CH];
        for dy in -r..=r {
            let ti = ((t.1 + dy) as usize * w + (t.0 - r) as usize) * ch;
            let si = ((s.1 + dy) as usize * w + (s.0 - r) as usize) * ch;
            let (trow, srow) = (&img[ti..ti + side * ch], &img[si..si + side * ch]);
            if sm.is_some() {
                for (tp, sp) in trow.chunks_exact(ch).zip(srow.chunks_exact(ch)) {
                    for c in 0..c8 {
                        acc[c][2] += tp[c] * sp[c];
                    }
                }
            } else {
                for (tp, sp) in trow.chunks_exact(ch).zip(srow.chunks_exact(ch)) {
                    for c in 0..c8 {
                        let (a, b) = (tp[c], sp[c]);
                        acc[c][0] += b;
                        acc[c][1] += b * b;
                        acc[c][2] += a * b;
                    }
                }
            }
        }
        for (c, mc) in m.iter_mut().enumerate() {
            let [ss, sss] = match sm {
                Some(sm) => sm[c],
                None => [acc[c][0], acc[c][1]],
            };
            *mc = [tm[c][0], tm[c][1], ss, sss, acc[c][2]];
        }
    }

    pub(super) fn dist_adapted(&self, t: (i32, i32), s: (i32, i32)) -> f32 {
        let mut m = [[0.0f32; 5]; MOM_CH];
        let ch = self.ch.min(MOM_CH);
        let n = self.moments(t, s, &mut m[..ch]);
        if n == 0 {
            return f32::INFINITY;
        }
        adapted_cost(self.adapt, n as f32, &m[..ch])
    }

    pub(super) fn dist_adapted_full(&self, k: usize, t: (i32, i32), s: (i32, i32)) -> f32 {
        let mut m = [[0.0f32; 5]; MOM_CH];
        let ch = self.ch.min(MOM_CH);
        self.moments_full(k, t, s, &mut m[..ch]);
        let side = (2 * self.r + 1) as usize;
        adapted_cost(self.adapt, (side * side) as f32, &m[..ch])
    }

    /// Gain and bias per channel for the match `t → s` (identity without adaptation).
    fn transform(&self, t: (i32, i32), k: usize, s: (i32, i32), out: &mut [(f32, f32)]) {
        if self.adapt.is_none() {
            out.iter_mut().for_each(|v| *v = (1.0, 0.0));
            return;
        }
        let mut m = [[0.0f32; 5]; MOM_CH];
        let ch = self.ch.min(MOM_CH);
        let n = if self.full[k] {
            self.moments_full(k, t, s, &mut m[..ch]);
            let side = (2 * self.r + 1) as usize;
            side * side
        } else {
            self.moments(t, s, &mut m[..ch])
        };
        for (c, o) in out.iter_mut().enumerate() {
            *o = if n == 0 || c >= ch { (1.0, 0.0) } else { fit(self.adapt, n as f32, m[c][0], m[c][1], m[c][2], m[c][3]) };
        }
    }

    /// One PatchMatch step at target `t` (`k` its number) whose match is `cur`: propagation
    /// from the matches of the neighbours visited just before it (`left` in its row, `up` in its
    /// column, in this pass's direction `step`), then random search in exponentially shrinking
    /// windows around the best (two numbers from `rng` per window).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn pm_step(
        &self,
        t: (i32, i32),
        k: usize,
        cur: ((i32, i32), f32),
        left: Option<(i32, i32)>,
        up: Option<(i32, i32)>,
        step: i32,
        rng: &mut Rng,
        search_radius: i32,
    ) -> ((i32, i32), f32) {
        let (mut best, mut bc) = cur;
        // A coherent match (a neighbour's, shifted) counts as `favour` times better than it is.
        let favour = 1.0 + self.coherence;
        let mut coherent = false;
        if let Some(c) = left {
            let cand = (c.0 + step, c.1);
            coherent |= cand == best;
            if self.is_valid(cand.0, cand.1) && cand != best {
                let d = self.dist(t, k, cand, bc * favour);
                if d < bc * favour {
                    (best, bc, coherent) = (cand, d, true);
                }
            }
        }
        if let Some(c) = up {
            let cand = (c.0, c.1 + step);
            coherent |= cand == best;
            if self.is_valid(cand.0, cand.1) && cand != best {
                let lim = if coherent { bc } else { bc * favour };
                let d = self.dist(t, k, cand, lim);
                if d < lim {
                    (best, bc, coherent) = (cand, d, true);
                }
            }
        }
        let mut rad = search_radius;
        while rad >= 1 {
            let cand = (best.0 + rng.range(-rad, rad), best.1 + rng.range(-rad, rad));
            if self.is_valid(cand.0, cand.1) && cand != best {
                let lim = if coherent { bc / favour } else { bc };
                let d = self.dist(t, k, cand, lim);
                if d < lim {
                    (best, bc, coherent) = (cand, d, false);
                }
            }
            rad /= 2;
        }
        (best, bc)
    }

    /// One PatchMatch pass over all target rows, in bands of the grid's rows: each band is
    /// visited in raster order (backwards on odd passes), propagating within itself, its random
    /// numbers seeded by its index. `nnf` and `cost` are per target. The bands run in parallel;
    /// when there are fewer of them than threads, each is also swept in wavefronts of blocks (a
    /// cell depends only on its neighbours before it in its row and its column), which visits
    /// every cell with the same neighbours and random numbers as the raster order.
    ///
    /// A neighbour in the target box that is no target offers the match `(0, 0)`, as it did when
    /// the field was kept over the whole box; one outside the box (or the band) offers none.
    pub(super) fn patchmatch(&self, nnf: &mut [(i32, i32)], cost: &mut [f32], pass: usize, seed: u64, search_radius: i32) {
        const BAND: usize = 16;
        let cells = &self.cells;
        let tb = cells.tbox;
        if tb.width() == 0 || tb.height() == 0 || nnf.len() < cells.len() || cost.len() < cells.len() {
            return;
        }
        let reverse = pass % 2 == 1;
        let step: i32 = if reverse { -1 } else { 1 };
        let band_seed = |bi: usize| seed ^ (bi as u64).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ ((pass as u64) << 40);
        // The bands crossing the target box: index, first row, rows in the box.
        let mut bands = Vec::new();
        let mut y = tb.y0;
        while y < tb.y1 {
            let bi = y / BAND;
            let end = ((bi + 1) * BAND).min(tb.y1);
            bands.push((bi, y, end - y));
            y = end;
        }
        let threads = par::threads();
        if tb.area() >= PAR_MIN_PATCHES * 8 && !bands.is_empty() && bands.len() < threads {
            self.patchmatch_wavefront(nnf, cost, &bands, reverse, &band_seed, search_radius);
            return;
        }
        let band_fn = |bi: usize, ys: usize, rows: usize, nn: &mut [(i32, i32)], co: &mut [f32]| {
            if self.ctl.cancelled() {
                return;
            }
            // The band's targets are `base..` among all of them.
            let base = cells.before_row(ys);
            let mut rng = Rng::new(band_seed(bi));
            for ry in 0..rows {
                if self.ctl.cancelled() {
                    return;
                }
                let ly = if reverse { rows - 1 - ry } else { ry };
                let y = ys + ly;
                // The row before this one in the pass's direction, when it is in the band.
                let nly = ly as i32 - step;
                let up_row = (nly >= 0 && (nly as usize) < rows).then(|| ys + nly as usize);
                let mut visit = |x: usize, k: usize| {
                    let left = self.offer(x as i32 - step, y, |j| nn[j - base]);
                    let up = up_row.and_then(|uy| self.offer(x as i32, uy, |j| nn[j - base]));
                    let li = k - base;
                    let t = (x as i32, y as i32);
                    (nn[li], co[li]) = self.pm_step(t, k, (nn[li], co[li]), left, up, step, &mut rng, search_radius);
                };
                let runs = cells.row(y);
                if reverse {
                    runs.iter().rev().for_each(|s| (s.x0..s.x1).rev().for_each(|x| visit(x, s.at(x))));
                } else {
                    runs.iter().for_each(|s| (s.x0..s.x1).for_each(|x| visit(x, s.at(x))));
                }
            }
        };
        // The targets, band by band.
        let mut parts = Vec::with_capacity(bands.len());
        let (mut nn_rest, mut co_rest) = (&mut nnf[..cells.len()], &mut cost[..cells.len()]);
        for &(bi, ys, rows) in &bands {
            let n = cells.before_row(ys + rows) - cells.before_row(ys);
            let (nn, nr) = std::mem::take(&mut nn_rest).split_at_mut(n);
            let (co, cr) = std::mem::take(&mut co_rest).split_at_mut(n);
            parts.push((bi, ys, rows, nn, co));
            (nn_rest, co_rest) = (nr, cr);
        }
        let par = tb.area() >= PAR_MIN_PATCHES && parts.len() > 1;
        par::each(parts, par, |(bi, ys, rows, nn, co)| band_fn(bi, ys, rows, nn, co));
    }

    /// The match a neighbour at `(x, y)` offers PatchMatch's propagation, `field(k)` giving the
    /// match of target `k`: none outside the target box, `(0, 0)` where it is no target.
    #[inline]
    fn offer(&self, x: i32, y: usize, field: impl Fn(usize) -> (i32, i32)) -> Option<(i32, i32)> {
        let tb = self.cells.tbox;
        if x < tb.x0 as i32 || x >= tb.x1 as i32 {
            return None;
        }
        Some(self.cells.index(x as usize, y).map_or((0, 0), field))
    }

    /// [`Solver::patchmatch`] swept in wavefronts: the cells of each band in blocks of a row, a
    /// block once the block before it in its row and the one above it (in the pass's direction)
    /// are done. A target's random numbers are those it would draw in raster order (each target
    /// draws the same count).
    fn patchmatch_wavefront(
        &self,
        nnf: &mut [(i32, i32)],
        cost: &mut [f32],
        bands: &[(usize, usize, usize)],
        reverse: bool,
        band_seed: &(dyn Fn(usize) -> u64 + Sync),
        search_radius: i32,
    ) {
        use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

        let cells = &self.cells;
        let tb = cells.tbox;
        let tw = tb.width();
        let step: i32 = if reverse { -1 } else { 1 };
        let pack = |(x, y): (i32, i32)| (u64::from(x as u32) << 32) | u64::from(y as u32);
        let unpack = |v: u64| ((v >> 32) as u32 as i32, v as u32 as i32);
        // Random numbers each target draws.
        let mut draws = 0u64;
        let mut rad = search_radius;
        while rad >= 1 {
            draws += 2;
            rad /= 2;
        }
        // Targets visited before each of a band's rows (in the pass's order).
        let row_base: Vec<Vec<u64>> = bands
            .iter()
            .map(|&(_, ys, rows)| {
                let mut base = Vec::with_capacity(rows);
                let mut n = 0u64;
                for pr in 0..rows {
                    base.push(n);
                    let y = ys + if reverse { rows - 1 - pr } else { pr };
                    n += (cells.before_row(y + 1) - cells.before_row(y)) as u64;
                }
                base
            })
            .collect();
        let block = (tw / 8).clamp(4, 32);
        let blocks = tw.div_ceil(block);
        let nn: Vec<AtomicU64> = nnf[..cells.len()].iter().map(|m| AtomicU64::new(pack(*m))).collect();
        let co: Vec<AtomicU32> = cost[..cells.len()].iter().map(|c| AtomicU32::new(c.to_bits())).collect();
        let field = |k: usize| unpack(nn[k].load(Relaxed));
        let sweep = |(j, pr, b): (usize, usize, usize)| {
            let (bi, ys, rows) = bands[j];
            let ly = if reverse { rows - 1 - pr } else { pr };
            let y = ys + ly;
            let (pc0, pc1) = (b * block, ((b + 1) * block).min(tw));
            // The targets visited before this block, in raster order.
            let first = if reverse { tw - 1 - pc0 } else { pc0 };
            let in_row = cells.in_row(y, tb.x0 + first, !reverse);
            let mut rng = Rng::new(band_seed(bi));
            rng.skip((row_base[j][pr] + in_row as u64) * draws);
            let nly = ly as i32 - step;
            let up_row = (nly >= 0 && (nly as usize) < rows).then(|| ys + nly as usize);
            for pc in pc0..pc1 {
                let x = tb.x0 + if reverse { tw - 1 - pc } else { pc };
                let Some(k) = cells.index(x, y) else { continue };
                let left = self.offer(x as i32 - step, y, field);
                let up = up_row.and_then(|uy| self.offer(x as i32, uy, field));
                let cur = (unpack(nn[k].load(Relaxed)), f32::from_bits(co[k].load(Relaxed)));
                let t = (x as i32, y as i32);
                let (m, c) = self.pm_step(t, k, cur, left, up, step, &mut rng, search_radius);
                nn[k].store(pack(m), Relaxed);
                co[k].store(c.to_bits(), Relaxed);
            }
        };
        // Wavefront `d`: block `b` of (visited) row `pr` with `pr + b = d`, in every band. Each
        // wave ends before the next starts.
        let longest = bands.iter().map(|b| b.2).max().unwrap_or(0);
        for d in 0..longest + blocks - 1 {
            if self.ctl.cancelled() {
                return;
            }
            let wave: Vec<(usize, usize, usize)> = bands
                .iter()
                .enumerate()
                .flat_map(|(j, &(_, _, rows))| (0..rows).filter(move |&pr| pr <= d && d - pr < blocks).map(move |pr| (j, pr, d - pr)))
                .collect();
            par::each(wave, true, sweep);
        }
        for (m, a) in nnf.iter_mut().zip(&nn) {
            *m = unpack(a.load(Relaxed));
        }
        for (c, a) in cost.iter_mut().zip(&co) {
            *c = f32::from_bits(a.load(Relaxed));
        }
    }

    /// Recompute every target's match cost (hole pixels change between iterations), and with
    /// colour adaptation the full target patches' own moments.
    pub(super) fn costs(&mut self, nnf: &[(i32, i32)], cost: &mut [f32]) -> Result<(), photocraft_raster::Cancelled> {
        let (w, ch, r) = (self.w, self.ch, self.r as usize);
        let par = self.cells.len() >= PAR_MIN_PATCHES;
        self.ctl.check()?;
        if !self.adapt.is_none() {
            let c8 = ch.min(MOM_CH);
            let side = 2 * r + 1;
            let mut tmom = std::mem::take(&mut self.tmom);
            let (img, full, ctl) = (self.m(), &self.full, &self.ctl);
            self.cells.rows_mut(&mut tmom, c8, par, |y, runs, out| {
                if ctl.cancelled() {
                    return;
                }
                let base = runs[0].k;
                for (i, x) in row_cells(runs).enumerate() {
                    if full[base + i] {
                        patch_moments(img, w, ch, c8, side, x - r, y - r, &mut out[i * c8..(i + 1) * c8]);
                    }
                }
            });
            self.tmom = tmom;
        }
        let this = &*self;
        this.cells.rows_mut(cost, 1, par, |y, runs, co| {
            if this.ctl.cancelled() {
                return;
            }
            let base = runs[0].k;
            for ((i, x), c) in row_cells(runs).enumerate().zip(co.iter_mut()) {
                *c = this.dist((x as i32, y as i32), base + i, nnf[base + i], f32::INFINITY);
            }
        });
        self.ctl.check()
    }

    /// Re-estimate hole pixels by a similarity-weighted vote of all overlapping patch matches
    /// (each copy adjusted by its gain and bias).
    pub(super) fn vote(&mut self, nnf: &[(i32, i32)], cost: &[f32]) {
        const CHUNK: usize = 1 << 12;
        let (w, h, ch, r) = (self.w, self.h, self.ch, self.r);
        // A cancelled solve is thrown away.
        if self.ctl.cancelled() {
            return;
        }
        let n = self.cells.len();
        let used = |k: usize| cost[k].is_finite();
        // σ² from the 75th percentile of match costs (Wexler et al. use a robust scale like this).
        let mut cs: Vec<f32> = (0..n).filter(|&k| used(k)).map(|k| cost[k]).collect();
        let sigma2 = if cs.is_empty() {
            1.0
        } else {
            let k = (cs.len() * 3 / 4).min(cs.len() - 1);
            let (_, v, _) = cs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
            v.max(1e-6)
        };
        drop(cs);
        // Each target's weight and photometric transform (computed once, used by every pixel it
        // covers).
        let adapted = !self.adapt.is_none();
        let par = n >= PAR_MIN_PATCHES;
        // A weight (zero for a target that doesn't vote) and its match's centre (as a pixel
        // index) per target.
        let mut wt = vec![0.0f32; n];
        let mut src = vec![0isize; n];
        par::chunks2_mut(&mut wt, CHUNK, &mut src, CHUNK, par, |i, wrow, srow| {
            for (j, (o, sc)) in wrow.iter_mut().zip(srow.iter_mut()).enumerate() {
                let k = i * CHUNK + j;
                if used(k) {
                    *o = (-cost[k] / (2.0 * sigma2)).exp().max(1e-8);
                    *sc = nnf[k].1 as isize * w as isize + nnf[k].0 as isize;
                }
            }
        });
        let mut gb: Vec<(f32, f32)> = Vec::new();
        if adapted {
            gb = vec![(1.0f32, 0.0f32); n * ch];
            let this = &*self;
            this.cells.rows_mut(&mut gb, ch, par, |y, runs, out| {
                if this.ctl.cancelled() {
                    return;
                }
                let base = runs[0].k;
                for (i, x) in row_cells(runs).enumerate() {
                    if used(base + i) {
                        this.transform((x as i32, y as i32), base + i, nnf[base + i], &mut out[i * ch..(i + 1) * ch]);
                    }
                }
            });
        }
        // The vote of each target that is a hole pixel (every pixel reads the old pixels): its
        // channels, then 1 when any patch voted, else 0.
        let per = ch + 1;
        let mut out = vec![0.0f32; n * per];
        {
            let votes = Votes { w, h, ch, r: r as usize, cells: &self.cells, img: &*self.img, wt: &wt, src: &src, gb: &gb };
            let (px, ctl) = (self.px, &self.ctl);
            self.cells.rows_mut(&mut out, per, par, |y, runs, orow| {
                if ctl.cancelled() {
                    return;
                }
                for (i, x) in row_cells(runs).enumerate() {
                    if px[y * w + x] != Px::Hole {
                        continue;
                    }
                    let (dst, got) = orow[i * per..(i + 1) * per].split_at_mut(ch);
                    let voted = match (ch, adapted) {
                        (1, false) => votes.pixel::<1, false>(x, y, dst),
                        (3, false) => votes.pixel::<3, false>(x, y, dst),
                        (4, false) => votes.pixel::<4, false>(x, y, dst),
                        (1, true) => votes.pixel::<1, true>(x, y, dst),
                        (3, true) => votes.pixel::<3, true>(x, y, dst),
                        (4, true) => votes.pixel::<4, true>(x, y, dst),
                        _ => votes.pixel_any(x, y, adapted, dst),
                    };
                    got[0] = if voted { 1.0 } else { 0.0 };
                }
            });
        }
        drop((wt, src, gb));
        for (k, x, y) in self.cells.iter() {
            let o = &out[k * per..(k + 1) * per];
            if o[ch] != 0.0 {
                let d = (y * w + x) * ch;
                self.img[d..d + ch].copy_from_slice(&o[..ch]);
            }
        }
    }
}

/// The columns of a row's runs, left to right.
fn row_cells(runs: &[Span]) -> impl Iterator<Item = usize> + '_ {
    runs.iter().flat_map(|s| s.x0..s.x1)
}

/// What [`Solver::vote`] votes with: per target its weight (zero when it doesn't vote), its
/// match's centre as a pixel index, and with colour adaptation its gain and bias per channel.
struct Votes<'a> {
    w: usize,
    h: usize,
    ch: usize,
    r: usize,
    cells: &'a Cells,
    img: &'a [f32],
    wt: &'a [f32],
    src: &'a [isize],
    gb: &'a [(f32, f32)],
}

impl Votes<'_> {
    /// `f(target, column)` for the targets of row `ty` in columns `xa..=xb`, left to right.
    #[inline(always)]
    fn each_in(&self, ty: usize, xa: usize, xb: usize, mut f: impl FnMut(usize, usize)) {
        let runs = self.cells.row(ty);
        let start = runs.partition_point(|s| s.x1 <= xa);
        for s in &runs[start..] {
            if s.x0 > xb {
                break;
            }
            for tx in s.x0.max(xa)..s.x1.min(xb + 1) {
                f(s.at(tx), tx);
            }
        }
    }

    /// Hole pixel `(x, y)`'s vote into `out` (`CH` channels): the weighted mean of what the
    /// matches of the patches covering it put there, summed patch by patch in raster order of
    /// their centres. False when no patch votes.
    #[inline(always)]
    fn pixel<const CH: usize, const ADAPT: bool>(&self, x: usize, y: usize, out: &mut [f32]) -> bool {
        let (w, r) = (self.w, self.r);
        let mut acc = [0.0f32; CH];
        let mut wsum = 0.0f32;
        for ty in y.saturating_sub(r)..=(y + r).min(self.h - 1) {
            let dy = ty as isize - y as isize;
            self.each_in(ty, x.saturating_sub(r), (x + r).min(w - 1), |k, tx| {
                let wt = self.wt[k];
                if wt == 0.0 {
                    return;
                }
                // The pixel of the match's patch that lands here.
                let si = (self.src[k] - dy * w as isize - (tx as isize - x as isize)) as usize * CH;
                let s = &self.img[si..si + CH];
                for c in 0..CH {
                    let v = if ADAPT {
                        let (g, b) = self.gb[k * CH + c];
                        g * s[c] + b
                    } else {
                        s[c]
                    };
                    acc[c] += v * wt;
                }
                wsum += wt;
            });
        }
        if wsum > 0.0 {
            for (d, a) in out.iter_mut().zip(&acc) {
                *d = a / wsum;
            }
            return true;
        }
        false
    }

    /// [`Votes::pixel`] for any number of channels.
    fn pixel_any(&self, x: usize, y: usize, adapted: bool, out: &mut [f32]) -> bool {
        let (w, r, ch) = (self.w, self.r, self.ch);
        let mut acc = vec![0.0f32; ch];
        let mut wsum = 0.0f32;
        for ty in y.saturating_sub(r)..=(y + r).min(self.h - 1) {
            let dy = ty as isize - y as isize;
            self.each_in(ty, x.saturating_sub(r), (x + r).min(w - 1), |k, tx| {
                let wt = self.wt[k];
                if wt == 0.0 {
                    return;
                }
                let si = (self.src[k] - dy * w as isize - (tx as isize - x as isize)) as usize * ch;
                for (c, a) in acc.iter_mut().enumerate() {
                    let v = if adapted {
                        let (g, b) = self.gb[k * ch + c];
                        g * self.img[si + c] + b
                    } else {
                        self.img[si + c]
                    };
                    *a += v * wt;
                }
                wsum += wt;
            });
        }
        if wsum > 0.0 {
            for (d, a) in out.iter_mut().zip(&acc) {
                *d = a / wsum;
            }
            return true;
        }
        false
    }
}

/// `Σv, Σv²` per channel (the first `out.len()`) of the `side × side` patch with top-left
/// `(x0, y0)`, summed row by row, pixel by pixel, as [`Solver::moments`] does.
#[allow(clippy::too_many_arguments)]
fn patch_moments(img: &[f32], w: usize, ch: usize, c8: usize, side: usize, x0: usize, y0: usize, out: &mut [[f32; 2]]) {
    out.iter_mut().for_each(|v| *v = [0.0; 2]);
    for y in y0..y0 + side {
        let row = &img[(y * w + x0) * ch..(y * w + x0 + side) * ch];
        for p in row.chunks_exact(ch) {
            for (c, o) in out.iter_mut().enumerate().take(c8) {
                let a = p[c];
                o[0] += a;
                o[1] += a * a;
            }
        }
    }
}

/// The adapted distance from the moments `[Σt, Σt², Σs, Σs², Σts]` of each channel over `n`
/// samples: what is left of the target after the source is fitted to it, per sample.
#[inline]
fn adapted_cost(a: Adapt, nf: f32, m: &[[f32; 5]]) -> f32 {
    let mut sum = 0.0f32;
    for mc in m {
        let (g, b) = fit(a, nf, mc[0], mc[1], mc[2], mc[3]);
        // Σ (t − g s − b)², expanded.
        sum += mc[1] + g * g * mc[3] + nf * b * b - 2.0 * g * mc[4] - 2.0 * b * mc[0] + 2.0 * g * b * mc[2];
    }
    sum.max(0.0) / (nf * m.len() as f32)
}

/// `Σ (t − s)²` over a target patch's known pixels (offsets `offs` from its centre, samples
/// `tv`, `CH` per pixel) against the source patch centred at pixel `base`, summed in that order;
/// `None` once the sum reaches `limit` (it only grows).
#[inline(always)]
pub(super) fn known_sse<const CH: usize>(image: &[f32], offs: &[isize], tv: &[f32], base: isize, limit: f32) -> Option<f32> {
    let mut e = 0.0f32;
    for (o, t) in offs.iter().zip(tv.as_chunks::<CH>().0) {
        let si = (base + o) as usize * CH;
        let s = &image[si..si + CH];
        for c in 0..CH {
            let d = t[c] - s[c];
            e += d * d;
        }
        if e >= limit {
            return None;
        }
    }
    Some(e)
}

/// [`known_sse`] of [`LANES`] sources at once, each against its own limit (summed in step: the
/// sums are independent, so the processor overlaps them).
#[inline(always)]
pub(super) fn known_sse_lanes<const CH: usize>(image: &[f32], offs: &[isize], tv: &[f32], bases: [isize; LANES], limits: [f32; LANES]) -> [Option<f32>; LANES] {
    let mut e = [0.0f32; LANES];
    for (o, t) in offs.iter().zip(tv.as_chunks::<CH>().0) {
        for l in 0..LANES {
            let si = (bases[l] + o) as usize * CH;
            let s = &image[si..si + CH];
            for c in 0..CH {
                let d = t[c] - s[c];
                e[l] += d * d;
            }
        }
        // Sums only grow: once all of them reached their limits, none can win.
        if (0..LANES).all(|l| e[l] >= limits[l]) {
            break;
        }
    }
    std::array::from_fn(|l| (e[l] < limits[l]).then_some(e[l]))
}

/// Sources [`known_sse_lanes`] sums at once.
pub(super) const LANES: usize = 4;

/// [`known_sse`] for any number of channels.
pub(super) fn known_sse_any(image: &[f32], ch: usize, offs: &[isize], tv: &[f32], base: isize, limit: f32) -> Option<f32> {
    let mut e = 0.0f32;
    for (o, t) in offs.iter().zip(tv.chunks_exact(ch.max(1))) {
        let si = (base + o) as usize * ch;
        for (tc, sc) in t.iter().zip(&image[si..si + ch]) {
            let d = tc - sc;
            e += d * d;
        }
        if e >= limit {
            return None;
        }
    }
    Some(e)
}

/// The best source so far of [`onion_peel`]'s search.
#[derive(Clone, Copy)]
pub(super) struct Best {
    /// Error per sample, and the place in the sources.
    pub(super) v: f32,
    pub(super) j: usize,
    /// Sums of squared errors from which `sum / n` is above `v`, and at least `v`.
    pub(super) worse: f32,
    pub(super) equal: f32,
}

/// The least sum `e ≥ 0` with `e / n > v` (`strict`) or `e / n ≥ v` in float arithmetic (the
/// quotient only grows with `e`); infinity for an infinite `v`.
pub(super) fn least_sum(v: f32, n: f32, strict: bool) -> f32 {
    let holds = |e: f32| if strict { e / n > v } else { e / n >= v };
    if !v.is_finite() || !n.is_finite() || n <= 0.0 {
        return f32::INFINITY;
    }
    let mut e = (v * n).max(0.0);
    while e > 0.0 && holds(e.next_down()) {
        e = e.next_down();
    }
    while !holds(e) && e < f32::INFINITY {
        e = e.next_up();
    }
    e
}
