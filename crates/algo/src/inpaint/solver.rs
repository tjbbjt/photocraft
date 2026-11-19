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
    /// What patches are compared on when guided ([`Solver::m`]): the picture with the guide in
    /// the hole, softened ([`Solver::refresh_match`]).
    pub(super) mimg: Option<Vec<f32>>,
    /// How much PatchMatch favours coherent matches: a match its left or upper neighbour's
    /// shifted by one is taken over a better one unless that is better by this share (0: the best
    /// wins). A guided fill sets it, so neighbouring pixels copy one piece of the picture and the
    /// vote stamps it rather than averaging many pieces that match a soft guide about equally.
    pub(super) coherence: f32,
    /// A guided fill's vote stamps: each pixel takes the best match covering it (the highest
    /// weight) rather than the mean of all of them, so the copies stay sharp. Off unguided.
    pub(super) stamp: bool,
    /// With [`Self::stamp`]: a pixel within a patch of the picture around the hole is still the
    /// mean, so a stamped fill blends into the picture rather than meeting it at a hard edge.
    pub(super) blend_edge: bool,
    /// A nearby fill's reach at this level ([`CompleteParams::nearby`]): sources at most this
    /// many pixels from their target across and down; 0 anywhere.
    pub(super) reach: i32,
    /// A nearby fill at the same depth ([`CompleteParams::same_depth`]): how far up or down
    /// each target may copy from.
    pub(super) depth: Option<Depth>,
    /// A nearby fill's count of how often each part of the picture is copied: a part copied
    /// more than once costs more for the next target ([`Uses`]).
    pub(super) uses: Option<Uses>,
    /// At the same depth with a sky: cloud copied only into the sky's clouds, sky into its sky
    /// ([`SkyLayout`]).
    pub(super) sky: Option<SkyView>,
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

    /// A guided fill's match image: the picture with the guide in the hole, softened by a box
    /// reaching twice the guide's softness (at least 2 px), so a match agrees with the guide in
    /// the large and is free in the small. Softened less, a shrub in a source patch still showed
    /// as a dark spot the guide's flat ground lacks, so sources without shrubs won and the fill
    /// had half the photo's structure; at twice, the fill is as textured as the photo. Compared sharp, a soft guide matched smooth sources best (a
    /// road, a haze) and the fill came out smeared; softened on both sides, a ground of shrubs
    /// matches the guide's soft ground as well as a road does, and the fill copies shrubs. The
    /// picture outside the hole never changes and the guide is fixed, so it is worked out once
    /// per level. Nothing unguided.
    pub(super) fn refresh_match(&mut self) {
        let Some((guide, soft)) = self.guide else { return };
        if self.mimg.is_some() {
            return;
        }
        let (w, h, ch, px) = (self.w, self.h, self.ch, self.px);
        let img = &self.img[..];
        let r = (2 * soft).max(2);
        let row = w * ch;
        let mut m = vec![0.0f32; img.len()];
        par::chunks_mut(&mut m, row.max(1), w * h >= PAR_MIN_PATCHES, |y, out| {
            // Column sums over the rows in reach, then the mean along the row.
            let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
            let mut cols = vec![0.0f32; row];
            for yy in y0..y1 {
                for x in 0..w {
                    let i = yy * w + x;
                    let from = if px[i] == Px::Hole { guide } else { img };
                    for (c, v) in cols[x * ch..(x + 1) * ch].iter_mut().zip(&from[i * ch..(i + 1) * ch]) {
                        *c += *v;
                    }
                }
            }
            let rows = (y1 - y0) as f32;
            for x in 0..w {
                let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
                let k = 1.0 / ((x1 - x0) as f32 * rows);
                for c in 0..ch {
                    out[x * ch + c] = (x0..x1).map(|xx| cols[xx * ch + c]).sum::<f32>() * k;
                }
            }
        });
        self.mimg = Some(m);
    }

    /// Whether target `t` may copy source `s`: within reach of it, for a nearby fill, and at
    /// about its depth, for one at the same depth.
    #[inline]
    pub(super) fn allowed(&self, t: (i32, i32), s: (i32, i32)) -> bool {
        (self.reach <= 0 || ((s.0 - t.0).abs() <= self.reach && (s.1 - t.1).abs() <= self.rows_of(t))) && self.sky.as_ref().is_none_or(|k| k.allows(t, s))
    }

    /// What copying `s` costs on top of the patch distance: [`Uses::factor`], 1 without.
    #[inline]
    fn uses_factor(&self, s: (i32, i32)) -> f32 {
        self.uses.as_ref().map_or(1.0, |u| u.factor(s))
    }

    /// The patch distance of copying `s` to target `t`, times what copying `s` costs
    /// ([`Self::uses_factor`]); past `limit` it may stop early, returning more than `limit`.
    #[inline]
    fn cost_of(&self, t: (i32, i32), k: usize, s: (i32, i32), limit: f32) -> f32 {
        let f = self.uses_factor(s);
        if f == 1.0 { self.dist(t, k, s, limit) } else { self.dist(t, k, s, limit / f) * f }
    }

    /// Counts how often each part of the picture is copied by `nnf` (with [`Self::uses`]).
    pub(super) fn count_uses(&mut self, nnf: &[(i32, i32)]) {
        if let Some(u) = &mut self.uses {
            u.count(nnf.get(..self.cells.len()).unwrap_or(nnf));
        }
    }

    /// How many rows up or down target `t` may copy from (a nearby fill's).
    #[inline]
    fn rows_of(&self, t: (i32, i32)) -> i32 {
        self.depth.as_ref().map_or(self.reach, |d| d.rows(t).min(self.reach))
    }

    /// A source `t` may copy ([`Self::allowed`]): one of 8 random tries, else the one found
    /// ahead for its block ([`Depth::fallback`]).
    fn within_reach(&self, t: (i32, i32), rng: &mut Rng) -> Option<(i32, i32)> {
        let (r, rows) = (self.reach, self.rows_of(t));
        for _ in 0..8 {
            let cand = (t.0 + rng.range(-r, r), t.1 + rng.range(-rows, rows));
            if self.is_valid(cand.0, cand.1) && self.allowed(t, cand) {
                return Some(cand);
            }
        }
        self.depth.as_ref().and_then(|d| d.fallback(t))
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
    pub(super) fn pm_step(
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
        // Nearby: a match out of reach (a random start, a coarser level's) is replaced by one
        // within it: a random one, or the nearest when the hole leaves few to pick from.
        if !self.allowed(t, best)
            && let Some(cand) = self.within_reach(t, rng)
        {
            (best, bc) = (cand, self.cost_of(t, k, cand, f32::INFINITY));
        }
        // A coherent match (a neighbour's, shifted) counts as `favour` times better than it is.
        let favour = 1.0 + self.coherence;
        let mut coherent = false;
        if let Some(c) = left {
            let cand = (c.0 + step, c.1);
            coherent |= cand == best;
            if self.is_valid(cand.0, cand.1) && cand != best && self.allowed(t, cand) {
                let d = self.cost_of(t, k, cand, bc * favour);
                if d < bc * favour {
                    (best, bc, coherent) = (cand, d, true);
                }
            }
        }
        if let Some(c) = up {
            let cand = (c.0, c.1 + step);
            coherent |= cand == best;
            if self.is_valid(cand.0, cand.1) && cand != best && self.allowed(t, cand) {
                let lim = if coherent { bc } else { bc * favour };
                let d = self.cost_of(t, k, cand, lim);
                if d < lim {
                    (best, bc, coherent) = (cand, d, true);
                }
            }
        }
        let mut rad = search_radius;
        while rad >= 1 {
            let cand = (best.0 + rng.range(-rad, rad), best.1 + rng.range(-rad, rad));
            if self.is_valid(cand.0, cand.1) && cand != best && self.allowed(t, cand) {
                let lim = if coherent { bc / favour } else { bc };
                let d = self.cost_of(t, k, cand, lim);
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
                *c = this.cost_of((x as i32, y as i32), base + i, nnf[base + i], f32::INFINITY);
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
            let votes = Votes {
                w,
                h,
                ch,
                r: r as usize,
                cells: &self.cells,
                img: &*self.img,
                wt: &wt,
                src: &src,
                gb: &gb,
                stamp: self.stamp,
                blend_edge: self.blend_edge.then_some(self.px),
            };
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
    /// Each pixel from the best match covering it ([`Solver::stamp`]), not the mean.
    stamp: bool,
    /// The pixel roles, when a pixel a patch of the picture covers takes the mean anyway
    /// ([`Solver::blend_edge`]).
    blend_edge: Option<&'a [Px]>,
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
        let mut best = (0.0f32, [0.0f32; CH]);
        let mut picture = false;
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
                let mut v = [0.0f32; CH];
                for c in 0..CH {
                    v[c] = if ADAPT {
                        let (g, b) = self.gb[k * CH + c];
                        g * s[c] + b
                    } else {
                        s[c]
                    };
                    acc[c] += v[c] * wt;
                }
                if self.stamp && wt > best.0 {
                    best = (wt, v);
                }
                picture |= self.blend_edge.is_some_and(|px| px[ty * w + tx] != Px::Hole);
                wsum += wt;
            });
        }
        if self.stamp && wsum > 0.0 && !picture {
            out[..CH].copy_from_slice(&best.1);
            return true;
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
        let (mut best_wt, mut best) = (0.0f32, vec![0.0f32; ch]);
        let mut picture = false;
        for ty in y.saturating_sub(r)..=(y + r).min(self.h - 1) {
            let dy = ty as isize - y as isize;
            self.each_in(ty, x.saturating_sub(r), (x + r).min(w - 1), |k, tx| {
                let wt = self.wt[k];
                if wt == 0.0 {
                    return;
                }
                let si = (self.src[k] - dy * w as isize - (tx as isize - x as isize)) as usize * ch;
                let stamp = self.stamp && wt > best_wt;
                if stamp {
                    best_wt = wt;
                }
                for (c, a) in acc.iter_mut().enumerate() {
                    let v = if adapted {
                        let (g, b) = self.gb[k * ch + c];
                        g * self.img[si + c] + b
                    } else {
                        self.img[si + c]
                    };
                    *a += v * wt;
                    if stamp {
                        best[c] = v;
                    }
                }
                picture |= self.blend_edge.is_some_and(|px| px[ty * w + tx] != Px::Hole);
                wsum += wt;
            });
        }
        if self.stamp && wsum > 0.0 && !picture {
            out.copy_from_slice(&best);
            return true;
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

/// Where a nearby fill copies from, up and down ([`CompleteParams::nearby`](super::CompleteParams::nearby)),
/// and a source it can always fall back on.
///
/// At the same depth ([`CompleteParams::same_depth`](super::CompleteParams::same_depth)): in a
/// landscape the distance goes with the height in the picture, so a target copies from about
/// its own rows, and a far, finer stretch of ground never lands in a near one. How many rows
/// grows with the target's distance from the horizon, as in perspective: just below it a row
/// spans much more ground than at the bottom, close by (and the sky's clouds the same, above).
/// A target with no source within reach in its own rows (deep in a hole along the bottom) may
/// go as far again up or down as the nearest rows that have one, twice over.
pub(super) struct Depth {
    /// Rows up or down any target may go.
    least: i32,
    /// And per row from the horizon: a target `d` rows from it may go `d * per_row` rows.
    per_row: f32,
    /// The horizon's row ([`horizon`]).
    horizon: f32,
    /// Per block of [`DEPTH_BLOCK`] px: rows to the nearest rows with a source within reach of
    /// all its pixels, and one such source (x < 0: none in its columns).
    to_sources: Vec<(u16, (i32, i32))>,
    /// Blocks across.
    bw: usize,
}

/// The block [`Depth`] measures in, px.
const DEPTH_BLOCK: usize = 8;

impl Depth {
    /// `valid`: the level's valid sources, `w` by `h`; `reach`: the fill's reach there; `least`,
    /// `per_row` and `horizon`: the rows up or down a target may go ([`Depth`]); `least` is at
    /// least a block, so a source found in a block's own rows is within it.
    pub(super) fn new(valid: &[bool], w: usize, h: usize, reach: usize, least: i32, per_row: f32, horizon: f32) -> Self {
        let (bw, bh) = (w.div_ceil(DEPTH_BLOCK), h.div_ceil(DEPTH_BLOCK));
        const NONE: (i32, i32) = (-1, -1);
        // A source in each block's rows within reach of all its pixels, if any.
        let mut found = vec![NONE; bw * bh];
        let mut next = vec![usize::MAX; w + 1];
        for (y, row) in valid.chunks_exact(w).enumerate().take(h) {
            // next[x]: the first valid column at or after x.
            next[w] = usize::MAX;
            for x in (0..w).rev() {
                next[x] = if row[x] { x } else { next[x + 1] };
            }
            for bx in 0..bw {
                let k = (y / DEPTH_BLOCK) * bw + bx;
                if found[k] != NONE {
                    continue;
                }
                let (x0, x1) = (bx * DEPTH_BLOCK, ((bx + 1) * DEPTH_BLOCK).min(w));
                let (a, b) = ((x1 - 1).saturating_sub(reach), (x0 + reach + 1).min(w));
                if a < b && next[a] < b {
                    found[k] = (next[a] as i32, y as i32);
                }
            }
        }
        // Blocks to the nearest such block up or down, in rows, with its source.
        let mut to_sources = vec![(u16::MAX, NONE); bw * bh];
        for bx in 0..bw {
            let mut d = (u16::MAX, NONE);
            for by in 0..bh {
                d = if found[by * bw + bx] != NONE { (0, found[by * bw + bx]) } else { (d.0.saturating_add(1), d.1) };
                to_sources[by * bw + bx] = d;
            }
            d = (u16::MAX, NONE);
            for by in (0..bh).rev() {
                d = if found[by * bw + bx] != NONE { (0, found[by * bw + bx]) } else { (d.0.saturating_add(1), d.1) };
                let k = &mut to_sources[by * bw + bx];
                if d.0 < k.0 {
                    *k = d;
                }
            }
        }
        for k in &mut to_sources {
            k.0 = k.0.saturating_mul(DEPTH_BLOCK as u16);
        }
        Self { least: least.max(DEPTH_BLOCK as i32), per_row, horizon, to_sources, bw }
    }

    #[inline]
    fn block(&self, t: (i32, i32)) -> Option<&(u16, (i32, i32))> {
        self.to_sources.get((t.1.max(0) as usize / DEPTH_BLOCK) * self.bw + t.0.max(0) as usize / DEPTH_BLOCK)
    }

    /// Rows up or down target `t` may copy from.
    #[inline]
    pub(super) fn rows(&self, t: (i32, i32)) -> i32 {
        let off = ((t.1 as f32 - self.horizon).abs() * self.per_row) as i32;
        off.max(self.least) + 2 * i32::from(self.block(t).map_or(0, |b| b.0))
    }

    /// A valid source within reach of `t` and within its rows, found ahead.
    #[inline]
    pub(super) fn fallback(&self, t: (i32, i32)) -> Option<(i32, i32)> {
        self.block(t).map(|b| b.1).filter(|s| s.0 >= 0)
    }
}

/// The row of the horizon in a landscape, for [`Depth`]: the row that best splits the known
/// pixels into sky above (bluer than red, clouds included) and ground below (not). `None` when
/// that split fits under three in four of them (no clear horizon), or the picture has no colour.
pub(super) fn horizon(img: &[f32], px: &[Px], w: usize, h: usize, ch: usize) -> Option<f32> {
    if ch < 3 {
        return None;
    }
    // Per row: known pixels, and how many look like sky.
    let mut rows = vec![(0u32, 0u32); h];
    for (i, k) in px.iter().enumerate().take(w * h) {
        if *k == Px::Hole {
            continue;
        }
        let p = &img[i * ch..i * ch + 3];
        let r = &mut rows[i / w];
        r.0 += 1;
        r.1 += u32::from(p[2] > p[0] * 1.04 + 1e-4);
    }
    let known: u64 = rows.iter().map(|r| u64::from(r.0)).sum();
    let ground: u64 = rows.iter().map(|r| u64::from(r.0 - r.1)).sum();
    if known == 0 {
        return None;
    }
    // Agreement with "sky above y, ground from y on", for each y.
    let (mut fit, mut best) = (ground, (ground, 0usize));
    for (y, r) in rows.iter().enumerate() {
        fit = fit + u64::from(r.1) - u64::from(r.0 - r.1);
        if fit > best.0 {
            best = (fit, y + 1);
        }
    }
    (best.0 * 4 >= known * 3 && best.1 > 0 && best.1 < h).then_some(best.1 as f32)
}

/// How often each part of the picture is copied, for a nearby fill: Classic takes whatever
/// matches best, and in a hole along a border the same few pieces nearby match best again and
/// again, in rows and streaks. Counted per block of a patch's size, from the matches of the pass
/// before: a block copied about once (a stretch copied whole, one pixel each) costs nothing more;
/// past that, each further copy of it costs [`Uses::weight`] times the patch distance more, so
/// the search takes another part of the picture that matches nearly as well.
pub(super) struct Uses {
    /// The block side, px.
    side: usize,
    /// Blocks across.
    bw: usize,
    /// Targets copying from each block.
    counts: Vec<u32>,
    /// The extra cost of each copy past the first, as a share of the patch distance.
    weight: f32,
}

impl Uses {
    pub(super) fn new(w: usize, h: usize, side: usize, weight: f32) -> Self {
        let side = side.max(1);
        let bw = w.div_ceil(side);
        Self { side, bw, counts: vec![0; bw * h.div_ceil(side)], weight }
    }

    pub(super) fn count(&mut self, nnf: &[(i32, i32)]) {
        self.counts.fill(0);
        for s in nnf {
            if let Some(c) = self.block(*s).and_then(|k| self.counts.get_mut(k)) {
                *c += 1;
            }
        }
    }

    #[inline]
    fn block(&self, s: (i32, i32)) -> Option<usize> {
        (s.0 >= 0 && s.1 >= 0).then(|| (s.1 as usize / self.side) * self.bw + s.0 as usize / self.side)
    }

    /// What copying `s` costs, times the patch distance: 1 for a block copied once or less.
    #[inline]
    pub(super) fn factor(&self, s: (i32, i32)) -> f32 {
        let n = self.block(s).and_then(|k| self.counts.get(k)).copied().unwrap_or(0) as f32;
        let once = (self.side * self.side) as f32;
        1.0 + self.weight * (n / once - 1.0).max(0.0)
    }
}

/// Where the sky's clouds go, for a fill at the same depth, laid out once on a level at least
/// [`SkyLayout::MIN_WIDTH`] px across. Above the horizon the known pixels split into two tones,
/// sky and cloud (Otsu); each cloud in the fill so far (the model's, or Classic's coarser one)
/// that touches a cloud of the picture finishes that cloud and stays. One touching none is a new
/// cloud, and becomes plain sky, unless the picture has [`SkyLayout::SCATTERED`] small clouds in
/// rows near it, a sky of many small clouds where another is one more of them.
pub(super) struct SkyLayout {
    /// Per pixel of the level it was laid out on: 0 below the horizon, 1 sky, 2 cloud.
    class: Vec<u8>,
    w: usize,
    h: usize,
    /// The tone between sky and cloud.
    split: f32,
}

impl SkyLayout {
    /// The narrowest level the layout is made on (or the finest, when narrower).
    pub(super) const MIN_WIDTH: usize = 256;
    /// Small clouds in nearby rows that let a new cloud stay.
    const SCATTERED: usize = 16;

    /// The layout of `img` (the level's picture and fill so far, `w` by `h`, `ch` channels, at
    /// least three) above `horizon`; new clouds in the fill become the sky's mean colour. `None`
    /// without two clear tones above the horizon.
    pub(super) fn new(img: &mut [f32], px: &[Px], w: usize, h: usize, ch: usize, horizon: f32) -> Option<Self> {
        let tone = |img: &[f32], i: usize| img[i * ch..i * ch + 3].iter().sum::<f32>() / 3.0;
        let hz = (horizon as usize).min(h);
        if ch < 3 || hz < 2 {
            return None;
        }
        // Otsu over the known sky's tones.
        const BINS: usize = 64;
        let mut hist = [0f64; BINS];
        for i in (0..w * hz).filter(|i| px[*i] != Px::Hole) {
            hist[((tone(img, i).clamp(0.0, 1.0) * BINS as f32) as usize).min(BINS - 1)] += 1.0;
        }
        let n: f64 = hist.iter().sum();
        let centre = |b: usize| (b as f64 + 0.5) / BINS as f64;
        let total: f64 = hist.iter().enumerate().map(|(b, v)| v * centre(b)).sum();
        let (mut n0, mut s0, mut best) = (0.0, 0.0, (-1.0, 0usize, 0.0, 0.0));
        for (b, v) in hist.iter().enumerate().take(BINS - 1) {
            n0 += v;
            s0 += v * centre(b);
            let n1 = n - n0;
            if n0 >= 1.0 && n1 >= 1.0 {
                let (m0, m1) = (s0 / n0, (total - s0) / n1);
                let between = n0 * n1 * (m1 - m0) * (m1 - m0);
                if between > best.0 {
                    best = (between, b, n0 / n, m1 - m0);
                }
            }
        }
        // Two clear tones, each at least 5 % of the sky.
        let (_, cut, low_share, apart) = best;
        if n < 20.0 || apart < 0.15 || !(0.05..=0.95).contains(&low_share) {
            return None;
        }
        let split = (cut + 1) as f32 / BINS as f32;
        let mut class = vec![0u8; w * h];
        for (i, c) in class.iter_mut().enumerate().take(w * hz) {
            *c = if tone(img, i) > split { 2 } else { 1 };
        }
        // The clouds (4-connected): whether each touches the picture, its known pixels, its rows.
        let mut comp = vec![u32::MAX; w * hz];
        let mut clouds: Vec<(bool, usize, usize, usize)> = Vec::new();
        let mut stack = Vec::new();
        for start in 0..w * hz {
            if class[start] != 2 || comp[start] != u32::MAX {
                continue;
            }
            let id = clouds.len() as u32;
            let mut c = (false, 0usize, usize::MAX, 0usize);
            comp[start] = id;
            stack.push(start);
            while let Some(i) = stack.pop() {
                if px[i] != Px::Hole {
                    c.0 = true;
                    c.1 += 1;
                }
                let (x, y) = (i % w, i / w);
                (c.2, c.3) = (c.2.min(y), c.3.max(y));
                for (nx, ny) in [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)] {
                    if nx >= w || ny >= hz {
                        continue;
                    }
                    let j = ny * w + nx;
                    if class[j] == 2 && comp[j] == u32::MAX {
                        comp[j] = id;
                        stack.push(j);
                    }
                }
            }
            clouds.push(c);
        }
        // Small clouds of the picture: under 0.5 % of its known sky.
        let sky_px = (0..w * hz).filter(|i| px[*i] != Px::Hole).count().max(1);
        let small: Vec<(usize, usize)> = clouds.iter().filter(|c| c.0 && c.1 * 200 <= sky_px).map(|c| (c.2, c.3)).collect();
        let keep: Vec<bool> = clouds
            .iter()
            .map(|c| {
                // A new cloud stays among enough small ones within twice its height.
                c.0 || {
                    let m = 2 * (c.3 - c.2 + 1);
                    small.iter().filter(|s| s.1 + m >= c.2 && s.0 <= c.3 + m).count() >= Self::SCATTERED
                }
            })
            .collect();
        let (mut sum, mut count) = ([0f64; 3], 0f64);
        for i in (0..w * hz).filter(|i| px[*i] != Px::Hole && class[*i] == 1) {
            for (c, s) in sum.iter_mut().enumerate() {
                *s += f64::from(img[i * ch + c]);
            }
            count += 1.0;
        }
        let mean = sum.map(|s| (s / count.max(1.0)) as f32);
        for i in 0..w * hz {
            if px[i] == Px::Hole && class[i] == 2 && !keep[comp[i] as usize] {
                img[i * ch..i * ch + 3].copy_from_slice(&mean);
                class[i] = 1;
            }
        }
        Some(Self { class, w, h, split })
    }

    /// The layout for a level `w` by `h` with picture `img`: its sources' tones per 4 x 4 cell.
    pub(super) fn view(&self, img: &[f32], w: usize, h: usize, ch: usize) -> SkyView {
        let (cw, chh) = (w.div_ceil(4), h.div_ceil(4));
        let (mut cell, mut n) = (vec![0.0f32; cw * chh], vec![0u16; cw * chh]);
        for y in 0..h {
            for x in 0..w {
                let (k, i) = ((y / 4) * cw + x / 4, y * w + x);
                cell[k] += img[i * ch..i * ch + ch.min(3)].iter().sum::<f32>() / ch.min(3) as f32;
                n[k] += 1;
            }
        }
        for (c, n) in cell.iter_mut().zip(&n) {
            *c /= f32::from((*n).max(1));
        }
        SkyView { class: self.class.clone(), lw: self.w, lh: self.h, sx: self.w as f32 / w as f32, sy: self.h as f32 / h as f32, cell, cw, split: self.split }
    }
}

/// A [`SkyLayout`] on one level.
pub(super) struct SkyView {
    class: Vec<u8>,
    lw: usize,
    lh: usize,
    /// From this level to the layout's.
    sx: f32,
    sy: f32,
    /// Each 4 x 4 cell's mean tone, cells across.
    cell: Vec<f32>,
    cw: usize,
    split: f32,
}

impl SkyView {
    /// Whether target `t` may copy `s`: cloud into cloud, sky into sky; either along a cloud's
    /// edge, where its outline forms; anything below the horizon.
    #[inline]
    pub(super) fn allows(&self, t: (i32, i32), s: (i32, i32)) -> bool {
        let (cx, cy) = (((t.0.max(0) as f32 * self.sx) as usize).min(self.lw - 1), ((t.1.max(0) as f32 * self.sy) as usize).min(self.lh - 1));
        let c = self.class[cy * self.lw + cx];
        if c == 0 {
            return true;
        }
        let edge = (cy.saturating_sub(1)..(cy + 2).min(self.lh))
            .any(|y| (cx.saturating_sub(1)..(cx + 2).min(self.lw)).any(|x| !matches!(self.class[y * self.lw + x], 0) && self.class[y * self.lw + x] != c));
        if edge {
            return true;
        }
        let tone = self.cell.get((s.1.max(0) as usize / 4) * self.cw + s.0.max(0) as usize / 4).copied().unwrap_or(self.split);
        (tone > self.split) == (c == 2)
    }
}
