//! The target cells of a pyramid level: the centres whose patch overlaps the hole, kept as runs
//! per row and numbered in raster order. The per-target tables of the solver (the match field,
//! its costs, the patches' moments, the vote's weights) hold one entry per target, not one per
//! cell of the hole's bounds: a fill area along an image's borders has bounds as large as the
//! image, and a table per cell of them took most of a 156 MP fill's 18 GB.

use super::common::{Bounds, PAR_MIN, Px};
use super::par;

/// A run of target cells in one row: columns `x0..x1`, the first one numbered `k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Span {
    pub(super) x0: usize,
    pub(super) x1: usize,
    pub(super) k: usize,
}

impl Span {
    /// The number of cell `x` of this run (`x0 <= x < x1`).
    #[inline]
    pub(super) fn at(&self, x: usize) -> usize {
        self.k + x - self.x0
    }
}

/// The target cells of a level, all within `tbox` (the hole's bounds grown by the patch radius).
pub(super) struct Cells {
    pub(super) tbox: Bounds,
    /// Per row of the box, its first run in `spans` (`height + 1` entries).
    first: Vec<usize>,
    spans: Vec<Span>,
}

impl Cells {
    /// The cells of `tb` within `r` (horizontally and vertically) of a hole pixel of a `w`-wide
    /// role grid, whose hole pixels all lie within `hb`.
    pub(super) fn around(w: usize, r: usize, px: &[Px], hb: Bounds, tb: Bounds) -> Cells {
        // Per row of the hole's bounds, its runs of hole pixels.
        let runs: Vec<Vec<(usize, usize)>> = par::map(hb.height(), hb.area() >= PAR_MIN, |ry| {
            let y = hb.y0 + ry;
            let row = &px[y * w + hb.x0..y * w + hb.x1];
            let mut out = Vec::new();
            let mut x = 0;
            while x < row.len() {
                if row[x] != Px::Hole {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < row.len() && row[x] == Px::Hole {
                    x += 1;
                }
                out.push((hb.x0 + start, hb.x0 + x));
            }
            out
        });
        // Per row of the box: the runs of the hole rows within `r`, each grown by `r`, merged.
        let rows: Vec<Vec<(usize, usize)>> = par::map(tb.height(), tb.area() >= PAR_MIN, |ty| {
            let y = tb.y0 + ty;
            let (lo, hi) = (y.saturating_sub(r).max(hb.y0), y.saturating_add(r + 1).min(hb.y1));
            let mut near: Vec<(usize, usize)> =
                (lo..hi).flat_map(|yy| runs[yy - hb.y0].iter().map(|&(a, b)| (a.saturating_sub(r).max(tb.x0), b.saturating_add(r).min(tb.x1)))).collect();
            near.sort_unstable();
            let mut out: Vec<(usize, usize)> = Vec::new();
            for (a, b) in near {
                match out.last_mut() {
                    Some(last) if a <= last.1 => last.1 = last.1.max(b),
                    _ if a < b => out.push((a, b)),
                    _ => {}
                }
            }
            out
        });
        let mut first = Vec::with_capacity(rows.len() + 1);
        let mut spans = Vec::new();
        let mut k = 0;
        for row in rows {
            first.push(spans.len());
            for (x0, x1) in row {
                spans.push(Span { x0, x1, k });
                k += x1 - x0;
            }
        }
        first.push(spans.len());
        Cells { tbox: tb, first, spans }
    }

    /// How many cells there are.
    pub(super) fn len(&self) -> usize {
        self.spans.last().map_or(0, |s| s.k + s.x1 - s.x0)
    }

    /// Row `y`'s runs, left to right (none outside the box).
    #[inline]
    pub(super) fn row(&self, y: usize) -> &[Span] {
        if y < self.tbox.y0 || y >= self.tbox.y1 {
            return &[];
        }
        let ry = y - self.tbox.y0;
        &self.spans[self.first[ry]..self.first[ry + 1]]
    }

    /// The cells in the rows before row `y` (clamped to the box).
    pub(super) fn before_row(&self, y: usize) -> usize {
        let ry = y.clamp(self.tbox.y0, self.tbox.y1) - self.tbox.y0;
        self.spans.get(self.first[ry]).map_or(self.len(), |s| s.k)
    }

    /// The number of cell `(x, y)`, if it is one.
    #[inline]
    pub(super) fn index(&self, x: usize, y: usize) -> Option<usize> {
        let row = self.row(y);
        let i = row.partition_point(|s| s.x1 <= x);
        row.get(i).filter(|s| s.x0 <= x).map(|s| s.at(x))
    }

    /// `(number, x, y)` of every cell, in raster order.
    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        (self.tbox.y0..self.tbox.y1).flat_map(move |y| self.row(y).iter().flat_map(move |s| (s.x0..s.x1).map(move |x| (s.at(x), x, y))))
    }

    /// `f(y, runs, part)` for each row of the box, `part` the row's share of `buf` (which holds
    /// `per` entries per cell, in cell order: a cell's are at `(number − runs[0].k) · per` in
    /// it), in parallel when `par`. Rows without cells are skipped.
    pub(super) fn rows_mut<T: Send>(&self, buf: &mut [T], per: usize, par: bool, f: impl Fn(usize, &[Span], &mut [T]) + Sync + Send) {
        let mut parts = Vec::new();
        let mut rest = &mut buf[..self.len() * per];
        for y in self.tbox.y0..self.tbox.y1 {
            let n = (self.before_row(y + 1) - self.before_row(y)) * per;
            if n == 0 {
                continue;
            }
            let (part, tail) = std::mem::take(&mut rest).split_at_mut(n);
            parts.push((y, part));
            rest = tail;
        }
        par::each(parts, par, |(y, part)| f(y, self.row(y), part));
    }

    /// The cells of row `y` before column `x` (`before`) or after it (`!before`).
    pub(super) fn in_row(&self, y: usize, x: usize, before: bool) -> usize {
        self.row(y).iter().map(|s| if before { x.clamp(s.x0, s.x1) - s.x0 } else { s.x1 - x.saturating_add(1).clamp(s.x0, s.x1) }).sum()
    }
}
