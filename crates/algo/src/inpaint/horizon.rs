//! The horizon of a picture, as a row per column, for a fill at the same depth.
//!
//! Found as a person traces it: the one line that runs across the whole picture where the
//! brightness steps and the smooth region above meets the busier one below. Not by colour: a
//! lake, haze, a grey sky or a blue-grey ridge all fool a "bluer is sky" rule. Per strip of
//! columns each row is scored, and the best path across the strips with only small steps
//! between them is kept, so a cloud's edge, which ends, loses to the line that runs on. Each
//! strip's row is then set to the pixel, and the line is carried into the hole: between known
//! strips straight across, past the last on its own slope there (or along a line the user drew).

use crate::par;

/// A line the user drew (the Ruler tool's): the horizon is looked for close to it, and follows
/// it where the picture has no pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hint {
    pub from: [f32; 2],
    pub to: [f32; 2],
}

impl Hint {
    /// The line's row at column `x` (extended past its ends).
    pub fn row_at(&self, x: f32) -> f32 {
        let (dx, dy) = (self.to[0] - self.from[0], self.to[1] - self.from[1]);
        if dx.abs() < 1e-3 { (self.from[1] + self.to[1]) / 2.0 } else { self.from[1] + (x - self.from[0]) * dy / dx }
    }
}

/// The side, in cells, the picture is measured on (its longer side shrunk to about this).
const GRID: usize = 512;
/// Strips across the picture.
const STRIPS: usize = 48;
/// The least mean score along the line for a horizon without a hint (a picture with no line
/// across it, a close-up, has none).
const MIN_SCORE: f32 = 0.5;
/// The cost of a row's step between strips, in score.
const STEP_COST: f32 = 0.08;
/// The steepest the line is carried past the last known strip, rows per column.
const MAX_SLOPE: f32 = 0.15;
/// The step's weight (signed: brighter above).
const STEP_SIGN: f32 = 1.0;
/// The weight of "busier below".
const CALM: f32 = 0.25;

/// The horizon of the `w × h` picture `img` (`ch` channels, the first `cc` colour) from the
/// pixels outside `hole`, a row per column; near `hint` when given. `None` without a clear one
/// (or, with a hint, never: then the hint's line where nothing better is found).
pub fn find(img: &[f32], hole: &[bool], w: usize, h: usize, ch: usize, cc: usize, hint: Option<Hint>) -> Option<Vec<f32>> {
    if w == 0 || h < 8 || ch == 0 || img.len() < w * h * ch || hole.len() < w * h {
        return hint.map(|hn| (0..w).map(|x| hn.row_at(x as f32 + 0.5).clamp(0.0, h as f32)).collect());
    }
    let cc = cc.clamp(1, ch);
    let lum = |i: usize| img[i * ch..i * ch + cc].iter().sum::<f32>() / cc as f32;
    // Cells: known pixels, their brightness, and how busy they are (steps to the right and down).
    let f = w.max(h).div_ceil(GRID).max(1);
    let (gw, gh) = (w.div_ceil(f), h.div_ceil(f));
    let cells: Vec<Vec<(u32, f32, f32, u32)>> = par::map(gh, w * h >= 1 << 20, |gy| {
        let mut row = vec![(0u32, 0f32, 0f32, 0u32); gw];
        for y in gy * f..((gy + 1) * f).min(h) {
            for x in 0..w {
                let i = y * w + x;
                if hole[i] {
                    continue;
                }
                let c = &mut row[x / f];
                let l = lum(i);
                c.0 += 1;
                c.1 += l;
                if x + 1 < w && !hole[i + 1] {
                    c.2 += (l - lum(i + 1)).abs();
                    c.3 += 1;
                }
                if y + 1 < h && !hole[i + w] {
                    c.2 += (l - lum(i + w)).abs();
                    c.3 += 1;
                }
            }
        }
        row
    });
    let full = (f * f) as u32;
    let cell = |gx: usize, gy: usize| {
        let c = cells[gy][gx];
        (c.0 * 2 >= full && c.3 > 0).then(|| (c.1 / c.0 as f32, c.2 / c.3 as f32))
    };
    // Scales: the spread of brightness and the mean busyness, over the known cells.
    let known: Vec<(f32, f32)> = (0..gh).flat_map(|gy| (0..gw).filter_map(move |gx| cell(gx, gy))).collect();
    if known.len() < 16 {
        return hint.map(|hn| (0..w).map(|x| hn.row_at(x as f32 + 0.5).clamp(0.0, h as f32)).collect());
    }
    let n = known.len() as f32;
    let mean_l = known.iter().map(|c| c.0).sum::<f32>() / n;
    let sigma = (known.iter().map(|c| (c.0 - mean_l).powi(2)).sum::<f32>() / n).sqrt().max(1e-4);
    let busy = (known.iter().map(|c| c.1).sum::<f32>() / n).max(1e-5);
    // Per strip, per row: mean brightness and busyness of its known cells (None: under half known).
    let sw = gw.div_ceil(STRIPS).max(1);
    let strips = gw.div_ceil(sw);
    let profile: Vec<Vec<Option<(f32, f32)>>> = (0..strips)
        .map(|s| {
            let (x0, x1) = (s * sw, ((s + 1) * sw).min(gw));
            (0..gh)
                .map(|gy| {
                    let got: Vec<(f32, f32)> = (x0..x1).filter_map(|gx| cell(gx, gy)).collect();
                    (got.len() * 2 >= x1 - x0).then(|| {
                        let k = got.len() as f32;
                        (got.iter().map(|c| c.0).sum::<f32>() / k, got.iter().map(|c| c.1).sum::<f32>() / k)
                    })
                })
                .collect()
        })
        .collect();
    // The score of the line between rows gy - 1 and gy in a strip: the step in brightness right
    // there, and how much busier the picture is below than above it.
    let (k, kk) = ((gh / 48).max(2), (gh / 6).max(4));
    let mean = |p: &[Option<(f32, f32)>], a: usize, b: usize, which: usize| -> Option<f32> {
        let v: Vec<f32> = p[a..b.min(p.len())].iter().flatten().map(|c| if which == 0 { c.0 } else { c.1 }).collect();
        (v.len() * 2 >= (b.min(p.len()) - a).max(1)).then(|| v.iter().sum::<f32>() / v.len() as f32)
    };
    let hint_row = |s: usize| hint.map(|hn| hn.row_at(((s * sw) as f32 + sw as f32 / 2.0) * f as f32) / f as f32);
    let band = (gh / 20).max(3) as f32;
    let score: Vec<Vec<Option<f32>>> = profile
        .iter()
        .enumerate()
        .map(|(s, p)| {
            (0..gh)
                .map(|gy| {
                    if gy < k || gy + k > gh {
                        return None;
                    }
                    if hint_row(s).is_some_and(|r| (gy as f32 - r).abs() > band) {
                        return None;
                    }
                    let step = (mean(p, gy - k, gy, 0)? - mean(p, gy, gy + k, 0)?) / sigma * STEP_SIGN;
                    let (up, down) = (mean(p, gy.saturating_sub(kk), gy, 1), mean(p, gy, gy + kk, 1));
                    let calm = match (up, down) {
                        (Some(u), Some(d)) => ((d - u) / busy).clamp(-2.0, 2.0) * CALM,
                        _ => 0.0,
                    };
                    Some(step + calm)
                })
                .collect()
        })
        .collect();
    let has = |s: usize| score[s].iter().any(Option::is_some);
    if !(0..strips).any(has) {
        return hint.map(|hn| (0..w).map(|x| hn.row_at(x as f32 + 0.5).clamp(0.0, h as f32)).collect());
    }
    // The best path across: each strip steps at most `m` rows from the last, at a cost per row.
    let m = ((sw as f32 * MAX_SLOPE * 2.0).round() as usize).max(1);
    let mut best = vec![0f32; gh];
    let mut back: Vec<Vec<u32>> = Vec::with_capacity(strips);
    for (s, sc) in score.iter().enumerate() {
        let mut next = vec![f32::NEG_INFINITY; gh];
        let mut from = vec![0u32; gh];
        for gy in 0..gh {
            // Out of the hint's band nothing goes (with data in the strip); elsewhere a row with
            // no score scores nothing.
            let here = match sc[gy] {
                Some(v) => v,
                None if has(s) && hint.is_some() => continue,
                None => 0.0,
            };
            for (py, &b) in best.iter().enumerate().take((gy + m + 1).min(gh)).skip(gy.saturating_sub(m)) {
                let v = if s == 0 { 0.0 } else { b } - STEP_COST * (py as f32 - gy as f32).abs();
                if v + here > next[gy] {
                    next[gy] = v + here;
                    from[gy] = py as u32;
                }
            }
        }
        best = next;
        back.push(from);
    }
    let mut gy = (0..gh).max_by(|a, b| best[*a].total_cmp(&best[*b]))?;
    let mut path = vec![0usize; strips];
    for s in (0..strips).rev() {
        path[s] = gy;
        gy = back[s][gy] as usize;
    }
    // How sure: the mean score along the path, over the strips with data.
    let along: Vec<f32> = (0..strips).filter_map(|s| score[s][path[s]]).collect();
    let sure = !along.is_empty() && along.iter().sum::<f32>() / along.len() as f32 >= MIN_SCORE;
    if !sure && hint.is_none() {
        return None;
    }
    // Each known strip's row set to the pixel: the sharpest brightness step within a cell and a
    // half of the path's row, over its known pixels.
    let k2 = (f / 2).max(2);
    let rows: Vec<Option<f32>> = (0..strips)
        .map(|s| {
            score[s][path[s]]?;
            if !sure {
                return None;
            }
            let (x0, x1) = (s * sw * f, ((s + 1) * sw * f).min(w));
            let step = ((x1 - x0) / 256).max(1);
            let c = path[s] * f;
            let (y0, y1) = (c.saturating_sub(f * 3 / 2 + k2), (c + f * 3 / 2 + k2).min(h));
            let line: Vec<Option<f32>> = (y0..y1)
                .map(|y| {
                    let v: Vec<f32> = (x0..x1).step_by(step).map(|x| y * w + x).filter(|&i| !hole[i]).map(lum).collect();
                    (v.len() * 2 >= (x1 - x0).div_ceil(step)).then(|| v.iter().sum::<f32>() / v.len() as f32)
                })
                .collect();
            let avg = |a: usize, b: usize| {
                let v: Vec<f32> = line[a..b].iter().flatten().copied().collect();
                (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32)
            };
            let at = (k2..line.len().saturating_sub(k2))
                .filter_map(|j| Some((j, (avg(j - k2, j)? - avg(j, j + k2)?).abs())))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(c, |(j, _)| y0 + j);
            Some(at as f32)
        })
        .collect();
    // A stray strip (a ridge's peak, a tree) takes its neighbours' middle.
    let known_s: Vec<usize> = (0..strips).filter(|s| rows[*s].is_some()).collect();
    let mut fixed: Vec<(f32, f32)> = known_s
        .iter()
        .enumerate()
        .map(|(j, &s)| {
            let r = |j: usize| rows[known_s[j]].unwrap_or(0.0);
            let mut t = [r(j.saturating_sub(1)), r(j), r((j + 1).min(known_s.len() - 1))];
            t.sort_by(f32::total_cmp);
            (((s * sw) as f32 + sw as f32 / 2.0) * f as f32, t[1])
        })
        .collect();
    let line_at = |x: f32| hint.map(|hn| hn.row_at(x));
    if fixed.is_empty() {
        // Nothing clear: the hint's line as drawn.
        return Some((0..w).map(|x| line_at(x as f32 + 0.5).unwrap_or(0.0).clamp(0.0, h as f32)).collect());
    }
    fixed.dedup_by(|a, b| (a.0 - b.0).abs() < 1.0);
    // Past the ends: along the hint (as far off it as the last known strip), or on the slope
    // of the last few known strips.
    let slope_at = |end: &[(f32, f32)]| -> f32 {
        if end.len() < 2 {
            return 0.0;
        }
        let n = end.len() as f32;
        let (mx, my) = (end.iter().map(|p| p.0).sum::<f32>() / n, end.iter().map(|p| p.1).sum::<f32>() / n);
        let (sxy, sxx) = end.iter().fold((0f32, 0f32), |(a, b), p| (a + (p.0 - mx) * (p.1 - my), b + (p.0 - mx).powi(2)));
        if sxx > 0.0 { (sxy / sxx).clamp(-MAX_SLOPE, MAX_SLOPE) } else { 0.0 }
    };
    let nfit = fixed.len().min(6);
    let (first, last) = (fixed[0], fixed[fixed.len() - 1]);
    let (lead, trail) = (slope_at(&fixed[..nfit]), slope_at(&fixed[fixed.len() - nfit..]));
    let outside = |x: f32, p: (f32, f32), slope: f32| match line_at(x) {
        Some(r) => r + p.1 - line_at(p.0).unwrap_or(r),
        None => p.1 + (x - p.0) * slope,
    };
    let raw: Vec<f32> = (0..w)
        .map(|x| {
            let x = x as f32 + 0.5;
            let r = if x <= first.0 {
                outside(x, first, lead)
            } else if x >= last.0 {
                outside(x, last, trail)
            } else {
                let j = fixed.partition_point(|p| p.0 <= x);
                let (a, b) = (fixed[j - 1], fixed[j]);
                a.1 + (b.1 - a.1) * (x - a.0) / (b.0 - a.0)
            };
            r.clamp(0.0, h as f32)
        })
        .collect();
    // Smoothed over half a strip each way: corners between strips rounded off.
    let rad = (sw * f / 2).max(1);
    let mut pre = vec![0f64; w + 1];
    for (x, r) in raw.iter().enumerate() {
        pre[x + 1] = pre[x] + f64::from(*r);
    }
    Some((0..w).map(|x| ((pre[(x + rad + 1).min(w)] - pre[x.saturating_sub(rad)]) / ((x + rad + 1).min(w) - x.saturating_sub(rad)) as f64) as f32).collect())
}
