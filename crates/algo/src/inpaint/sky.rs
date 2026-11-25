//! Clouds at the hole's edge, for a nearby fill at the same depth: a cloud the edge cuts goes
//! on into the hole (it copies cloud there), and where the edge is sky the fill is sky, with no
//! new cloud copied in (a cloud's top grazing the border got a slab of cloud above it, copied up
//! from the cloud itself).
//!
//! Made to hold for any sky, not one picture: sky and cloud are told apart per picture by
//! brightness and colour strength together (a grey cloud, a hazy or a lake-reflected sky), each
//! hole pixel follows the part of the edge nearest it (not a fixed box, which reached past a cut
//! cloud), and an overcast or mostly cloudy sky, or one without a clear split, has no rule.

use super::common::Px;
use std::collections::VecDeque;

/// Sky and cloud are told apart when their means lie this many pooled standard deviations apart.
const SPLIT: f32 = 1.5;
/// No rule when cloud covers more than this share of the known sky (the sky's own gaps are too
/// small to copy from: they came out as streaks of blue), or less than [`MIN_CLOUD`].
const MAX_CLOUD: f32 = 0.6;
const MIN_CLOUD: f32 = 0.01;
/// The rule needs at least this many known sky pixels.
const MIN_SKY_PX: usize = 256;
/// Next to a cut cloud, the fill is cloud up to 1/this of the cut's length into the hole.
const DOME: u32 = 1;
/// A cloud is cut (and finished) when the edge runs across at least 1/this of its width; less
/// and it only grazes the border, and the fill beside it is sky.
const GRAZE: f32 = 4.0;

/// Which hole pixels copy only sky or only cloud, and which sources are cloud or lie by one.
pub(super) struct SkyRule {
    sky_only: Vec<bool>,
    near_cloud: Vec<bool>,
    /// Hole pixels finishing a cut cloud: they copy cloud.
    cloud_only: Vec<bool>,
    /// The known cloud.
    cloud: Vec<bool>,
}

impl SkyRule {
    /// From the known pixels above `horizon_at` (a row per column) of `img` (`w` × `h`, `ch`
    /// channels, RGB first), `px` the level's pixels, `r` the patch radius. `None` when the sky
    /// has no clear sky/cloud split, is mostly cloud, or no hole pixel is held to sky.
    pub(super) fn new(img: &[f32], px: &[Px], w: usize, h: usize, ch: usize, horizon_at: impl Fn(i32) -> i32, r: usize) -> Option<Self> {
        if ch < 3 || w == 0 || h == 0 {
            return None;
        }
        let top: Vec<usize> = (0..w).map(|x| horizon_at(x as i32).clamp(0, h as i32) as usize).collect();
        let above = |i: usize| i / w < top[i % w];
        let known = |i: usize| matches!(px[i], Px::Source | Px::Known);
        // Brightness and colour strength: a cloud is lighter or greyer than the sky beside it.
        let feat = |i: usize| {
            let (r, g, b) = (img[i * ch], img[i * ch + 1], img[i * ch + 2]);
            ((r + g + b) / 3.0, r.max(g).max(b) - r.min(g).min(b))
        };
        let sky_px: Vec<usize> = (0..w * h).filter(|&i| known(i) && above(i)).collect();
        if sky_px.len() < MIN_SKY_PX {
            return None;
        }
        // Each feature scaled by its spread, then two means (k-means, k = 2).
        let n = sky_px.len() as f32;
        let mean = sky_px.iter().fold((0.0, 0.0), |a, &i| {
            let f = feat(i);
            (a.0 + f.0 / n, a.1 + f.1 / n)
        });
        let var = sky_px.iter().fold((0.0, 0.0), |a, &i| {
            let f = feat(i);
            (a.0 + (f.0 - mean.0).powi(2) / n, a.1 + (f.1 - mean.1).powi(2) / n)
        });
        let sd = (var.0.sqrt().max(1e-4), var.1.sqrt().max(1e-4));
        let z = |i: usize| {
            let f = feat(i);
            ((f.0 - mean.0) / sd.0, (f.1 - mean.1) / sd.1)
        };
        // Cloudiness: lighter, greyer. Start the means at its two ends.
        let cloudiness = |p: (f32, f32)| p.0 - p.1;
        let (mut c0, mut c1) = ((-1.0f32, 1.0f32), (1.0f32, -1.0f32));
        let mut sizes = (0usize, 0usize);
        for _ in 0..12 {
            let (mut s0, mut s1) = ((0.0f32, 0.0f32, 0usize), (0.0f32, 0.0f32, 0usize));
            for &i in &sky_px {
                let p = z(i);
                let d0 = (p.0 - c0.0).powi(2) + (p.1 - c0.1).powi(2);
                let d1 = (p.0 - c1.0).powi(2) + (p.1 - c1.1).powi(2);
                let s = if d1 < d0 { &mut s1 } else { &mut s0 };
                s.0 += p.0;
                s.1 += p.1;
                s.2 += 1;
            }
            if s0.2 == 0 || s1.2 == 0 {
                return None;
            }
            c0 = (s0.0 / s0.2 as f32, s0.1 / s0.2 as f32);
            c1 = (s1.0 / s1.2 as f32, s1.1 / s1.2 as f32);
            sizes = (s0.2, s1.2);
        }
        if cloudiness(c0) > cloudiness(c1) {
            std::mem::swap(&mut c0, &mut c1);
            sizes = (sizes.1, sizes.0);
        }
        // Spread within the groups, against the distance between their means.
        let mut within = 0.0f32;
        for &i in &sky_px {
            let p = z(i);
            let d0 = (p.0 - c0.0).powi(2) + (p.1 - c0.1).powi(2);
            let d1 = (p.0 - c1.0).powi(2) + (p.1 - c1.1).powi(2);
            within += d0.min(d1);
        }
        let within = (within / n / 2.0).sqrt().max(1e-4);
        let apart = ((c1.0 - c0.0).powi(2) + (c1.1 - c0.1).powi(2)).sqrt();
        let share = sizes.1 as f32 / n;
        if apart < SPLIT * within || !(MIN_CLOUD..=MAX_CLOUD).contains(&share) {
            return None;
        }
        let is_cloud = |i: usize| {
            let p = z(i);
            (p.0 - c1.0).powi(2) + (p.1 - c1.1).powi(2) < (p.0 - c0.0).powi(2) + (p.1 - c0.1).powi(2)
        };
        let cloud: Vec<bool> = (0..w * h).map(|i| known(i) && above(i) && is_cloud(i)).collect();
        // The hole's edge: known pixels beside a hole pixel.
        let beside_hole = |i: usize| {
            let (x, y) = (i % w, i / w);
            (x > 0 && px[i - 1] == Px::Hole) || (x + 1 < w && px[i + 1] == Px::Hole) || (y > 0 && px[i - w] == Px::Hole) || (y + 1 < h && px[i + w] == Px::Hole)
        };
        let edge: Vec<bool> = (0..w * h).map(|i| known(i) && above(i) && beside_hole(i)).collect();
        // Each edge pixel's cloud share among the edge pixels within a few patches of it, so a
        // stray pixel doesn't decide, and a cut cloud counts across its whole cut.
        let k = 2 * (2 * r + 1);
        let mut sum = vec![(0u32, 0u32); (w + 1) * (h + 1)];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let (l, u, ul) = (sum[(y + 1) * (w + 1) + x], sum[y * (w + 1) + x + 1], sum[y * (w + 1) + x]);
                let e = (u32::from(edge[i]), u32::from(edge[i] && cloud[i]));
                sum[(y + 1) * (w + 1) + x + 1] = (l.0 + u.0 - ul.0 + e.0, l.1 + u.1 - ul.1 + e.1);
            }
        }
        let cloudy_edge = |i: usize| {
            let (x, y) = (i % w, i / w);
            let (x0, y0, x1, y1) = (x.saturating_sub(k), y.saturating_sub(k), (x + k + 1).min(w), (y + k + 1).min(h));
            let at = |x: usize, y: usize| sum[y * (w + 1) + x];
            let (a, b, c, d) = (at(x1, y1), at(x0, y1), at(x1, y0), at(x0, y0));
            let (e, cl) = (a.0 + d.0 - b.0 - c.0, a.1 + d.1 - b.1 - c.1);
            2 * cl >= e
        };
        // The runs of cloudy edge (8-connected): a cut cloud's run is about the length of its cut.
        const NO_RUN: u32 = u32::MAX;
        let mut run = vec![NO_RUN; w * h];
        let mut run_len: Vec<u32> = Vec::new();
        let cloudy: Vec<bool> = (0..w * h).map(|i| edge[i] && cloudy_edge(i)).collect();
        let mut stack = Vec::new();
        for i0 in 0..w * h {
            if !cloudy[i0] || run[i0] != NO_RUN {
                continue;
            }
            let id = run_len.len() as u32;
            let mut n = 0u32;
            run[i0] = id;
            stack.push(i0);
            while let Some(i) = stack.pop() {
                n += 1;
                let (x, y) = (i % w, i / w);
                for (dx, dy) in [(-1i32, -1i32), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if cloudy[j] && run[j] == NO_RUN {
                        run[j] = id;
                        stack.push(j);
                    }
                }
            }
            run_len.push(n);
        }
        // Each run's cloud: the known cloud it borders (4-connected), as wide as a disc of its area.
        let mut blob = vec![NO_RUN; w * h];
        let mut blob_size: Vec<u32> = Vec::new();
        for i0 in 0..w * h {
            if !cloud[i0] || blob[i0] != NO_RUN {
                continue;
            }
            let id = blob_size.len() as u32;
            let mut n = 0u32;
            blob[i0] = id;
            stack.push(i0);
            while let Some(i) = stack.pop() {
                n += 1;
                let (x, y) = (i % w, i / w);
                let mut go = |j: usize| {
                    if cloud[j] && blob[j] == NO_RUN {
                        blob[j] = id;
                        stack.push(j);
                    }
                };
                if x > 0 {
                    go(i - 1);
                }
                if x + 1 < w {
                    go(i + 1);
                }
                if y > 0 {
                    go(i - w);
                }
                if y + 1 < h {
                    go(i + w);
                }
            }
            blob_size.push(n);
        }
        let mut run_cloud = vec![0u32; run_len.len()];
        for (&ru, &bl) in run.iter().zip(&blob) {
            if ru != NO_RUN && bl != NO_RUN {
                let k = &mut run_cloud[ru as usize];
                *k = (*k).max(blob_size[bl as usize]);
            }
        }
        // A run that only grazes its cloud continues nothing.
        let cut: Vec<bool> = run_len.iter().zip(&run_cloud).map(|(&n, &a)| n as f32 * GRAZE >= 2.0 * (a as f32 / std::f32::consts::PI).sqrt()).collect();
        let cloudy: Vec<bool> = (0..w * h).map(|i| cloudy[i] && cut[run[i] as usize]).collect();
        // Each hole pixel takes the nearest edge pixel's call (breadth first through the hole).
        // Next to a cut cloud it copies cloud, within the cut's length of the edge ([`DOME`]),
        // so the cloud goes on past the border; farther in, sky.
        const UNSET: u8 = 0;
        let mut call = vec![UNSET; w * h];
        let mut from = vec![(NO_RUN, 0u32); w * h];
        let mut queue = VecDeque::new();
        for i in 0..w * h {
            if edge[i] {
                call[i] = if cloudy[i] { 2 } else { 1 };
                from[i] = (run[i], 0);
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            let (x, y) = (i % w, i / w);
            let (id, d) = from[i];
            let mut go = |j: usize| {
                if px[j] == Px::Hole && call[j] == UNSET {
                    let cloud_here = call[i] == 2 && id != NO_RUN && DOME * (d + 1) <= run_len[id as usize];
                    call[j] = if cloud_here { 2 } else { 1 };
                    from[j] = (id, d + 1);
                    queue.push_back(j);
                }
            };
            if x > 0 {
                go(i - 1);
            }
            if x + 1 < w {
                go(i + 1);
            }
            if y > 0 {
                go(i - w);
            }
            if y + 1 < h {
                go(i + w);
            }
        }
        let sky_only: Vec<bool> = (0..w * h).map(|i| px[i] == Px::Hole && above(i) && call[i] == 1).collect();
        if !sky_only.iter().any(|s| *s) {
            return None;
        }
        let near_cloud = super::levels::dilate_mask(&cloud, w, h, r + 1);
        let cloud_call: Vec<bool> = (0..w * h).map(|i| px[i] == Px::Hole && above(i) && call[i] == 2).collect();
        // Where a finished cloud meets the sky the fill is free a few patches each way, so the
        // cloud's outline is the picture's, not the line where the two calls meet.
        let m = 2 * (2 * r + 1);
        let by_cloud = super::levels::dilate_mask(&cloud_call, w, h, m);
        let by_sky = super::levels::dilate_mask(&sky_only, w, h, m);
        let sky_only: Vec<bool> = sky_only.iter().zip(&by_cloud).map(|(s, c)| *s && !*c).collect();
        let cloud_only: Vec<bool> = cloud_call.iter().zip(&by_sky).map(|(c, s)| *c && !*s).collect();
        Some(Self { sky_only, near_cloud, cloud_only, cloud })
    }

    /// Whether target `t` may copy source `s` (`w` across).
    #[inline]
    pub(super) fn allows(&self, t: (i32, i32), s: (i32, i32), w: usize) -> bool {
        let at = |p: (i32, i32)| p.1 as usize * w + p.0 as usize;
        let (t, s) = (at(t), at(s));
        if self.cloud_only.get(t).copied().unwrap_or(false) {
            return self.cloud.get(s).copied().unwrap_or(false);
        }
        !(self.sky_only.get(t).copied().unwrap_or(false) && self.near_cloud.get(s).copied().unwrap_or(false))
    }
}
