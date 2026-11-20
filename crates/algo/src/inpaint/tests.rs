use super::*;

/// The mean is matched first: on a nearly flat bright patch with a noisy deviation ratio the
/// gain stays where the bias can still reach the target's mean.
#[test]
fn fit_matches_the_mean_before_the_contrast() {
    let a = Adapt { gain: (0.9, 1.1), bias: 0.04 };
    // Target mean 0.70, source mean 0.70; deviations 0.01 vs 0.004 (ratio 2.5).
    let n = 25.0;
    let (mt, ms, sdt, sds) = (0.70f32, 0.70f32, 0.01f32, 0.004f32);
    let (g, b) = fit(a, n, mt * n, (sdt * sdt + mt * mt) * n, ms * n, (sds * sds + ms * ms) * n);
    assert!((g * ms + b - mt).abs() < 1e-5, "mean {} vs {mt} (g {g}, b {b})", g * ms + b);
    // Where the mean can't be reached anyway, the limits hold as before.
    let (g, b) = fit(a, n, 0.2 * n, 0.04 * n, 0.9 * n, 0.81 * n);
    assert!((a.gain.0..=a.gain.1).contains(&g) && b.abs() <= a.bias + 1e-6);
}

/// A zero-width image has no pixels however tall it is claimed to be: no per-row table.
#[test]
fn zero_width_images_complete_without_sizing_by_height() {
    let p = CompleteParams::default();
    assert_eq!(complete(0, 1 << 40, 3, &[], &[], &p), Some(Vec::new()));
    assert_eq!(complete(1 << 40, 0, 3, &[], &[], &p), Some(Vec::new()));
}

/// Dimensions whose product wraps must be refused, not matched against tiny buffers.
#[test]
#[should_panic]
fn wrapping_dimension_products_are_refused() {
    let _ = complete(1 << 40, 1 << 40, 1, &[], &[], &CompleteParams::default());
}

/// A block size beyond `i32` falls back to the membrane fill instead of looping on a zero step.
#[test]
fn a_huge_synthesis_block_does_not_hang() {
    let (w, h) = (8, 8);
    let img = vec![0.5f32; w * h];
    let mut hole = vec![false; w * h];
    hole[27] = true;
    let out = synthesize(w, h, 1, &img, &hole, usize::MAX, 1);
    assert_eq!(out.len(), w * h);
}

/// Smooth gradient plus deterministic fine noise.
fn texture(w: usize, h: usize) -> Vec<f32> {
    let mut rng = Rng::new(7);
    (0..w * h)
        .flat_map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let n = (rng.next() % 1000) as f32 / 1000.0 * 0.04 - 0.02;
            [0.3 + 0.004 * x + n, 0.5 + 0.002 * y + n, 0.4 + n]
        })
        .collect()
}

fn disc(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Vec<bool> {
    (0..w * h).map(|i| ((i % w) as f32 - cx).hypot((i / w) as f32 - cy) < r).collect()
}

fn hole_rmse(a: &[f32], b: &[f32], hole: &[bool], ch: usize) -> f32 {
    let (mut e, mut n) = (0.0f32, 0);
    for (i, hh) in hole.iter().enumerate() {
        if *hh {
            for c in 0..ch {
                e += (a[i * ch + c] - b[i * ch + c]).powi(2);
                n += 1;
            }
        }
    }
    (e / n as f32).sqrt()
}

#[test]
fn completion_fills_blemish_from_texture() {
    let (w, h, ch) = (96, 96, 3);
    let clean = texture(w, h);
    let hole = disc(w, h, 48.0, 48.0, 9.0);
    let mut dirty = clean.clone();
    for (i, hh) in hole.iter().enumerate() {
        if *hh {
            dirty[i * ch..i * ch + ch].copy_from_slice(&[0.02, 0.02, 0.02]);
        }
    }
    let out = complete(w, h, ch, &dirty, &hole, &CompleteParams::default()).unwrap();
    let before = hole_rmse(&dirty, &clean, &hole, ch);
    let after = hole_rmse(&out, &clean, &hole, ch);
    assert!(after < 0.06 && after < before * 0.15, "rmse before {before} after {after}");
    // Outside the hole untouched.
    assert!(out.iter().zip(&dirty).zip(hole.iter().flat_map(|h| [*h; 3])).all(|((a, b), h)| h || a == b));
    // Deterministic.
    assert_eq!(out, complete(w, h, ch, &dirty, &hole, &CompleteParams::default()).unwrap());
}

#[test]
fn completion_continues_structure() {
    // Vertical stripes: the filled hole should keep the stripes (not a blur).
    let (w, h, ch) = (80, 80, 1);
    let clean: Vec<f32> = (0..w * h).map(|i| if (i % w / 4) % 2 == 0 { 0.2 } else { 0.8 }).collect();
    let hole = disc(w, h, 40.0, 40.0, 7.0);
    let dirty: Vec<f32> = clean.iter().zip(&hole).map(|(c, h)| if *h { 0.5 } else { *c }).collect();
    let out = complete(w, h, ch, &dirty, &hole, &CompleteParams::default()).unwrap();
    let err = hole_rmse(&out, &clean, &hole, ch);
    assert!(err < 0.15, "{err}");
}

#[test]
fn proximity_finds_matching_offset() {
    let (w, h, ch) = (80, 60, 1);
    // Periodic pattern with period 10 in x: a match is any multiple-of-10 displacement.
    let img: Vec<f32> = (0..w * h).map(|i| ((i % w) as f32 * std::f32::consts::TAU / 10.0).sin() * 0.4 + 0.5).collect();
    let hole = disc(w, h, 40.0, 30.0, 5.0);
    let (dx, dy) = best_offset(w, h, ch, &img, &hole, 3, 20).unwrap();
    assert_eq!(dx.rem_euclid(10), 0, "dx {dx} dy {dy}");
    assert!((dx, dy) != (0, 0));
}

#[test]
fn proximity_search_clamps_hostile_radii_without_overflow_or_hanging() {
    let (w, h, ch) = (24, 20, 1);
    let img: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
    let hole = disc(w, h, 12.0, 10.0, 2.0);
    // User-supplied values used to overflow i32 loops, cast a huge usize
    // ring into a negative number, or take billions of search iterations.
    let result = best_offset(w, h, ch, &img, &hole, usize::MAX, i32::MAX);
    if let Some((dx, dy)) = result {
        assert!(dx.abs() <= w as i32 && dy.abs() <= h as i32);
    }
    assert_eq!(best_offset(w, h, ch, &img, &hole, 2, -1), None);
    assert_eq!(best_offset(w, h, ch, &img, &hole, 2, 0), None);
}

/// Structure, fine noise and flat patches (so that some matches tie exactly), `ch` channels.
fn busy(w: usize, h: usize, ch: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let flat = (x / 9 + y / 7) % 5 == 0;
            let n = (rng.next() % 1000) as f32 / 1000.0;
            (0..ch).map(move |c| if flat { 0.25 + 0.1 * c as f32 } else { (0.2 + 0.004 * x as f32 + 0.1 * ((x + 2 * y + c) % 6) as f32 + 0.05 * n).min(1.0) })
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Roles: a hole, a band that is context only, a corner that is no image.
fn roles(w: usize, h: usize, hole: &[bool]) -> Vec<Px> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            if hole[i] {
                Px::Hole
            } else if x < 4 && y < 6 {
                Px::Void
            } else if (w * 3 / 4..w * 3 / 4 + 3).contains(&x) {
                Px::Known
            } else {
                Px::Source
            }
        })
        .collect()
}

#[test]
fn a_hole_is_as_thick_as_it_is_through_not_along() {
    let (w, h) = (240, 120);
    let all = Bounds { x0: 0, y0: 0, x1: w, y1: h };
    let thickness = |hole: &[bool]| {
        let px: Vec<Px> = hole.iter().map(|&o| if o { Px::Hole } else { Px::Source }).collect();
        hole_thickness(w, hole_bounds(w, h, &px).unwrap_or(all), &px)
    };
    // A band 200 px long and 9 px thick; a disc of radius 20; the band touching the edge
    // counts the outside of the image as not hole.
    let band: Vec<bool> = (0..w * h).map(|i| (20..220).contains(&(i % w)) && (50..59).contains(&(i / w))).collect();
    assert!((9..=11).contains(&thickness(&band)), "{}", thickness(&band));
    assert!((39..=42).contains(&thickness(&disc(w, h, 120.0, 60.0, 20.0))));
    let edge: Vec<bool> = (0..w * h).map(|i| i / w < 9).collect();
    assert!((9..=11).contains(&thickness(&edge)), "{}", thickness(&edge));
}

/// A long thin hole goes the thin way (a shallow pyramid that starts from a membrane): it is
/// filled, only it changes, and the result does not depend on the thread count either.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn thin_holes_are_filled_the_same_on_any_number_of_threads() {
    let (w, h) = (300, 200);
    let img = busy(w, h, 3, 5);
    // A wavy line 6 px thick across the image.
    let hole: Vec<bool> = (0..w * h).map(|i| (((i % w) as f32 * 0.05).sin() * 30.0 + 100.0 - (i / w) as f32).abs() < 3.0).collect();
    let px = roles(w, h, &hole);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| complete_masked(w, h, 3, &img, &px, &CompleteParams::default(), &photocraft_raster::Interrupt::NONE)).unwrap().unwrap()
    };
    let one = run(1);
    assert_eq!(bits(&one), bits(&run(7)));
    assert!((0..w * h).all(|i| hole[i] || (0..3).all(|c| one[i * 3 + c].to_bits() == img[i * 3 + c].to_bits())));
    assert!((0..w * h).filter(|&i| hole[i]).all(|i| (0..3).all(|c| one[i * 3 + c].is_finite())));
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn results_do_not_depend_on_the_thread_count() {
    // Big enough for the parallel passes, and for PatchMatch to sweep in wavefronts where
    // there are fewer bands than threads (never with one thread).
    let (w, h) = (300, 260);
    for (ch, adapt) in [(3, Adapt::NONE), (4, Adapt { gain: (0.9, 1.1), bias: 0.04 })] {
        let img = busy(w, h, ch, 3);
        let hole = disc(w, h, 140.0, 130.0, 40.0);
        let px = roles(w, h, &hole);
        let p = CompleteParams { adapt, ..Default::default() };
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            bits(&pool.install(|| complete_masked(w, h, ch, &img, &px, &p, &photocraft_raster::Interrupt::NONE)).unwrap().unwrap())
        };
        let one = run(1);
        assert_eq!(one, run(3), "{ch} channels, 3 threads");
        assert_eq!(one, run(16), "{ch} channels, 16 threads");
    }
}

/// The onion peel as first written: every source in order, every known pixel summed.
fn onion_peel_plain(w: usize, h: usize, ch: usize, r: i32, img: &mut [f32], px: &[Px], sources: &[(i32, i32)]) {
    let mut known: Vec<bool> = px.iter().map(|k| matches!(k, Px::Known | Px::Source)).collect();
    loop {
        let front: Vec<usize> = (0..w * h)
            .filter(|&i| {
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                !known[i]
                    && px[i] == Px::Hole
                    && (-1..=1).any(|dy: i32| {
                        (-1..=1).any(|dx: i32| {
                            let (nx, ny) = (x + dx, y + dy);
                            nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h && known[ny as usize * w + nx as usize]
                        })
                    })
            })
            .collect();
        if front.is_empty() {
            break;
        }
        let vals: Vec<Vec<f32>> = front
            .iter()
            .map(|&i| {
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                let mut best = (f32::INFINITY, (0i32, 0i32));
                for &(sx, sy) in sources {
                    let (mut e, mut n) = (0.0f32, 0usize);
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let (tx, ty) = (x + dx, y + dy);
                            if tx < 0 || ty < 0 || tx as usize >= w || ty as usize >= h || !known[ty as usize * w + tx as usize] {
                                continue;
                            }
                            let (ti, si) = ((ty as usize * w + tx as usize) * ch, ((sy + dy) as usize * w + (sx + dx) as usize) * ch);
                            for c in 0..ch {
                                let d = img[ti + c] - img[si + c];
                                e += d * d;
                            }
                            n += 1;
                        }
                    }
                    if n > 0 && e / (n as f32) < best.0 {
                        best = (e / n as f32, (sx, sy));
                    }
                }
                let si = (best.1.1 as usize * w + best.1.0 as usize) * ch;
                img[si..si + ch].to_vec()
            })
            .collect();
        for (&i, v) in front.iter().zip(vals) {
            img[i * ch..(i + 1) * ch].copy_from_slice(&v);
            known[i] = true;
        }
    }
}

/// #2805's case (upstream's own speed-up of the onion peel, superseded here): repeating
/// texture with many exact ties, two holes; the result must be the plain scan's to the bit.
#[test]
fn onion_peel_picks_exactly_what_the_plain_scan_picks() {
    let (w, h, ch, r) = (48, 40, 3, 3);
    let mut img = texture(w, h);
    for y in 0..h {
        for x in 6..w / 2 {
            for c in 0..ch {
                img[(y * w + x) * ch + c] = img[(y * w + x % 6) * ch + c];
            }
        }
    }
    let none = photocraft_raster::Interrupt::NONE;
    for hole in [disc(w, h, 20.0, 18.0, 7.5), disc(w, h, 34.0, 22.0, 5.0)] {
        let px: Vec<Px> = hole.iter().map(|h| if *h { Px::Hole } else { Px::Source }).collect();
        let near = integral(w, h, &hole);
        let sources: Vec<(i32, i32)> = (r..h as i32 - r)
            .flat_map(|y| (r..w as i32 - r).map(move |x| (x, y)))
            .filter(|&(x, y)| window_count(&near, w, h, x - r, y - r, x + r + 1, y + r + 1) == 0)
            .collect();
        let (mut fast, mut plain) = (img.clone(), img.clone());
        onion_peel(w, h, ch, r, &mut fast, &px, hole_bounds(w, h, &px).unwrap(), &sources, &none).unwrap();
        onion_peel_plain(w, h, ch, r, &mut plain, &px, &sources);
        assert!(fast.iter().zip(&plain).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_ne!(fast, img);
    }
}

#[test]
fn onion_peel_takes_the_plain_scans_sources() {
    let none = photocraft_raster::Interrupt::NONE;
    for (w, h, ch, seed) in [(44, 36, 1, 1), (50, 40, 3, 2), (40, 40, 4, 3)] {
        let mut img = busy(w, h, ch, seed);
        // Also an image of a few exact values, where many sources tie.
        if seed == 3 {
            img.iter_mut().for_each(|v| *v = (*v * 3.0).round() / 3.0);
        }
        let hole = disc(w, h, w as f32 * 0.45, h as f32 * 0.5, 7.5);
        let px = roles(w, h, &hole);
        let valid = valid_sources(w, h, 3, &px, &none).unwrap();
        let sources = Sources::new(&valid, w).all(&valid, w);
        let (mut a, mut b) = (img.clone(), img.clone());
        onion_peel(w, h, ch, 3, &mut a, &px, hole_bounds(w, h, &px).unwrap(), &sources, &none).unwrap();
        onion_peel_plain(w, h, ch, 3, &mut b, &px, &sources);
        assert_eq!(bits(&a), bits(&b), "{w}×{h}×{ch}");
    }
}

#[test]
fn least_sum_is_where_the_quotient_crosses() {
    let mut rng = Rng::new(5);
    for i in 0..4000 {
        let n = (1 + rng.next() % 49) as f32;
        let v = match i % 4 {
            0 => 0.0,
            1 => (rng.next() % 1000) as f32 * 1e-9,
            _ => (rng.next() % 1_000_000) as f32 / 1_000_000.0,
        };
        for strict in [true, false] {
            let holds = |e: f32| if strict { e / n > v } else { e / n >= v };
            let e = least_sum(v, n, strict);
            assert!(holds(e), "{v} {n} {strict}: {e}");
            assert!(e == 0.0 || !holds(e.next_down()), "{v} {n} {strict}: {e} is not the least");
        }
    }
    assert_eq!(least_sum(f32::INFINITY, 4.0, true), f32::INFINITY);
}

#[test]
fn valid_sources_are_the_wholly_copyable_patches() {
    let (w, h, r) = (37, 29, 2);
    let mut rng = Rng::new(9);
    let px: Vec<Px> = (0..w * h)
        .map(|_| match rng.next() % 23 {
            0 => Px::Hole,
            1 => Px::Void,
            2 => Px::Known,
            _ => Px::Source,
        })
        .collect();
    let valid = valid_sources(w, h, r, &px, &photocraft_raster::Interrupt::NONE).unwrap();
    let source: Vec<bool> = px.iter().map(|k| *k == Px::Source).collect();
    let ss = integral(w, h, &source);
    let (ri, full) = (r as i32, ((2 * r + 1) * (2 * r + 1)) as u32);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let inside = x >= ri && y >= ri && x + ri < w as i32 && y + ri < h as i32;
            let want = inside && window_count(&ss, w, h, x - ri, y - ri, x + ri + 1, y + ri + 1) == full;
            assert_eq!(valid[y as usize * w + x as usize], want, "({x}, {y})");
        }
    }
    // Counting per row and block finds the centres in order.
    let sources = Sources::new(&valid, w);
    let list = sources.all(&valid, w);
    assert!(list.len() > 10 && list.len() == sources.len());
    for (k, p) in list.iter().enumerate() {
        assert_eq!(sources.nth(k, &valid, w), *p);
    }
    // And on a grid wider than a block, with empty blocks and rows.
    let (w, h) = (300, 9);
    let wide: Vec<bool> = (0..w * h).map(|i| (i % w) % 7 == 3 && (i % w) / 64 != 2 && i / w != 4).collect();
    let sources = Sources::new(&wide, w);
    let list = sources.all(&wide, w);
    for (k, p) in list.iter().enumerate() {
        assert_eq!(sources.nth(k, &wide, w), *p);
    }
}

#[test]
fn full_patches_measure_as_clipped_ones() {
    let (w, h, ch, r) = (46, 38, 4, 3);
    let mut img = busy(w, h, ch, 4);
    let hole = disc(w, h, 20.0, 18.0, 6.0);
    let px = roles(w, h, &hole);
    let none = photocraft_raster::Interrupt::NONE;
    let hb = hole_bounds(w, h, &px).unwrap();
    let tb = hb.grow(r, w, h);
    let valid = valid_sources(w, h, r, &px, &none).unwrap();
    let picks: Vec<(i32, i32)> = Sources::new(&valid, w).all(&valid, w).iter().step_by(37).copied().collect();
    assert!(picks.len() > 5);
    for adapt in [Adapt::NONE, Adapt { gain: (0.8, 1.25), bias: 0.08 }] {
        for keep in [false, true] {
            let cells = Cells::around(w, r, &px, hb, tb);
            let mut s = Solver {
                w,
                h,
                ch,
                r: r as i32,
                img: &mut img,
                guide: None,
                mimg: None,
                coherence: 0.0,
                stamp: false,
                blend_edge: false,
                reach: 0,
                depth: None,
                uses: None,
                px: &px,
                valid: valid.clone(),
                cells,
                full: Vec::new(),
                tmom: Vec::new(),
                smom: Vec::new(),
                adapt,
                ctl: none,
            };
            s.prepare(keep);
            let nnf = vec![picks[0]; s.cells.len()];
            let mut cost = vec![f32::INFINITY; s.cells.len()];
            s.costs(&nnf, &mut cost).unwrap();
            let mut checked = 0;
            let cells: Vec<(usize, usize, usize)> = s.cells.iter().collect();
            for (k, x, y) in cells {
                if !s.full[k] {
                    continue;
                }
                let t = (x as i32, y as i32);
                for &src in &picks {
                    if adapt.is_none() {
                        let exact = s.dist_clipped(t, src, f32::INFINITY);
                        assert_eq!(s.dist_full(t, src, f32::INFINITY).to_bits(), exact.to_bits());
                        for cutoff in [exact * 0.5, exact, exact * 2.0] {
                            let (a, b) = (s.dist_full(t, src, cutoff), s.dist_clipped(t, src, cutoff));
                            assert_eq!(a < cutoff, b < cutoff);
                        }
                    } else {
                        let (mut m1, mut m2) = ([[0.0f32; 5]; 4], [[0.0f32; 5]; 4]);
                        s.moments_full(k, t, src, &mut m1);
                        assert_eq!(s.moments(t, src, &mut m2), (2 * r + 1) * (2 * r + 1));
                        assert_eq!(m1.map(|c| c.map(f32::to_bits)), m2.map(|c| c.map(f32::to_bits)));
                        assert_eq!(s.dist_adapted_full(k, t, src).to_bits(), s.dist_adapted(t, src).to_bits());
                    }
                    checked += 1;
                }
            }
            assert!(checked > 100, "{checked}");
        }
    }
}

#[test]
fn owned_completion_fills_in_place_or_gives_the_pixels_back() {
    let (w, h, ch) = (64, 52, 3);
    let img = busy(w, h, ch, 6);
    let hole = disc(w, h, 30.0, 26.0, 8.0);
    let px = roles(w, h, &hole);
    let (p, none) = (CompleteParams::default(), photocraft_raster::Interrupt::NONE);
    let borrowed = complete_masked(w, h, ch, &img, &px, &p, &none).unwrap().unwrap();
    let owned = complete_masked_owned(w, h, ch, img.clone(), &px, &p, &none).unwrap().unwrap();
    assert_eq!(bits(&owned), bits(&borrowed));
    // Nothing may be copied: no completion, and the pixels come back as they were.
    let known: Vec<Px> = px.iter().map(|k| if *k == Px::Source { Px::Known } else { *k }).collect();
    assert!(complete_masked(w, h, ch, &img, &known, &p, &none).unwrap().is_none());
    let back = complete_masked_owned(w, h, ch, img.clone(), &known, &p, &none).unwrap().unwrap_err();
    assert_eq!(bits(&back), bits(&img));
    // Cancelled, it stops.
    let yes = || true;
    assert!(complete_masked_owned(w, h, ch, img, &px, &p, &photocraft_raster::Interrupt::cancel_only(&yes)).is_err());
}

#[test]
fn synthesis_fills_hole_with_plausible_values() {
    let (w, h, ch) = (64, 64, 3);
    let clean = texture(w, h);
    let hole = disc(w, h, 32.0, 32.0, 6.0);
    let out = synthesize(w, h, ch, &clean, &hole, 8, 3);
    let err = hole_rmse(&out, &clean, &hole, ch);
    assert!(err < 0.15, "{err}");
    assert!(out.iter().all(|v| v.is_finite()));
}

/// The target cells are the centres within the patch radius of a hole pixel, numbered in
/// raster order, however ragged the hole.
#[test]
fn target_cells_are_the_centres_near_the_hole() {
    let mut rng = Rng::new(5);
    for (w, h, r) in [(41usize, 33usize, 3usize), (17, 60, 1), (64, 9, 2), (5, 5, 3)] {
        let hole: Vec<bool> = (0..w * h).map(|i| (i % w + i / w).is_multiple_of(11) || rng.next().is_multiple_of(17)).collect();
        let px = roles(w, h, &hole);
        let hb = hole_bounds(w, h, &px).unwrap();
        let tb = hb.grow(r, w, h);
        let cells = Cells::around(w, r, &px, hb, tb);
        let mut k = 0;
        for y in 0..h {
            for x in 0..w {
                let near = (y.saturating_sub(r)..(y + r + 1).min(h)).any(|yy| (x.saturating_sub(r)..(x + r + 1).min(w)).any(|xx| hole[yy * w + xx]));
                assert_eq!(cells.index(x, y), near.then_some(k), "({x}, {y}) in {w}×{h}, r {r}");
                k += usize::from(near);
            }
        }
        assert_eq!(cells.len(), k);
        assert!(cells.iter().enumerate().all(|(i, (n, x, y))| i == n && cells.index(x, y) == Some(n)));
        for y in tb.y0..tb.y1 {
            for x in tb.x0..tb.x1 {
                let row: Vec<usize> = cells.iter().filter(|c| c.2 == y).map(|c| c.1).collect();
                assert_eq!(cells.in_row(y, x, true), row.iter().filter(|c| **c < x).count());
                assert_eq!(cells.in_row(y, x, false), row.iter().filter(|c| **c > x).count());
            }
            assert_eq!(cells.before_row(y + 1) - cells.before_row(y), cells.row(y).iter().map(|s| s.x1 - s.x0).sum::<usize>());
        }
    }
}

/// A fill area along all four borders (the selection of a panorama's empty edges) keeps its
/// per-target tables to the ring, not to its bounds (the whole image).
#[test]
fn a_border_ring_has_targets_only_along_the_borders() {
    let (w, h, r, band) = (400, 300, 3, 6);
    let hole: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            x < band || y < band || x >= w - band || y >= h - band
        })
        .collect();
    let px = roles(w, h, &hole);
    let hb = hole_bounds(w, h, &px).unwrap();
    let cells = Cells::around(w, r, &px, hb, hb.grow(r, w, h));
    let ring = w * h - (w - 2 * (band + r)) * (h - 2 * (band + r));
    assert_eq!(cells.len(), ring);
    assert!(cells.len() * 5 < hb.area(), "{} targets of {}", cells.len(), hb.area());
    // And the ring fills from the image (deterministically, whatever the thread count).
    let img = busy(w, h, 3, 8);
    let p = CompleteParams { adapt: Adapt { gain: (0.8, 1.25), bias: 0.08 }, ..Default::default() };
    let a = complete(w, h, 3, &img, &hole, &p).expect("filled");
    let b = complete(w, h, 3, &img, &hole, &p).expect("filled");
    assert_eq!(bits(&a), bits(&b));
    assert!(a.iter().all(|v| v.is_finite()));
}

/// A guide decides the fill's layout: a hole across the line between a bright upper half and a
/// dark lower one, guided by the picture as it was, comes out with the line where it was, and
/// closer to the picture than the unguided fill. The pixels outside the hole stay as they are.
#[test]
fn a_guide_steers_the_fill_towards_it() {
    let (w, h, ch) = (120, 96, 3);
    let base = busy(w, h, ch, 6);
    let truth: Vec<f32> = base.iter().enumerate().map(|(i, v)| if i / ch / w < 40 { (v + 0.4).min(1.0) } else { v * 0.5 }).collect();
    let hole = disc(w, h, 60.0, 40.0, 22.0);
    let px = roles(w, h, &hole);
    let p = CompleteParams::default();
    let none = photocraft_raster::Interrupt::NONE;
    let plain = complete_masked_owned(w, h, ch, truth.clone(), &px, &p, &none).unwrap().unwrap();
    let guided = complete_guided_owned(w, h, ch, truth.clone(), &px, &truth, 1.0, &p, &none).unwrap().unwrap();
    // The layout, not the pixels: the fill stamps pieces of the texture, so compare 7 × 7 means.
    let soft = |v: &[f32]| -> Vec<f32> {
        (0..v.len())
            .map(|i| {
                let (x, y, c) = ((i / ch) % w, i / ch / w, i % ch);
                let (x0, x1, y0, y1) = (x.saturating_sub(3), (x + 4).min(w), y.saturating_sub(3), (y + 4).min(h));
                let n = ((x1 - x0) * (y1 - y0)) as f32;
                (y0..y1).flat_map(|yy| (x0..x1).map(move |xx| (yy * w + xx) * ch + c)).map(|j| v[j]).sum::<f32>() / n
            })
            .collect()
    };
    let st = soft(&truth);
    let (e_plain, e_guided) = (hole_rmse(&soft(&plain), &st, &hole, ch), hole_rmse(&soft(&guided), &st, &hole, ch));
    assert!(e_guided < e_plain, "guided {e_guided} vs unguided {e_plain}");
    // Bright above the line, dark below it, inside the hole too.
    let mean = |y: usize| (40..80).map(|x| guided[(y * w + x) * ch]).sum::<f32>() / 40.0;
    assert!(mean(30) > mean(50) + 0.2, "{} above vs {} below", mean(30), mean(50));
    for (i, hh) in hole.iter().enumerate() {
        if !*hh {
            assert_eq!(bits(&guided[i * ch..(i + 1) * ch]), bits(&truth[i * ch..(i + 1) * ch]));
        }
    }
}

/// A soft guide (a model's fill scaled up) still decides the layout: guided by the picture
/// blurred over 6 pixels and told so, the fill keeps the line between the bright and dark halves
/// and comes out sharp, as textured as the picture rather than as soft as the guide.
#[test]
fn a_soft_guide_gives_the_layout_and_the_picture_the_detail() {
    let (w, h, ch) = (120, 96, 3);
    let base = busy(w, h, ch, 6);
    let truth: Vec<f32> = base.iter().enumerate().map(|(i, v)| if i / ch / w < 40 { (v + 0.4).min(1.0) } else { v * 0.5 }).collect();
    // The guide: 6 × 6 block means, scaled back up.
    let mut soft = truth.clone();
    for by in (0..h).step_by(6) {
        for bx in (0..w).step_by(6) {
            for c in 0..ch {
                let cells: Vec<usize> = (by..(by + 6).min(h)).flat_map(|y| (bx..(bx + 6).min(w)).map(move |x| y * w + x)).collect();
                let mean = cells.iter().map(|i| truth[i * ch + c]).sum::<f32>() / cells.len() as f32;
                for i in cells {
                    soft[i * ch + c] = mean;
                }
            }
        }
    }
    let hole = disc(w, h, 60.0, 40.0, 22.0);
    let px = roles(w, h, &hole);
    let p = CompleteParams::default();
    let none = photocraft_raster::Interrupt::NONE;
    let guided = complete_guided_owned(w, h, ch, truth.clone(), &px, &soft, 6.0, &p, &none).unwrap().unwrap();
    let mean = |y: usize| (40..80).map(|x| guided[(y * w + x) * ch]).sum::<f32>() / 40.0;
    assert!(mean(30) > mean(50) + 0.2, "{} above vs {} below", mean(30), mean(50));
    // Detail: neighbouring pixels in the hole differ about as much as in the picture, far more
    // than in the guide.
    let roughness = |v: &[f32]| {
        let d: Vec<f32> = (0..w * h).filter(|&i| hole[i] && i % w + 1 < w && hole[i + 1]).map(|i| (v[i * ch] - v[(i + 1) * ch]).abs()).collect();
        d.iter().sum::<f32>() / d.len() as f32
    };
    let (r_fill, r_truth, r_guide) = (roughness(&guided), roughness(&truth), roughness(&soft));
    assert!(r_fill > 0.5 * r_truth && r_fill > 2.0 * r_guide, "fill {r_fill}, picture {r_truth}, guide {r_guide}");
}

/// A model's fill lighter and flatter than the picture around it takes the picture's tones; the
/// picture itself is left alone.
#[test]
fn a_guide_lighter_than_the_picture_takes_its_tones() {
    let (w, h, ch) = (96usize, 64usize, 3usize);
    let hole: Vec<bool> = (0..w * h).map(|i| (32..64).contains(&(i % w))).collect();
    // Rows alternate dark and light around the hole (mean 0.3); the guide in it is light and flat.
    let mut g: Vec<f32> = (0..w * h)
        .flat_map(|i| {
            let odd = ((i / w) % 2) as f32;
            let v = if hole[i] { 0.6 + 0.02 * odd } else { 0.2 + 0.2 * odd };
            [v, v, v]
        })
        .collect();
    let before = g.clone();
    match_guide_tones(w, h, ch, 3, &mut g, &hole, 16);
    let hole_mean = (0..w * h).filter(|&i| hole[i]).map(|i| g[i * ch]).sum::<f32>() / (0..w * h).filter(|&i| hole[i]).count() as f32;
    assert!((hole_mean - 0.3).abs() < 0.03, "the hole's mean {hole_mean}");
    assert!(g.iter().zip(&before).enumerate().all(|(k, (a, b))| hole[k / ch] || a == b), "the picture is untouched");
}

/// Nearby: every match a search step gives is within reach of its target, also from a start
/// out of reach (a random one, a coarser level's) and with the whole picture searched.
#[test]
fn a_nearby_fill_takes_matches_only_within_reach() {
    let (w, h, ch, r) = (90, 60, 3, 3);
    let mut img = busy(w, h, ch, 4);
    let hole = disc(w, h, 45.0, 30.0, 8.0);
    let px: Vec<Px> = hole.iter().map(|hh| if *hh { Px::Hole } else { Px::Source }).collect();
    let none = photocraft_raster::Interrupt::NONE;
    let hb = hole_bounds(w, h, &px).unwrap();
    let valid = valid_sources(w, h, r, &px, &none).unwrap();
    let far = Sources::new(&valid, w).all(&valid, w).into_iter().max_by_key(|p| (p.0 - 45).abs() + (p.1 - 30).abs()).unwrap();
    let reach = 12;
    let mut s = Solver {
        w,
        h,
        ch,
        r: r as i32,
        img: &mut img,
        guide: None,
        mimg: None,
        coherence: 0.0,
        stamp: false,
        blend_edge: false,
        reach,
        depth: Some(solver::Depth::new(&valid, w, h, reach as usize, reach, 0.0, &[])),
        uses: None,
        px: &px,
        valid: valid.clone(),
        cells: Cells::around(w, r, &px, hb, hb.grow(r, w, h)),
        full: Vec::new(),
        tmom: Vec::new(),
        smom: Vec::new(),
        adapt: Adapt::NONE,
        ctl: none,
    };
    s.prepare(false);
    assert!(!s.allowed((45, 30), far) && s.allowed((45, 30), (45 + reach, 30 - reach)));
    let mut rng = Rng::new(9);
    let cells: Vec<(usize, usize, usize)> = s.cells.iter().collect();
    for (k, x, y) in cells {
        let t = (x as i32, y as i32);
        let (m, c) = s.pm_step(t, k, (far, 0.0), Some(far), Some(far), 1, &mut rng, w as i32);
        assert!(s.allowed(t, m), "{t:?} took {m:?}");
        assert!(c.is_finite());
    }
}

/// Same depth: a target copies from about its own rows, more the farther it is from the
/// horizon; one with no source within reach in its rows, as far again as the nearest rows that
/// have one, twice over; and each block has a source to fall back on.
#[test]
fn a_same_depth_fill_copies_from_about_its_own_rows() {
    let (w, h, reach) = (64, 64, 8);
    // Sources only in the top 16 rows.
    let valid: Vec<bool> = (0..w * h).map(|i| i / w < 16).collect();
    let d = solver::Depth::new(&valid, w, h, reach, 2, 0.0, &[]);
    assert_eq!(d.rows((30, 8)), 2, "a row with sources");
    // Row 40 is in block row 5 (of 8 rows each); the nearest with sources is block row 1.
    assert_eq!(d.rows((30, 40)), 2 + 2 * 32);
    // From a horizon at row 40: half the distance to it.
    let d = solver::Depth::new(&valid, w, h, reach, 2, 0.5, &[40.0; 64]);
    assert_eq!((d.rows((30, 10)), d.rows((30, 40))), (15, 2 + 2 * 32));
    // Sources across the picture's top, and down its left 40 columns: block column 7 (columns
    // 56..64) has none within 8 columns of all its pixels but in the top block row, 5 block rows
    // up; block column 5 (40..48) has column 39.
    let valid: Vec<bool> = (0..w * h).map(|i| i % w < 40 || i / w < 8).collect();
    let d = solver::Depth::new(&valid, w, h, reach, 2, 0.0, &[]);
    assert_eq!(d.rows((44, 40)), 2, "valid columns within reach");
    assert_eq!(d.rows((60, 40)), 2 + 2 * 40);
    for t in [(44, 40), (60, 40), (63, 63)] {
        let s = d.fallback(t).unwrap();
        assert!(valid[s.1 as usize * w + s.0 as usize] && (s.0 - t.0).abs() <= reach as i32 && (s.1 - t.1).abs() <= d.rows(t) + 8, "{t:?}: {s:?}");
    }
}

/// Along a tilted horizon, the same depth is the same distance from it, not the same row: a
/// source 10 columns on, where the horizon is 5 rows lower, matches a target 5 rows higher.
#[test]
fn the_same_depth_follows_a_tilted_horizon() {
    let (w, h, ch, r) = (80, 60, 3, 3);
    let mut img = busy(w, h, ch, 2);
    let px: Vec<Px> = (0..w * h).map(|i| if i % w >= 64 { Px::Hole } else { Px::Source }).collect();
    let none = photocraft_raster::Interrupt::NONE;
    let hb = hole_bounds(w, h, &px).unwrap();
    let valid = valid_sources(w, h, r, &px, &none).unwrap();
    // The horizon sinks half a row per column.
    let horizon: Vec<f32> = (0..w).map(|x| 20.0 + x as f32 / 2.0).collect();
    let mut s = Solver {
        w,
        h,
        ch,
        r: r as i32,
        img: &mut img,
        guide: None,
        mimg: None,
        coherence: 0.0,
        stamp: false,
        blend_edge: false,
        reach: 40,
        depth: Some(solver::Depth::new(&valid, w, h, 40, 1, 0.0, &horizon)),
        uses: None,
        px: &px,
        valid: valid.clone(),
        cells: Cells::around(w, r, &px, hb, hb.grow(r, w, h)),
        full: Vec::new(),
        tmom: Vec::new(),
        smom: Vec::new(),
        adapt: Adapt::NONE,
        ctl: none,
    };
    s.prepare(false);
    assert!(s.allowed((70, 30), (60, 25)), "10 columns back, the horizon is 5 rows higher");
    assert!(!s.allowed((70, 30), (60, 30)), "the same row is 5 rows deeper there");
}

/// The horizon: the row that best splits the picture into sky (bluer than red) above and
/// ground below, per column along a tilted one; none without a clear split, or without colour.
#[test]
fn the_horizon_is_where_the_sky_meets_the_ground() {
    let (w, h) = (40, 30);
    let px = vec![Px::Source; w * h];
    let land: Vec<f32> = (0..w * h).flat_map(|i| if i / w < 12 { [0.2, 0.3, 0.6] } else { [0.4, 0.3, 0.2] }).collect();
    assert_eq!(solver::horizon(&land, &px, w, h, 3), Some(12.0));
    // A hole over part of it changes nothing.
    let holed: Vec<Px> = (0..w * h).map(|i| if i % w < 10 { Px::Hole } else { Px::Source }).collect();
    assert_eq!(solver::horizon(&land, &holed, w, h, 3), Some(12.0));
    let stripes: Vec<f32> = (0..w * h).flat_map(|i| if i % 2 == 0 { [0.2, 0.3, 0.6] } else { [0.4, 0.3, 0.2] }).collect();
    assert_eq!(solver::horizon(&stripes, &px, w, h, 3), None, "no clear split");
    assert_eq!(solver::horizon(&vec![0.5; w * h], &px, w, h, 1), None, "no colour");
    // Tilted: from row 40 on the left to row 60 on the right, 320 columns across; the hole over
    // the right 60 columns takes the row of the last strip with pixels.
    let (w, h) = (320, 100);
    let at = |x: usize| 40.0 + 20.0 * x as f32 / w as f32;
    let tilted: Vec<f32> = (0..w * h).flat_map(|i| if ((i / w) as f32) < at(i % w) { [0.2, 0.3, 0.6] } else { [0.4, 0.3, 0.2] }).collect();
    let holed: Vec<Px> = (0..w * h).map(|i| if i % w >= 260 { Px::Hole } else { Px::Source }).collect();
    let line = solver::horizon_line(&tilted, &holed, w, h, 3).unwrap();
    assert!((line[100] - at(100)).abs() <= 1.5, "{} at column 100, not {}", line[100], at(100));
    assert!((line[310] - line[250]).abs() <= 1.5 && line[310] > 52.0, "held on past the last: {} / {}", line[310], line[250]);
}

/// A dark ridge of the sky's blue-grey, under bright haze: the colour split alone puts the
/// horizon at the ridge's foot (below it, the ground's brown); the line snaps to the ridge's top,
/// where the brightness drops, as a person would draw it.
#[test]
fn the_horizon_is_the_ridges_top_when_the_ridge_is_not_brown() {
    let (w, h) = (480, 200);
    let px = vec![Px::Source; w * h];
    let img: Vec<f32> = (0..w * h)
        .flat_map(|i| match i / w {
            0..=99 => [0.45, 0.46, 0.50],
            100..=105 => [0.12, 0.12, 0.14],
            _ => [0.20, 0.19, 0.18],
        })
        .collect();
    let line = solver::horizon_line(&img, &px, w, h, 3).unwrap();
    assert!(line.iter().all(|r| (r - 100.0).abs() <= 3.0), "{} / {}", line[0], line[300]);
}

/// Seamless levels: a fill lighter than the picture beside it is brought down to it, all the
/// way across a strip along the picture's edge; a step too large to be a level (a cloud by sky)
/// is left as it is.
#[test]
fn a_fill_takes_the_levels_of_the_picture_beside_it() {
    let (w, h, ch) = (160, 64, 3);
    let hole: Vec<bool> = (0..w * h).map(|i| i % w >= 96).collect();
    let mut img: Vec<f32> = (0..w * h).flat_map(|i| if i % w >= 96 { [0.43; 3] } else { [0.4; 3] }).collect();
    seamless_levels(&mut img, &hole, w, h, ch, 3);
    for x in [100, 150] {
        let v = img[(32 * w + x) * ch];
        assert!((v - 0.4).abs() < 0.004, "column {x}: {v}");
    }
    assert_eq!(img[(32 * w + 20) * ch], 0.4, "the picture is kept");
    let mut big: Vec<f32> = (0..w * h).flat_map(|i| if i % w >= 96 { [0.7; 3] } else { [0.4; 3] }).collect();
    seamless_levels(&mut big, &hole, w, h, ch, 3);
    assert_eq!(big[(32 * w + 120) * ch], 0.7);
}

/// A mask grown by r: each set pixel spreads r px across and down (a square), no farther.
#[test]
fn a_mask_grows_by_its_radius_each_way() {
    let (w, h) = (30, 20);
    let mut m = vec![false; w * h];
    m[10 * w + 15] = true;
    let g = dilate_mask(&m, w, h, 3);
    for y in 0..h {
        for x in 0..w {
            let inside = (12..=18).contains(&x) && (7..=13).contains(&y);
            assert_eq!(g[y * w + x], inside, "({x}, {y})");
        }
    }
    assert_eq!(dilate_mask(&m, w, h, 0), m);
}

/// A transparent edge's fringe (alpha below 1) and the pixels within `fringe` of it are not
/// sources; the opaque picture past them is.
#[test]
fn the_fringe_of_a_transparent_edge_is_not_copied() {
    let (w, h, ch) = (20, 6, 4);
    // Transparent left of x 5, half at x 5, opaque from x 6.
    let img: Vec<f32> = (0..w * h)
        .flat_map(|i| {
            let a = match i % w {
                0..5 => 0.0,
                5 => 0.5,
                _ => 1.0,
            };
            [0.3, 0.6, 0.9, a]
        })
        .collect();
    let mut source: Vec<bool> = (0..w * h).map(|i| i % w >= 5).collect();
    drop_fringe(&mut source, &img, w, h, ch, 2);
    for x in 0..w {
        assert_eq!(source[2 * w + x], x >= 8, "x {x}");
    }
    // With nothing left it keeps the sources it had.
    let mut thin: Vec<bool> = (0..w * h).map(|i| i % w == 6).collect();
    drop_fringe(&mut thin, &img, w, h, ch, 2);
    assert!(thin[2 * w + 6]);
}

/// The fallback source of a block with no source in its rows lies inside the sources by more
/// than a block's shift, not on their edge.
#[test]
fn a_fallback_source_lies_inside_the_picture() {
    let (w, h) = (64, 100);
    // Sources below row 50 only: block row 6 (rows 48..56) has sources from row 50, and one
    // inside them by more than 4 + 1 rows at row 55.
    let valid: Vec<bool> = (0..w * h).map(|i| i / w >= 50).collect();
    let d = Depth::new(&valid, w, h, 64, 2, 0.0, &[]);
    let s = d.fallback((20, 4)).expect("a fallback");
    assert_eq!(s.1, 55, "{s:?}");
    assert_eq!(d.rows((20, 4)), 2 + 2 * 48, "the rows measured to the sources' edge");
    assert!(valid[s.1 as usize * w + s.0 as usize]);
}

/// The sky rule: above the horizon, where the hole's edge is mostly cloud a target may copy
/// cloud; where it is sky, only sky away from any cloud. Below the horizon, anything.
#[test]
fn the_sky_goes_on_with_clouds_only_where_the_edge_is_cloud() {
    let (w, h, ch) = (64, 64, 3);
    // Sky above row 40, ground below; a grey cloud over columns 0..20, rows 6..30.
    let img: Vec<f32> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            if y >= 40 {
                [0.35, 0.3, 0.25]
            } else if x < 20 && y < 30 {
                [0.45, 0.46, 0.48]
            } else {
                [0.2, 0.25, 0.35]
            }
        })
        .collect();
    // The hole: a strip along the top.
    let px: Vec<Px> = (0..w * h).map(|i| if i / w < 6 { Px::Hole } else { Px::Source }).collect();
    let k = solver::SkyRule::new(&img, &px, w, h, ch, |_| 40, 1, 8).expect("a sky rule");
    let (cloud, sky) = ((10, 15), (50, 20));
    assert!(k.allows((4, 2), cloud, w), "over the cut cloud, cloud goes on");
    assert!(!k.allows((50, 2), cloud, w), "over sky, no new cloud");
    assert!(k.allows((50, 2), sky, w));
    // No split without clouds.
    let clear: Vec<f32> = (0..w * h).flat_map(|i| if i / w >= 40 { [0.35, 0.3, 0.25] } else { [0.2, 0.25, 0.35] }).collect();
    assert!(solver::SkyRule::new(&clear, &px, w, h, ch, |_| 40, 1, 8).is_none());
}

/// Uses: a block copied once (a stretch copied whole) costs nothing more; each further copy of
/// it costs the weight times the patch distance more.
#[test]
fn a_part_copied_more_than_once_costs_more() {
    let mut u = solver::Uses::new(20, 20, 2, 0.5);
    // Block (0, 0) copied whole once: 4 targets, one per pixel.
    let mut nnf = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
    u.count(&nnf);
    assert_eq!(u.factor((0, 0)), 1.0);
    // ... and three more times over: 16 targets in all.
    nnf.extend(std::iter::repeat_n([(0, 0), (1, 0), (0, 1), (1, 1)], 3).flatten());
    u.count(&nnf);
    assert_eq!(u.factor((1, 1)), 1.0 + 0.5 * 3.0);
    assert_eq!(u.factor((5, 5)), 1.0, "a block nobody copies");
    assert_eq!(u.factor((-3, 2)), 1.0, "off the picture");
}

/// Same depth, in a search: every match a step gives is within the rows its target may copy
/// from, also from a start out of them.
#[test]
fn a_same_depth_fill_takes_matches_only_from_its_rows() {
    let (w, h, ch, r) = (120, 80, 3, 3);
    let mut img = busy(w, h, ch, 5);
    // A strip down the right edge, as along a panorama's border.
    let px: Vec<Px> = (0..w * h).map(|i| if i % w >= 96 && i / w >= 10 { Px::Hole } else { Px::Source }).collect();
    let none = photocraft_raster::Interrupt::NONE;
    let hb = hole_bounds(w, h, &px).unwrap();
    let valid = valid_sources(w, h, r, &px, &none).unwrap();
    let reach = 40;
    let depth = Some(solver::Depth::new(&valid, w, h, reach, 8, 0.0, &[]));
    let mut s = Solver {
        w,
        h,
        ch,
        r: r as i32,
        img: &mut img,
        guide: None,
        mimg: None,
        coherence: 0.0,
        stamp: false,
        blend_edge: false,
        reach: reach as i32,
        depth,
        uses: None,
        px: &px,
        valid: valid.clone(),
        cells: Cells::around(w, r, &px, hb, hb.grow(r, w, h)),
        full: Vec::new(),
        tmom: Vec::new(),
        smom: Vec::new(),
        adapt: Adapt::NONE,
        ctl: none,
    };
    s.prepare(false);
    assert!(s.allowed((100, 40), (70, 46)) && !s.allowed((100, 40), (70, 60)), "8 rows up or down beside the strip");
    let far = (3, 3);
    let mut rng = Rng::new(4);
    let cells: Vec<(usize, usize, usize)> = s.cells.iter().collect();
    for (k, x, y) in cells {
        let t = (x as i32, y as i32);
        let (m, _) = s.pm_step(t, k, (far, 0.0), Some(far), Some(far), 1, &mut rng, w as i32);
        assert!(s.allowed(t, m), "{t:?} took {m:?}");
    }
}

/// Layout only: the guide decides where the bright and dark parts go, and the fill is Classic's
/// otherwise: no guide colour is matched against, so the pieces are the picture's.
#[test]
fn a_layout_only_guide_lays_out_the_hole_and_classic_fills_it() {
    let (w, h, ch) = (120, 96, 3);
    let base = busy(w, h, ch, 6);
    let truth: Vec<f32> = base.iter().enumerate().map(|(i, v)| if i / ch / w < 40 { (v + 0.4).min(1.0) } else { v * 0.5 }).collect();
    let hole = disc(w, h, 60.0, 40.0, 22.0);
    let px = roles(w, h, &hole);
    let none = photocraft_raster::Interrupt::NONE;
    // A guide that is the truth's layout in flat tones: bright above the line, dark below.
    let flat: Vec<f32> = (0..w * h * ch).map(|i| if i / ch / w < 40 { 0.8 } else { 0.2 }).collect();
    let p = CompleteParams { layout_only: true, nearby: true, ..Default::default() };
    let out = complete_guided_owned(w, h, ch, truth.clone(), &px, &flat, 1.0, &p, &none).unwrap().unwrap();
    let mean = |y: usize| (45..75).map(|x| out[(y * w + x) * ch]).sum::<f32>() / 30.0;
    assert!(mean(30) > mean(50) + 0.2, "{} above vs {} below", mean(30), mean(50));
    // Textured like the picture, not flat like the guide.
    let spread = |y: usize| {
        let m = mean(y);
        ((45..75).map(|x| (out[(y * w + x) * ch] - m).powi(2)).sum::<f32>() / 30.0).sqrt()
    };
    assert!(spread(30) > 0.05 && spread(50) > 0.02, "spread {} / {}", spread(30), spread(50));
}

/// A nearby fill stamps: away from the picture, each pixel is one of the picture's own (a sharp
/// copy, not a blend of near misses); beside it, a blend.
#[test]
fn a_nearby_fill_copies_pixels_whole_away_from_the_picture() {
    let (w, h, ch) = (96, 64, 3);
    let img = busy(w, h, ch, 9);
    let hole = disc(w, h, 48.0, 32.0, 14.0);
    let px = roles(w, h, &hole);
    let none = photocraft_raster::Interrupt::NONE;
    let p = CompleteParams { nearby: true, ..Default::default() };
    let out = complete_masked_owned(w, h, ch, img.clone(), &px, &p, &none).unwrap().unwrap();
    let picture: std::collections::HashSet<[u32; 3]> =
        (0..w * h).filter(|i| !hole[*i]).map(|i| [img[i * ch].to_bits(), img[i * ch + 1].to_bits(), img[i * ch + 2].to_bits()]).collect();
    let deep = disc(w, h, 48.0, 32.0, 8.0);
    let whole =
        (0..w * h).filter(|i| deep[*i]).filter(|i| picture.contains(&[out[i * ch].to_bits(), out[i * ch + 1].to_bits(), out[i * ch + 2].to_bits()])).count();
    let n = deep.iter().filter(|d| **d).count();
    assert_eq!(whole, n, "{whole} of {n} pixels deep in the hole are copies");
}
