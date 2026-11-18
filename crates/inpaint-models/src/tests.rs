use super::*;

/// The tiles' middles cover every pixel of a side exactly once, each inside its own tile,
/// whatever the side's length.
#[test]
fn tile_middles_cover_a_side_once() {
    for n in [1, 300, 512, 513, 700, 767, 768, 1024, 1873, 2048, 6243] {
        let mut hits = vec![0u8; n];
        for (s, lo, hi) in spans(n) {
            assert!(s + TILE <= n.max(TILE), "{n}: tile at {s} runs past the end");
            assert!(s <= lo && lo < hi && hi <= s + TILE, "{n}: middle {lo}..{hi} of the tile at {s}");
            hits[lo..hi].iter_mut().for_each(|v| *v += 1);
        }
        assert!(hits.iter().all(|v| *v == 1), "{n}: {:?}", hits.iter().position(|v| *v != 1));
    }
}

/// Every hole pixel is filled once, by the tile whose middle holds it, and nothing else changes;
/// only tiles with some of the hole run.
#[test]
fn tiles_fill_the_whole_hole_and_nothing_else() {
    let (w, h) = (1300, 1300);
    let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
    // A border ring and a blob, like a panorama's empty edges and one object.
    let hole: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            x < 30 || y < 20 || x >= w - 40 || y >= h - 25 || ((x as i32 - 600).pow(2) + (y as i32 - 350).pow(2)) < 900
        })
        .collect();
    let pic = Picture { w, h, rgb: &rgb, hole: &hole };
    let ran = std::cell::Cell::new(0);
    let out = tiles(&pic, &Interrupt::NONE, (0.0, 1.0), |trgb, thole| {
        ran.set(ran.get() + 1);
        Ok(trgb.chunks(3).zip(thole).flat_map(|(p, h)| if *h { [7, 8, 9] } else { [p[0], p[1], p[2]] }).collect())
    })
    .unwrap();
    for i in 0..w * h {
        let want: [u8; 3] = if hole[i] { [7, 8, 9] } else { [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]] };
        assert_eq!(out[i * 3..i * 3 + 3], want, "pixel {i}");
    }
    assert_eq!(ran.get(), tiles_to_fill(&pic).len());
    assert!(ran.get() < spans(w).len() * spans(h).len(), "tiles without any of the hole ran");
}

/// A tile run once is kept, per model: the same tile again (a fill after a preview that stopped
/// part-way) does not run the model.
#[test]
fn a_tile_run_once_is_kept() {
    let rgb: Vec<u8> = (0..TILE * TILE * 3).map(|i| (i % 241) as u8).collect();
    let hole = vec![true; TILE * TILE];
    let calls = std::cell::Cell::new(0);
    let first = run_tile(Model::Lama, &rgb, &hole, || {
        calls.set(calls.get() + 1);
        Ok(vec![9u8; 3])
    })
    .unwrap();
    let again = run_tile(Model::Lama, &rgb, &hole, || {
        calls.set(calls.get() + 1);
        Ok(vec![1u8; 3])
    })
    .unwrap();
    assert_eq!((first, again, calls.get()), (vec![9u8; 3], vec![9u8; 3], 1));
    assert_eq!(run_tile(Model::Migan, &rgb, &hole, || Ok(vec![2u8; 3])).unwrap(), vec![2u8; 3], "another model runs");
}

/// A cancelled fill stops before its next tile.
#[test]
fn a_cancelled_fill_stops() {
    let (w, h) = (600, 600);
    let rgb = vec![0u8; w * h * 3];
    let hole = vec![true; w * h];
    let cancel = || true;
    let ignore = |_: f32| {};
    let ctl = Interrupt::new(&cancel, &ignore);
    let r = tiles(&Picture { w, h, rgb: &rgb, hole: &hole }, &ctl, (0.0, 1.0), |t, _| Ok(t.to_vec()));
    assert_eq!(r, Err(Error::Cancelled));
}

/// Hole pixels take their nearest known colour, so nothing black is left to blend in.
#[test]
fn hole_pixels_take_the_nearest_known_colour() {
    let n = TILE * TILE;
    let mut rgb = vec![0u8; n * 3];
    let mut hole = vec![true; n];
    // Known: the left column red, the right column blue.
    for y in 0..TILE {
        for (x, c) in [(0, [200, 0, 0]), (TILE - 1, [0, 0, 200])] {
            let i = y * TILE + x;
            hole[i] = false;
            rgb[i * 3..i * 3 + 3].copy_from_slice(&c);
        }
    }
    let out = nearest_known(&rgb, &hole);
    let at = |x: usize, y: usize| &out[(y * TILE + x) * 3..(y * TILE + x) * 3 + 3];
    assert_eq!(at(10, 100), [200, 0, 0]);
    assert_eq!(at(TILE - 10, 300), [0, 0, 200]);
    assert!(out.chunks(3).all(|p| p != [0, 0, 0]));
}

/// The mix keeps the textured fill beside textured ground and the smooth fill beside smooth sky.
#[test]
fn the_mix_follows_the_surroundings() {
    let (w, h) = (256, 128);
    // Top half smooth sky, bottom half noisy ground; the hole is a band down the middle.
    let rgb: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let v = if y < h / 2 { 120 } else { ((x * 37 + y * 91) % 97) as u8 + 60 };
            [v, v, v]
        })
        .collect();
    let hole: Vec<bool> = (0..w * h).map(|i| (100..156).contains(&(i % w))).collect();
    let smooth: Vec<u8> = (0..w * h).flat_map(|i| if hole[i] { [120, 120, 120] } else { [rgb[i * 3]; 3] }).collect();
    let rough: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let v = if hole[i] { ((i * 53) % 97) as u8 + 60 } else { rgb[i * 3] };
            [v; 3]
        })
        .collect();
    let pic = Picture { w, h, rgb: &rgb, hole: &hole };
    let out = mix(&pic, &smooth, &rough);
    let at = |x: usize, y: usize| out[(y * w + x) * 3];
    // Deep in the sky the smooth fill, deep in the ground the rough one.
    assert_eq!(at(128, 10), 120);
    assert_eq!(at(128, 120), rough[(120 * w + 128) * 3]);
    // Outside the hole nothing changes.
    assert_eq!(at(10, 120), rgb[(120 * w + 10) * 3]);
}

#[test]
fn methods_parse() {
    for m in [Model::Lama, Model::Migan, Model::Mix] {
        assert_eq!(Model::parse(m.id()), Some(m));
    }
    assert_eq!(Model::parse("classic"), None);
}
