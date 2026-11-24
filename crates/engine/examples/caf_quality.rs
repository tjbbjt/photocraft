//! Quality of Content-Aware Fill on a folder of photos, through the same plan and fill as the
//! Fill dialog's Content-Aware (Shift+F5).
//!
//! ```text
//! cargo run --release -p photocraft-engine --example caf_quality -- <photos dir> <out dir> [methods]
//! ```
//!
//! `methods`: comma-separated, from `classic`, `classicDepth`, `lama`, `migan`, `mix` (default
//! `classic,classicDepth,lama,mix`); the model methods stamp at the same depth, as the dialog
//! starts. A model whose weights aren't there is reported as `n/a`.
//!
//! Each photo (PNG, JPEG or TIFF) is shrunk to at most [`LONG_SIDE`] px and gets holes where it
//! has pixels, placed the same way in every photo (nothing is chosen by looking at it, except the
//! hole on the horizon, which needs one to be found): discs in three rows and two columns, one on
//! the horizon, a large one across it, and wedges in the top corners like a panorama's missing
//! edges. A hole that would cover a transparent pixel is skipped. Measured in the hole against the
//! photo:
//!
//! * `ssim`: structural similarity of the luminance (1 is identical; a plausible fill need not
//!   match, so read it as a trend across many photos);
//! * `texture`: mean gradient of the fill over the photo's (1 is ideal; below 1 it is smoother);
//! * `seam`: the same on the band along the hole's edge (above 1 the edge shows).
//!
//! A photo with transparent areas also has those filled as they are (`real`), with no score: the
//! sheet shows them. Written to the out dir: `scores.csv` (one row per photo, hole and method), a
//! summary on stdout (means per method, and per hole and method), and one sheet per hole,
//! `<photo>-<hole>.png`: photo | hole | each method.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image, SampleType};
use photocraft_doc::{AlphaChannel, ColorMode, PixelFormat, Surface};
use photocraft_engine::Session;
use photocraft_engine::edit_menu_cmds::{caf_fill, content_aware_plan, model_guide};
use photocraft_geom::Rect;
use photocraft_raster::Interrupt;
use serde_json::{Value, json};

/// The longest side a photo is shrunk to: big enough for real texture, small enough that a run
/// over 40 photos takes minutes.
const LONG_SIDE: usize = 2048;

/// A hole, in fractions of the photo: a disc (centre, radius in fractions of the shorter side),
/// or a wedge in a top corner (its width and height along the edges).
#[derive(Clone, Copy)]
enum Shape {
    Disc { c: (f64, f64), r: f64 },
    Corner { right: bool, w: f64, h: f64 },
}

/// A photo, as straight (not premultiplied) RGBA in 0..1, quantised to 8 bits as the document holds it.
struct Photo {
    name: String,
    w: usize,
    h: usize,
    rgba: Vec<f32>,
}

fn load(path: &Path) -> Result<Photo, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let img = photocraft_codecs::decode(&bytes).map_err(|e| format!("{e:?}"))?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let ch = img.layout().channels();
    let px = img.to_normalized();
    let rgba: Vec<f32> = px
        .chunks_exact(ch)
        .flat_map(|p| match ch {
            1 => [p[0], p[0], p[0], 1.0],
            2 => [p[0], p[0], p[0], p[1]],
            3 => [p[0], p[1], p[2], 1.0],
            _ => [p[0], p[1], p[2], p[3]],
        })
        .collect();
    let (w, h, rgba) = shrink(w, h, &rgba);
    let mut rgba: Vec<f32> = rgba.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() / 255.0).collect();
    clear_black_border(w, h, &mut rgba);
    let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(Photo { name, w, h, rgba })
}

/// Makes black reaching the photo's border transparent: a panorama exported without
/// transparency has black where its edges are missing, which would otherwise count as picture
/// (holes cut there would score a fill against black, and fills could copy it).
fn clear_black_border(w: usize, h: usize, rgba: &mut [f32]) {
    let black = |rgba: &[f32], i: usize| rgba[i * 4..i * 4 + 3].iter().all(|v| *v <= 3.0 / 255.0);
    let mut seen = vec![false; w * h];
    let mut stack: Vec<usize> = (0..w).flat_map(|x| [x, (h - 1) * w + x]).chain((0..h).flat_map(|y| [y * w, y * w + w - 1])).collect();
    while let Some(i) = stack.pop() {
        if seen[i] || !black(rgba, i) {
            continue;
        }
        seen[i] = true;
        rgba[i * 4 + 3] = 0.0;
        let (x, y) = (i % w, i / w);
        if x > 0 {
            stack.push(i - 1);
        }
        if x + 1 < w {
            stack.push(i + 1);
        }
        if y > 0 {
            stack.push(i - w);
        }
        if y + 1 < h {
            stack.push(i + w);
        }
    }
}

/// `rgba` shrunk by box averaging so its long side is at most [`LONG_SIDE`]; colours weighted by
/// alpha, so a transparent edge doesn't darken the pixels next to it.
fn shrink(w: usize, h: usize, rgba: &[f32]) -> (usize, usize, Vec<f32>) {
    let f = (w.max(h) as f64 / LONG_SIDE as f64).max(1.0);
    if f <= 1.0 {
        return (w, h, rgba.to_vec());
    }
    let (nw, nh) = (((w as f64 / f).round() as usize).max(1), ((h as f64 / f).round() as usize).max(1));
    let mut out = vec![0.0f32; nw * nh * 4];
    for y in 0..nh {
        let (y0, y1) = (y * h / nh, ((y + 1) * h / nh).max(y * h / nh + 1));
        for x in 0..nw {
            let (x0, x1) = (x * w / nw, ((x + 1) * w / nw).max(x * w / nw + 1));
            let (mut c, mut a, mut n) = ([0.0f64; 3], 0.0f64, 0.0f64);
            for sy in y0..y1.min(h) {
                for sx in x0..x1.min(w) {
                    let p = &rgba[(sy * w + sx) * 4..][..4];
                    let pa = f64::from(p[3]);
                    for (c, v) in c.iter_mut().zip(p) {
                        *c += f64::from(*v) * pa;
                    }
                    a += pa;
                    n += 1.0;
                }
            }
            let o = &mut out[(y * nw + x) * 4..][..4];
            for (o, c) in o.iter_mut().zip(c) {
                *o = if a > 0.0 { (c / a) as f32 } else { 0.0 };
            }
            o[3] = if n > 0.0 { (a / n) as f32 } else { 0.0 };
        }
    }
    (nw, nh, out)
}

fn mask(shape: Shape, w: usize, h: usize) -> Vec<bool> {
    let (wf, hf) = (w as f64, h as f64);
    let short = wf.min(hf);
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
            match shape {
                Shape::Disc { c, r } => (x - c.0 * wf).hypot(y - c.1 * hf) < r * short,
                Shape::Corner { right, w: cw, h: ch } => {
                    let dx = if right { wf - x } else { x };
                    dx / (cw * wf) + y / (ch * hf) < 1.0
                }
            }
        })
        .collect()
}

/// The holes of a photo: the same places in every photo, plus two on its horizon if one is found.
fn holes(p: &Photo) -> Vec<(&'static str, Shape)> {
    use Shape::{Corner, Disc};
    let mut v = vec![
        ("top-l", Disc { c: (0.3, 0.2), r: 0.06 }),
        ("top-r", Disc { c: (0.7, 0.2), r: 0.06 }),
        ("mid-l", Disc { c: (0.3, 0.5), r: 0.06 }),
        ("mid-r", Disc { c: (0.7, 0.5), r: 0.06 }),
        ("low-l", Disc { c: (0.3, 0.8), r: 0.06 }),
        ("low-r", Disc { c: (0.7, 0.8), r: 0.06 }),
        ("corner-l", Corner { right: false, w: 0.3, h: 0.2 }),
        ("corner-r", Corner { right: true, w: 0.3, h: 0.2 }),
    ];
    let rgb: Vec<f32> = p.rgba.as_chunks::<4>().0.iter().flat_map(|q| [q[0], q[1], q[2]]).collect();
    let none = vec![false; p.w * p.h];
    if let Some(hz) = photocraft_algo::inpaint::find_horizon(&rgb, &none, p.w, p.h, 3, 3, None) {
        let y = f64::from(hz[p.w / 2]) / p.h as f64;
        v.push(("horizon", Disc { c: (0.5, y), r: 0.06 }));
        v.push(("big", Disc { c: (0.5, y), r: 0.15 }));
    }
    v
}

/// A session with `p` as its one layer, and `hole` loaded as its selection (through an alpha
/// channel, as Select › Load Selection does).
fn session(p: &Photo, hole: &[bool]) -> Result<Session, String> {
    let mut s = Session::new();
    let err = |e: photocraft_engine::EngineError| e.to_string();
    s.execute("file.new", json!({"width": p.w, "height": p.h, "depth": 8, "background": "transparent"})).map_err(err)?;
    let area = Rect::new(0, 0, p.w as i32, p.h as i32);
    s.edit("photo", |doc, active| {
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let fmt = surf.format();
        let data: Vec<f32> = p.rgba.as_chunks::<4>().0.iter().flat_map(|q| photocraft_raster::from_rgba(&fmt, [q[0], q[1], q[2], q[3]])).collect();
        surf.write_region(area, &data);
        let mut plane = Surface::new(PixelFormat::new(ColorMode::Grayscale, doc.depth, false));
        let vals: Vec<f32> = hole.iter().map(|&m| if m { 1.0 } else { 0.0 }).collect();
        plane.write_region(area, &vals);
        doc.channels.push(AlphaChannel::new("hole", plane));
        Ok(())
    })
    .map_err(err)?;
    s.execute("select.loadSelection", json!({"channel": {"name": "hole"}})).map_err(err)?;
    Ok(s)
}

/// The Fill dialog's Content-Aware params for `method`, with its starting settings.
fn params(method: &str) -> Value {
    json!({
        "sampling": "rectangular",
        "colorAdaptation": "default",
        "method": method,
        "detail": true,
        "stamp": true,
        "matchDepth": true,
        "edgeGrow": 2.0,
        "edgeFeather": 8.0,
        "output": "current",
    })
}

/// One fill of the session's selection with `method`: the window it covers, its fill area and
/// its pixels as RGB, and how long it took.
struct Filled {
    rect: Rect,
    hole: Vec<bool>,
    rgb: Vec<f32>,
    ms: f64,
}

fn fill(s: &Session, method: &str) -> Result<Filled, String> {
    let st = s.active().ok_or("no document")?;
    let doc = &st.doc;
    let id = st.active_layer.ok_or("no layer")?;
    let t = Instant::now();
    let plan = content_aware_plan(doc, id, &params(method), "edit.contentAwareFill").map_err(|e| e.to_string())?;
    let guide = model_guide(&plan, &Interrupt::NONE).map_err(|e| e.to_string())?;
    if plan.model.is_some() && guide.is_none() {
        return Err("the model gave no fill".into());
    }
    let (w, h) = (plan.rect.width() as usize, plan.rect.height() as usize);
    let n = plan.fmt.channels();
    let with = plan.model.as_ref().zip(guide.as_ref());
    let out = caf_fill(w, h, &plan.fmt, plan.img.clone(), &plan.hole, &plan.source, &plan.opts, with, &Interrupt::NONE).map_err(|_| "cancelled".to_string())?;
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let rgb = out.chunks_exact(n).flat_map(|q| photocraft_raster::to_rgba(&plan.fmt, q)[..3].to_vec()).collect();
    Ok(Filled { rect: plan.rect, hole: plan.hole, rgb, ms })
}

/// The photo's RGB over `r`.
fn crop_rgb(p: &Photo, r: Rect) -> Vec<f32> {
    let mut v = Vec::with_capacity(r.width() as usize * r.height() as usize * 3);
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            let q = &p.rgba[(y as usize * p.w + x as usize) * 4..][..4];
            v.extend_from_slice(&q[..3]);
        }
    }
    v
}

fn luma(rgb: &[f32]) -> Vec<f32> {
    rgb.as_chunks::<3>().0.iter().map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect()
}

/// Mean SSIM of the luminance over the hole's pixels, each in a 7 × 7 window (clipped at the edges).
fn ssim(a: &[f32], b: &[f32], hole: &[bool], w: usize, h: usize) -> f64 {
    let (la, lb) = (luma(a), luma(b));
    let (c1, c2) = (0.01f64.powi(2), 0.03f64.powi(2));
    let (mut sum, mut n) = (0.0f64, 0usize);
    for (i, _) in hole.iter().enumerate().filter(|(_, m)| **m) {
        let (x, y) = (i % w, i / w);
        let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut k) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for yy in y.saturating_sub(3)..(y + 4).min(h) {
            for xx in x.saturating_sub(3)..(x + 4).min(w) {
                let (va, vb) = (f64::from(la[yy * w + xx]), f64::from(lb[yy * w + xx]));
                sa += va;
                sb += vb;
                saa += va * va;
                sbb += vb * vb;
                sab += va * vb;
                k += 1.0;
            }
        }
        let (ma, mb) = (sa / k, sb / k);
        let (va, vb, cov) = (saa / k - ma * ma, sbb / k - mb * mb, sab / k - ma * mb);
        sum += ((2.0 * ma * mb + c1) * (2.0 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
        n += 1;
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn gradient(img: &[f32], w: usize, h: usize) -> Vec<f32> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let (r, d) = ((y * w + (x + 1).min(w - 1)) * 3, ((y + 1).min(h - 1) * w + x) * 3);
            (0..3).map(|c| (img[r + c] - img[i * 3 + c]).abs() + (img[d + c] - img[i * 3 + c]).abs()).sum()
        })
        .collect()
}

fn mean_over(v: &[f32], m: &[bool]) -> f64 {
    let (s, n) = v.iter().zip(m).filter(|(_, m)| **m).fold((0.0f64, 0usize), |(s, n), (v, _)| (s + f64::from(*v), n + 1));
    if n == 0 { 0.0 } else { s / n as f64 }
}

/// `(texture, seam)`: the fill's mean gradient over the photo's, in the hole and on the band
/// 2 px either side of its edge.
fn texture_and_seam(orig: &[f32], out: &[f32], hole: &[bool], w: usize, h: usize) -> (f64, f64) {
    let (go, gf) = (gradient(orig, w, h), gradient(out, w, h));
    let grown = photocraft_algo::remove::dilate(w, h, hole, 2);
    let outside: Vec<bool> = hole.iter().map(|m| !*m).collect();
    let inner: Vec<bool> = photocraft_algo::remove::dilate(w, h, &outside, 2).iter().map(|o| !*o).collect();
    let band: Vec<bool> = grown.iter().zip(&inner).map(|(g, i)| *g && !*i).collect();
    let ratio = |m: &[bool]| {
        let o = mean_over(&go, m);
        if o > 0.0 { mean_over(&gf, m) / o } else { 1.0 }
    };
    (ratio(hole), ratio(&band))
}

fn write_png(path: &Path, data: &[f32], w: usize, h: usize) {
    match Image::from_normalized(w as u32, h as u32, ChannelLayout::Rgb, SampleType::U8, data) {
        Ok(img) => match photocraft_codecs::encode(&img, Format::Png, &EncodeOptions::default()) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(path, bytes) {
                    eprintln!("{}: {e}", path.display());
                }
            }
            Err(e) => eprintln!("{}: {e:?}", path.display()),
        },
        Err(e) => eprintln!("{}: {e:?}", path.display()),
    }
}

/// A sheet of panels side by side, each `pw × ph` RGB, with a 4 px grey gap.
fn sheet(panels: &[Vec<f32>], pw: usize, ph: usize) -> (Vec<f32>, usize) {
    let gap = 4;
    let sw = panels.len() * (pw + gap) - gap;
    let mut out = vec![0.5f32; sw * ph * 3];
    for (k, p) in panels.iter().enumerate() {
        for y in 0..ph {
            let dst = (y * sw + k * (pw + gap)) * 3;
            out[dst..dst + pw * 3].copy_from_slice(&p[y * pw * 3..(y + 1) * pw * 3]);
        }
    }
    (out, sw)
}

/// `rgb` (`w` wide) over `r`, a rectangle in its coordinates.
fn sub(rgb: &[f32], w: usize, r: (usize, usize, usize, usize)) -> Vec<f32> {
    (r.1..r.3).flat_map(|y| rgb[(y * w + r.0) * 3..(y * w + r.2) * 3].iter().copied()).collect()
}

/// The photo's pixels over `rect` with the hole shown in magenta.
fn marked(p: &Photo, rect: Rect, hole: &[bool]) -> Vec<f32> {
    let mut v = crop_rgb(p, rect);
    for (i, _) in hole.iter().enumerate().filter(|(_, m)| **m) {
        v[i * 3..i * 3 + 3].copy_from_slice(&[1.0, 0.0, 1.0]);
    }
    v
}

#[derive(Default)]
struct Sums {
    ssim: f64,
    texture: f64,
    seam: f64,
    ms: f64,
    n: usize,
}

impl Sums {
    fn add(&mut self, ssim: f64, texture: f64, seam: f64, ms: f64) {
        self.ssim += ssim;
        self.texture += texture;
        self.seam += seam;
        self.ms += ms;
        self.n += 1;
    }

    fn line(&self, label: &str) -> String {
        let n = self.n.max(1) as f64;
        format!("{label:<28} {:>4} {:>7.4} {:>8.3} {:>6.3} {:>9.0}", self.n, self.ssim / n, self.texture / n, self.seam / n, self.ms / n)
    }
}

/// The panel crop of a hole: its bounds and half their size around them, within the window.
fn panel_rect(hole: &[bool], w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, _) in hole.iter().enumerate().filter(|(_, m)| **m) {
        let (x, y) = (i % w, i / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
    }
    if x1 == 0 {
        return None;
    }
    let pad = (x1 - x0).max(y1 - y0) / 2 + 8;
    Some((x0.saturating_sub(pad), y0.saturating_sub(pad), (x1 + pad).min(w), (y1 + pad).min(h)))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: caf_quality <photos dir> <out dir> [classic,classicDepth,lama,migan,mix]";
    let dir = PathBuf::from(args.get(1).expect(usage));
    let out_dir = PathBuf::from(args.get(2).expect(usage));
    let methods: Vec<String> = args.get(3).map_or("classic,classicDepth,lama,mix", String::as_str).split(',').map(str::to_string).collect();
    std::fs::create_dir_all(&out_dir).expect("out dir");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("photos dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff")))
        .collect();
    files.sort();
    let mut csv = String::from("photo,hole,method,ssim,texture,seam,ms\n");
    let mut per_method: BTreeMap<String, Sums> = BTreeMap::new();
    let mut per_hole: BTreeMap<(String, String), Sums> = BTreeMap::new();
    let mut unavailable: BTreeMap<String, String> = BTreeMap::new();
    let started = Instant::now();
    for path in &files {
        let photo = match load(path) {
            Ok(p) => p,
            Err(e) => {
                println!("{}: can't read it, skipped ({e})", path.display());
                continue;
            }
        };
        println!("== {} ({}x{})", photo.name, photo.w, photo.h);
        let opaque: Vec<bool> = photo.rgba.as_chunks::<4>().0.iter().map(|q| q[3] > 0.99).collect();
        let mut cases: Vec<(String, Vec<bool>, bool)> = Vec::new();
        for (name, shape) in holes(&photo) {
            let m = mask(shape, photo.w, photo.h);
            // Only where the photo has pixels: a hole over a transparent one has nothing to score.
            if m.iter().zip(&opaque).all(|(h, o)| !*h || *o) && m.iter().any(|h| *h) {
                cases.push((name.to_string(), m, true));
            }
        }
        let clear = opaque.iter().filter(|o| !**o).count() as f64 / opaque.len().max(1) as f64;
        if (0.002..0.6).contains(&clear) {
            cases.push(("real".to_string(), opaque.iter().map(|o| !*o).collect(), false));
        }
        for (case, m, scored) in &cases {
            let s = match session(&photo, m) {
                Ok(s) => s,
                Err(e) => {
                    println!("  {case}: {e}");
                    continue;
                }
            };
            let mut panels: Vec<Vec<f32>> = Vec::new();
            let mut frame: Option<(Rect, (usize, usize, usize, usize))> = None;
            for method in &methods {
                if unavailable.contains_key(method) {
                    continue;
                }
                let f = match fill(&s, method) {
                    Ok(f) => f,
                    Err(e) => {
                        println!("  {case} {method}: n/a ({e})");
                        if method != "classic" && method != "classicDepth" {
                            unavailable.insert(method.clone(), e);
                        }
                        continue;
                    }
                };
                let (w, h) = (f.rect.width() as usize, f.rect.height() as usize);
                let orig = crop_rgb(&photo, f.rect);
                if *scored {
                    let q = ssim(&orig, &f.rgb, &f.hole, w, h);
                    let (tex, seam) = texture_and_seam(&orig, &f.rgb, &f.hole, w, h);
                    println!("  {case:<9} {method:<13} ssim {q:.4}  texture {tex:.3}  seam {seam:.3}  {:.0} ms", f.ms);
                    csv.push_str(&format!("{},{case},{method},{q:.5},{tex:.4},{seam:.4},{:.0}\n", photo.name, f.ms));
                    per_method.entry(method.clone()).or_default().add(q, tex, seam, f.ms);
                    let group = case.trim_end_matches("-l").trim_end_matches("-r").to_string();
                    per_hole.entry((group, method.clone())).or_default().add(q, tex, seam, f.ms);
                } else {
                    println!("  {case:<9} {method:<13} filled in {:.0} ms", f.ms);
                }
                // The sheet's frame: the first method's window and the hole's surroundings in it
                // (every method plans the same window for the same selection).
                let (rect, pr) = match frame {
                    Some(fr) => fr,
                    None => match panel_rect(&f.hole, w, h) {
                        Some(pr) => {
                            frame = Some((f.rect, pr));
                            panels.push(sub(&orig, w, pr));
                            panels.push(sub(&marked(&photo, f.rect, &f.hole), w, pr));
                            (f.rect, pr)
                        }
                        None => continue,
                    },
                };
                if rect == f.rect {
                    panels.push(sub(&f.rgb, w, pr));
                }
            }
            if let Some((_, pr)) = frame {
                let (pw, ph) = (pr.2 - pr.0, pr.3 - pr.1);
                let (data, sw) = sheet(&panels, pw, ph);
                write_png(&out_dir.join(format!("{}-{case}.png", photo.name)), &data, sw, ph);
            }
        }
    }
    std::fs::write(out_dir.join("scores.csv"), csv).expect("scores.csv");
    println!();
    println!("{:<28} {:>4} {:>7} {:>8} {:>6} {:>9}", "mean", "n", "ssim", "texture", "seam", "ms");
    for (m, s) in &per_method {
        println!("{}", s.line(m));
    }
    println!();
    for ((hole, m), s) in &per_hole {
        println!("{}", s.line(&format!("{hole} {m}")));
    }
    for (m, e) in &unavailable {
        println!("{m}: not measured ({e})");
    }
    println!("{} photos in {:.0} s; sheets and scores.csv in {}", files.len(), started.elapsed().as_secs_f64(), out_dir.display());
}
