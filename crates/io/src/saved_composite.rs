//! The composite a PSD/PSB file stores beside its layers (Preferences › File Handling › Maximize
//! PSD and PSB File Compatibility), read at a reduced size: what a large document can show the
//! moment it opens, while its canvas is drawn layer by layer.
//!
//! Only the rows and samples kept are decoded: a 1.1 gigapixel PSB read to a 4096-pixel side
//! decodes about one row in ten and keeps one sample in ten of each, so it costs tens of
//! megabytes, not the gigabytes of the full composite. Files whose composite is a placeholder
//! (saved without Maximize Compatibility), ZIP-compressed composites (which can't be read a row at
//! a time), 32-bit and colour modes other than RGB and Grayscale have none to show here.

use photocraft_psd::PsdFile;
use photocraft_psd::compression::{Compression, packbits};
use photocraft_psd::header::{ColorMode, Version};

use crate::IoError;

/// A file's stored composite, reduced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavedComposite {
    /// Size in pixels of the reduced image.
    pub width: u32,
    pub height: u32,
    /// The document's size: each reduced pixel stands for a `factor`-sided square of it.
    pub doc_width: u32,
    pub doc_height: u32,
    pub factor: u32,
    /// Straight (unpremultiplied) RGBA8, row by row.
    pub rgba: Vec<u8>,
}

/// The facts about a file's composite that decoding it needs.
struct Layout {
    width: u32,
    height: u32,
    planes: usize,
    depth: u16,
    psb: bool,
    /// Colour channels read: 3 (RGB) or 1 (Grayscale).
    color: usize,
    /// The plane holding the composite's transparency, if it has one.
    alpha: Option<usize>,
    compression: Compression,
}

/// Reads `path`'s stored composite reduced so its longer side is at most `max_side`. `Ok(None)`
/// when the file has none to show (see the module notes); `cancelled` is asked between rows.
///
/// Only the header, the image resources (for the "real merged data" flag) and the start of the
/// layer section (for its layer count, whose sign marks a transparent composite) are read before
/// seeking past the layers to the composite: parsing the layer records of a gigapixel PSB took
/// seconds and isn't needed here.
#[cfg(any(unix, windows))]
pub fn saved_composite_at(path: &std::path::Path, max_side: u32, cancelled: &(dyn Fn() -> bool + Sync)) -> Result<Option<SavedComposite>, IoError> {
    use photocraft_psd::source::ReadAt;
    let failed = |e: String| IoError::Read(format!("{}: {e}", path.display()));
    let src = photocraft_psd::source::FileSource::open(path).map_err(|e| failed(e.to_string()))?;
    let total = src.len();
    let at = |offset: u64, len: usize| -> Result<Vec<u8>, IoError> {
        let mut buf = vec![0u8; len];
        src.read_at(offset, &mut buf).map_err(|e| failed(e.to_string()))?;
        Ok(buf)
    };
    let be16 = |b: &[u8]| u16::from_be_bytes([b[0], b[1]]);
    let be32 = |b: &[u8]| u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
    let h = at(0, 26)?;
    if &h[0..4] != b"8BPS" {
        return Ok(None);
    }
    let psb = match be16(&h[4..6]) {
        1 => false,
        2 => true,
        _ => return Ok(None),
    };
    let (planes, height, width, depth, mode) = (be16(&h[12..14]), be32(&h[14..18]), be32(&h[18..22]), be16(&h[22..24]), be16(&h[24..26]));
    let color = match mode {
        3 => 3usize,
        1 => 1,
        _ => return Ok(None),
    };
    if !matches!(depth, 8 | 16) || width == 0 || height == 0 || usize::from(planes) < color {
        return Ok(None);
    }
    // Colour mode data, then the image resources.
    let mut pos = 26u64;
    pos += 4 + u64::from(be32(&at(pos, 4)?));
    let res_len = u64::from(be32(&at(pos, 4)?));
    let res_end = pos + 4 + res_len;
    let mut r = pos + 4;
    while r + 12 <= res_end {
        let head = at(r, 7)?;
        if &head[0..4] != b"8BIM" && &head[0..4] != b"MeSa" {
            break;
        }
        let id = be16(&head[4..6]);
        // A Pascal name padded to an even length, its length byte included.
        let name = 1 + u64::from(head[6]);
        let name = name + name % 2;
        let size = u64::from(be32(&at(r + 6 + name, 4)?));
        let data = r + 6 + name + 4;
        if id == 1057 && size >= 5 && at(data + 4, 1)?[0] == 0 {
            return Ok(None); // a placeholder composite (saved without Maximize Compatibility)
        }
        r = data + size + size % 2;
    }
    // The layer and mask section: its length, then the layer info's length and layer count.
    let wide = if psb { 8usize } else { 4 };
    let len_at = |p: u64| -> Result<u64, IoError> {
        let b = at(p, wide)?;
        Ok(if psb { u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) } else { u64::from(be32(&b)) })
    };
    let lm = res_end;
    let lm_len = len_at(lm)?;
    let mut alpha = None;
    if lm_len >= (wide + 2) as u64 && len_at(lm + wide as u64)? >= 2 {
        let count = i16::from_be_bytes({
            let b = at(lm + 2 * wide as u64, 2)?;
            [b[0], b[1]]
        });
        if count < 0 && usize::from(planes) > color {
            alpha = Some(color);
        }
    } else if usize::from(planes) > color {
        // No layers: an extra channel is the composite's transparency.
        alpha = Some(color);
    }
    let image = lm + wide as u64 + lm_len;
    if image + 2 > total {
        return Ok(None);
    }
    let compression = Compression::from_u16(be16(&at(image, 2)?));
    let data = image + 2;
    let layout = Layout { width, height, planes: usize::from(planes), depth, psb, color, alpha, compression };
    let read = |offset: u64, len: u64| -> Result<Vec<u8>, IoError> {
        let end = data.checked_add(offset).and_then(|o| o.checked_add(len));
        if end.is_none_or(|e| e > total) {
            return Err(IoError::Read("the stored composite couldn't be read: a row lies past the end of the file".into()));
        }
        at(data + offset, usize::try_from(len).map_err(|_| IoError::Read("a row too long to read".into()))?)
    };
    decode(&layout, &read, max_side, cancelled)
}

/// [`saved_composite_at`] for a parsed file (its composite in memory or still in its source).
pub fn saved_composite(file: &PsdFile, max_side: u32, cancelled: &(dyn Fn() -> bool + Sync)) -> Result<Option<SavedComposite>, IoError> {
    let h = &file.header;
    let color = match h.color_mode {
        ColorMode::Rgb => 3usize,
        ColorMode::Grayscale => 1,
        _ => return Ok(None),
    };
    if !matches!(h.depth, 8 | 16) || h.width == 0 || h.height == 0 || usize::from(h.channels) < color {
        return Ok(None);
    }
    if file.has_real_merged_data() == Some(false) {
        return Ok(None);
    }
    let layout = Layout {
        width: h.width,
        height: h.height,
        planes: usize::from(h.channels),
        depth: h.depth,
        psb: h.version == Version::Psb,
        color,
        alpha: file.merged_has_alpha().then_some(color),
        compression: file.image_data.compression,
    };
    let read = |offset: u64, len: u64| -> Result<Vec<u8>, IoError> {
        let failed = |e: String| IoError::Read(format!("the stored composite couldn't be read: {e}"));
        match &file.image_data.stored {
            Some(s) => {
                if offset.checked_add(len).is_none_or(|end| end > s.len) {
                    return Err(failed("a row lies past the end of the image data".into()));
                }
                s.source.read_vec(s.offset + offset, len).map_err(|e| failed(e.to_string()))
            }
            None => {
                let (start, end) = (usize::try_from(offset).ok(), usize::try_from(offset.saturating_add(len)).ok());
                match (start, end) {
                    (Some(a), Some(b)) if b <= file.image_data.data.len() => Ok(file.image_data.data[a..b].to_vec()),
                    _ => Err(failed("a row lies past the end of the image data".into())),
                }
            }
        }
    };
    decode(&layout, &read, max_side, cancelled)
}

/// Decodes the composite `l` describes, reading its encoded bytes (from the start of the image
/// data, after the compression field) through `read`.
fn decode(
    l: &Layout,
    read: &(dyn Fn(u64, u64) -> Result<Vec<u8>, IoError> + Sync),
    max_side: u32,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<Option<SavedComposite>, IoError> {
    let compression = l.compression;
    if !matches!(compression, Compression::Raw | Compression::Rle) {
        return Ok(None);
    }
    let (color, alpha) = (l.color, l.alpha);
    let (w, hgt) = (l.width as usize, l.height as usize);
    let planes = l.planes;
    let bps = usize::from(l.depth / 8);
    let row_bytes = w * bps;
    let factor = l.width.max(l.height).div_ceil(max_side.max(1)).max(1);
    let f = factor as usize;
    let (ow, oh) = (w.div_ceil(f), hgt.div_ceil(f));
    // Where each encoded row of each plane starts and how long it is.
    let rows = planes * hgt;
    let (table_len, starts, lens) = if compression == Compression::Rle {
        let cs: usize = if l.psb { 4 } else { 2 };
        let table = read(0, (rows * cs) as u64)?;
        let lens: Vec<u64> = table
            .chunks_exact(cs)
            .map(|c| if cs == 4 { u64::from(u32::from_be_bytes([c[0], c[1], c[2], c[3]])) } else { u64::from(u16::from_be_bytes([c[0], c[1]])) })
            .collect();
        let mut starts = Vec::with_capacity(rows);
        let mut at = 0u64;
        for l in &lens {
            starts.push(at);
            at = at.saturating_add(*l);
        }
        ((rows * cs) as u64, starts, lens)
    } else {
        (0, Vec::new(), Vec::new())
    };
    let row_of = |plane: usize, y: usize| -> Result<Vec<u8>, IoError> {
        let i = plane * hgt + y;
        if compression == Compression::Rle {
            let packed = read(table_len + starts[i], lens[i])?;
            packbits::decode(&packed, row_bytes).map_err(|e| IoError::Read(format!("the stored composite couldn't be decoded: {e}")))
        } else {
            read((i * row_bytes) as u64, row_bytes as u64)
        }
    };
    // The high byte of a 16-bit sample, the sample itself at 8 bits.
    let sample = |row: &[u8], x: usize| row[x * bps];
    // Output rows `range`, as RGBA8.
    let band = |range: std::ops::Range<usize>| -> Result<Vec<u8>, IoError> {
        let mut out = Vec::new();
        out.try_reserve_exact(range.len() * ow * 4).map_err(|_| IoError::Unsupported("not enough memory for the stored composite's preview".into()))?;
        for oy in range {
            if cancelled() {
                return Err(IoError::Cancelled);
            }
            let y = (oy * f + f / 2).min(hgt - 1);
            let colour: Vec<Vec<u8>> = (0..color).map(|p| row_of(p, y)).collect::<Result<_, _>>()?;
            let a_row = alpha.map(|p| row_of(p, y)).transpose()?;
            for ox in 0..ow {
                let x = (ox * f + f / 2).min(w - 1);
                let a = a_row.as_deref().map_or(255u8, |r| sample(r, x));
                let mut px = [0u8; 3];
                for (c, out) in px.iter_mut().enumerate() {
                    let v = sample(&colour[c.min(color - 1)], x);
                    // The stored composite is matted against white where it is translucent.
                    *out = if a == 255 || a == 0 { v } else { unmatte(v, a) };
                }
                out.extend_from_slice(&[px[0], px[1], px[2], a]);
            }
        }
        Ok(out)
    };
    // Bands of rows on every core: each row is read and decoded on its own, and a gigapixel
    // file's thousands of rows took seconds one after another.
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get).clamp(1, 16);
    let per = oh.div_ceil(threads).max(1);
    let band = &band;
    let bands: Vec<Result<Vec<u8>, IoError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..oh).step_by(per).map(|start| scope.spawn(move || band(start..(start + per).min(oh)))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(IoError::Unsupported("the stored composite's reader stopped".into())))).collect()
    });
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(ow * oh * 4).map_err(|_| IoError::Unsupported("not enough memory for the stored composite's preview".into()))?;
    for b in bands {
        rgba.extend_from_slice(&b?);
    }
    Ok(Some(SavedComposite { width: ow as u32, height: oh as u32, doc_width: l.width, doc_height: l.height, factor, rgba }))
}

/// The colour `v` held before it was matted against white at alpha `a`.
fn unmatte(v: u8, a: u8) -> u8 {
    let (v, a) = (f32::from(v), f32::from(a) / 255.0);
    ((v - 255.0 * (1.0 - a)) / a).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::Document;

    fn psd_bytes(doc: &Document, psb: bool) -> Vec<u8> {
        crate::export(doc, if psb { "psb" } else { "psd" }, &crate::ExportOptions::default()).expect("export").bytes
    }

    fn two_tone(w: u32, h: u32) -> Document {
        let mut doc = Document::new("t", photocraft_doc::Size::new(w, h), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
        let mut l = photocraft_doc::Layer::raster("a", photocraft_doc::PixelFormat::RGBA8);
        let s = l.surface_mut().unwrap();
        s.fill_rect(photocraft_geom::Rect::new(0, 0, w as i32 / 2, h as i32), &[1.0, 0.0, 0.0, 1.0]);
        s.fill_rect(photocraft_geom::Rect::new(w as i32 / 2, 0, w as i32, h as i32), &[0.0, 0.0, 1.0, 1.0]);
        doc.layers.push(l);
        doc
    }

    #[test]
    fn a_layered_file_shows_its_stored_composite_reduced() {
        for psb in [false, true] {
            let bytes = psd_bytes(&two_tone(400, 100), psb);
            let file = PsdFile::from_bytes(&bytes).unwrap();
            let c = saved_composite(&file, 100, &|| false).unwrap().expect("a composite to show");
            assert_eq!((c.width, c.height, c.factor), (100, 25, 4), "psb {psb}");
            assert_eq!((c.doc_width, c.doc_height), (400, 100));
            let px = |x: u32, y: u32| &c.rgba[((y * c.width + x) * 4) as usize..][..4];
            assert_eq!(px(10, 10), [255, 0, 0, 255], "left half red");
            assert_eq!(px(90, 10), [0, 0, 255, 255], "right half blue");
        }
    }

    /// Reading from the file (header and resources only, then a seek past the layers) gives the
    /// same picture as reading the parsed file.
    #[cfg(any(unix, windows))]
    #[test]
    fn reading_the_file_directly_matches_the_parsed_file() {
        for psb in [false, true] {
            let bytes = psd_bytes(&two_tone(400, 100), psb);
            let path = std::env::temp_dir().join(format!("saved-composite-{}-{psb}.psd", std::process::id()));
            std::fs::write(&path, &bytes).unwrap();
            let direct = saved_composite_at(&path, 100, &|| false);
            let _ = std::fs::remove_file(&path);
            let parsed = saved_composite(&PsdFile::from_bytes(&bytes).unwrap(), 100, &|| false).unwrap();
            assert_eq!(direct.unwrap(), parsed, "psb {psb}");
        }
    }

    #[test]
    fn a_composite_already_small_enough_is_read_whole() {
        let bytes = psd_bytes(&two_tone(40, 10), false);
        let file = PsdFile::from_bytes(&bytes).unwrap();
        let c = saved_composite(&file, 4096, &|| false).unwrap().unwrap();
        assert_eq!((c.width, c.height, c.factor), (40, 10, 1));
    }

    #[test]
    fn a_cancelled_read_stops() {
        let bytes = psd_bytes(&two_tone(40, 10), false);
        let file = PsdFile::from_bytes(&bytes).unwrap();
        assert!(matches!(saved_composite(&file, 4096, &|| true), Err(IoError::Cancelled)));
    }

    #[test]
    fn damaged_image_data_is_an_error_not_a_panic() {
        let bytes = psd_bytes(&two_tone(40, 10), false);
        let mut file = PsdFile::from_bytes(&bytes).unwrap();
        file.image_data.compression = Compression::Rle;
        file.image_data.data.truncate(8);
        file.image_data.stored = None;
        assert!(saved_composite(&file, 4096, &|| false).is_err());
    }

    #[test]
    fn unmatting_restores_the_colour_under_the_alpha() {
        // A channel at 0 under 50% alpha, matted on white, reads 128; ~0 comes back.
        assert_eq!(unmatte(128, 128), 2);
        assert_eq!(unmatte(255, 128), 255);
    }
}
