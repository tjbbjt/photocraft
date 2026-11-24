use super::optimizer::{optimize, web_document, web_exif, web_warnings, web_xmp};
use super::settings::WebSettings;
use super::{EngineError, Result, Session, Value, bad, join, json, other, stem, write_file};

// ---------- Export Preferences / Quick Export ----------

/// File › Export › Export Preferences…: sets the given `export.*` preferences through `prefs.set`
/// (validated like any other preference) and returns the whole section.
pub(super) fn export_preferences(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.export.exportPreferences";
    let mut values = serde_json::Map::new();
    for (k, key) in [
        ("quickExportFormat", "export.quickExportFormat"),
        ("quickExportLocation", "export.quickExportLocation"),
        ("jpegQuality", "export.jpegQuality"),
        ("webpLossless", "export.webpLossless"),
        ("webpQuality", "export.webpQuality"),
        ("metadata", "export.metadata"),
        ("convertToSrgb", "export.convertToSrgb"),
    ] {
        if let Some(v) = p.get(k) {
            values.insert(key.into(), v.clone());
        }
    }
    if !values.is_empty() {
        s.execute("prefs.set", json!({"values": values})).map_err(|e| bad(cmd, e.to_string()))?;
    }
    Ok(json!({"section": "export", "values": s.prefs().get("export").unwrap_or(Value::Null)}))
}

/// The Quick Export format's extension (Export Preferences › Format).
fn quick_export_ext(s: &Session) -> String {
    serde_json::to_value(s.prefs().export.quick_export_format).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_else(|| "png".into())
}

/// Where Quick Export writes the active document with no `path` given: `<name>.<format>` beside
/// its file, when Export Preferences › Location is Same Folder and the document has a file.
pub fn quick_export_target(s: &Session) -> Option<String> {
    let same = s.prefs().export.quick_export_location == crate::prefs::ExportLocation::SameFolder;
    let dp = s.active()?.path.as_deref().filter(|_| same)?;
    let dir = std::path::Path::new(dp).parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
    Some(join(&dir, &format!("{}.{}", stem(dp), quick_export_ext(s))))
}

/// Whether [`quick_export_target`] is the active document's own file (a document opened from
/// `photo.png` and exported as PNG), which Quick Export must not replace without asking (#6599).
pub fn quick_export_target_is_own_file(s: &Session) -> bool {
    let own = s.active().and_then(|d| d.path.as_deref());
    quick_export_target(s).zip(own).is_some_and(|(target, own)| crate::file_cmds::same_file(&target, own))
}

/// File › Export › Quick Export as <format>: the format, quality, metadata, colour space and
/// location from Export Preferences. `path` overrides the location ("ask" needs one).
///
/// A document with artboards writes one file per artboard instead, `<artboard name>.<format>`,
/// into the folder `dir` names (else the folder of `path`, else the location), as the reference
/// application does (#12902): it wrote one image of the whole canvas with every board side by side.
///
/// Flattening and encoding run as a job: a storyboard's boards took 13.7 s on the UI thread, and
/// the compositor closed the window that stopped answering.
pub(super) fn quick_export(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.export.quickExport";
    let has_boards = s.active().is_some_and(|d| !d.doc.artboards().is_empty());
    if has_boards {
        return quick_export_artboards(s, p);
    }
    let prefs = s.prefs().export.clone();
    let fmt = quick_export_ext(s);
    let path = match p.get("path").and_then(Value::as_str).filter(|v| !v.is_empty()) {
        Some(v) => v.to_string(),
        None => {
            let Some(target) = quick_export_target(s) else {
                return Err(bad(cmd, "missing `path` (Export Preferences › Location is \"ask\", or the document was never saved)"));
            };
            // The document's own file is never replaced by an export the user didn't name (#6599).
            if quick_export_target_is_own_file(s) {
                return Err(bad(cmd, format!("the export would replace the document's own file {target}; give a `path`")));
            }
            target
        }
    };
    // The name typed in the save dialog wins over Export Preferences' format: Quick Export as PNG
    // renamed to `photo.jpeg` must write a JPEG, not PNG bytes upload checks refuse (#8519).
    let fmt = match crate::file_cmds::extension(&path).as_deref() {
        None => fmt,
        Some("jpg" | "jpeg") => "jpg".into(),
        Some(e @ ("png" | "gif" | "webp" | "exr")) => e.into(),
        Some(e) => return Err(bad(cmd, format!("Quick Export writes PNG, JPEG, GIF, WebP or EXR, not .{e}; use Export As"))),
    };
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    crate::jobs::run(
        s,
        "Quick Export",
        false,
        move |ctx| {
            ctx.progress(0.0, "Quick Export");
            export_quick(&doc, &path, &fmt, &prefs, cmd)
        },
        |s, out| {
            crate::automate_cmds::fire_event(s, "export");
            Ok(out)
        },
    )
}

/// One Quick Export file: `doc` written to `path` as `fmt` with Export Preferences `prefs`.
fn export_quick(doc: &photocraft_doc::Document, path: &str, fmt: &str, prefs: &crate::prefs::Export, cmd: &str) -> Result<Value> {
    if fmt == "exr" {
        // EXR keeps floating-point scene values. The ordinary exporter converts to linear
        // light; the web/sRGB pipeline would instead clamp HDR and negative values.
        let opts = photocraft_io::ExportOptions { xmp: photocraft_io::XmpEmbed::None, ..Default::default() };
        let out = photocraft_io::export(doc, "exr", &opts).map_err(other)?;
        write_file(path, &out.bytes)?;
        return Ok(json!({"path": path, "format": fmt, "bytes": out.bytes.len(), "warnings": out.warnings}));
    }
    let metadata = serde_json::to_value(prefs.metadata).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
    let mut o = serde_json::Map::new();
    o.insert(
        "metadata".into(),
        json!(if metadata == "all" {
            "all"
        } else if metadata == "none" {
            "none"
        } else {
            "copyright"
        }),
    );
    o.insert("convertToSrgb".into(), json!(prefs.convert_to_srgb));
    o.insert("transparency".into(), json!(true));
    o.insert("matte".into(), json!("#ffffff"));
    match fmt {
        "jpg" => {
            o.insert("format".into(), json!("jpeg"));
            o.insert("quality".into(), json!(prefs.jpeg_quality.min(100)));
            // Quick Export JPGs are progressive (SOF2) at every quality, as the reference
            // application writes them; Save for Web keeps its own checkbox (#13787).
            o.insert("progressive".into(), json!(true));
        }
        "gif" => {
            o.insert("format".into(), json!("gif"));
        }
        "webp" => {
            // WebP uses the regular exporter. Its optional quality is in the Photoshop-style
            // 0–12 save scale: invert that mapping so the codec receives the chosen 1–100.
            // Keeping quality absent preserves the historical lossless Quick Export behavior.
            let st = WebSettings::from_params(&Value::Object(o.clone()), cmd)?;
            let (mut wdoc, _, _) = web_document(doc, &json!({}), &st)?;
            // The regular exporter embeds the document's XMP and EXIF as they are: give it only
            // what the metadata choice keeps, as the Save for Web formats get.
            wdoc.metadata.xmp = web_xmp(doc, st.metadata).map(std::sync::Arc::from);
            wdoc.metadata.exif = web_exif(doc, st.metadata).map(|e| std::sync::Arc::new(e.to_vec()));
            let quality = if prefs.webp_lossless { None } else { Some((f64::from(prefs.webp_quality.clamp(1, 100)) - 1.0) * 12.0 / 99.0) };
            let warnings = crate::file_cmds::save_doc(&wdoc, path, quality)?;
            return Ok(json!({"path": path, "format": fmt, "warnings": warnings}));
        }
        _ => {
            o.insert("format".into(), json!("png24"));
        }
    }
    let q = Value::Object(o);
    let st = WebSettings::from_params(&q, cmd)?;
    let (wdoc, _, _) = web_document(doc, &json!({}), &st)?;
    let buf = photocraft_compose::flatten(&wdoc);
    let xmp = web_xmp(doc, st.metadata);
    let exif = web_exif(doc, st.metadata);
    let opt = optimize(
        &buf.px,
        wdoc.size.width as usize,
        wdoc.bounds(),
        &st,
        wdoc.icc_profile.as_deref().map(|v| v.as_slice()),
        exif,
        xmp.as_deref(),
        wdoc.resolution_dpi,
        false,
    )?;
    write_file(path, &opt.bytes)?;
    Ok(json!({"path": path, "format": fmt, "bytes": opt.bytes.len(), "warnings": web_warnings(&st, exif, wdoc.resolution_dpi)}))
}

/// [`quick_export`] of a document with artboards (#12902): each artboard with pixels becomes
/// `<artboard name>.<format>` in one folder, top board of the Layers panel first. Names an earlier
/// board took (letter case ignored) get `-2`, `-3`…, as Artboards to Files numbers them. Every
/// board is made before the first file is written, and the document's own file is never replaced.
/// The boards are made and written in a job, a progress step per board.
/// `{dir?, path?}` → `{files: [{path, format, …}], format}`.
fn quick_export_artboards(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.export.quickExport";
    let prefs = s.prefs().export.clone();
    let named = |key: &str| p.get(key).and_then(Value::as_str).filter(|v| !v.is_empty()).map(str::to_string);
    let path_dir = |path: &str| std::path::Path::new(path).parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
    let (dir, fmt) = match (named("dir"), named("path")) {
        (Some(dir), _) => (dir, quick_export_ext(s)),
        // A file named in the save dialog: its folder, and its extension's format.
        (None, Some(path)) => {
            let fmt = match crate::file_cmds::extension(&path).as_deref() {
                None => quick_export_ext(s),
                Some("jpg" | "jpeg") => "jpg".into(),
                Some(e @ ("png" | "gif" | "webp" | "exr")) => e.into(),
                Some(e) => return Err(bad(cmd, format!("Quick Export writes PNG, JPEG, GIF, WebP or EXR, not .{e}; use Export As"))),
            };
            (path_dir(&path), fmt)
        }
        (None, None) => match quick_export_target(s) {
            Some(target) => (path_dir(&target), quick_export_ext(s)),
            None => return Err(bad(cmd, "missing `dir` (Export Preferences › Location is \"ask\", or the document was never saved)")),
        },
    };
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = d.doc.clone();
    let own = d.path.clone();
    crate::jobs::run(
        s,
        "Quick Export",
        false,
        move |ctx| {
            ctx.progress(0.0, "Quick Export");
            // Top board of the Layers panel first, as Artboards to Files writes them.
            let boards: Vec<photocraft_doc::Document> = doc
                .artboards()
                .into_iter()
                .rev()
                .filter_map(|(id, _, _)| crate::artboard_cmds::artboard_document(&doc, id).transpose())
                .collect::<Result<_>>()?;
            if boards.is_empty() {
                return Err(bad(cmd, "the artboards are empty"));
            }
            let names: Vec<&str> = boards.iter().map(|b| b.name.as_str()).collect();
            let paths: Vec<String> =
                crate::artboard_cmds::artboard_file_stems("", &names, false).into_iter().map(|stem| join(&dir, &format!("{stem}.{fmt}"))).collect();
            if let Some(own) = own.as_deref()
                && let Some(path) = paths.iter().find(|path| crate::file_cmds::same_file(path, own))
            {
                return Err(bad(cmd, format!("the export would replace the document's own file {path}; rename that artboard or choose another folder")));
            }
            let n = boards.len();
            let mut files = Vec::new();
            for (i, (board, path)) in boards.iter().zip(&paths).enumerate() {
                ctx.check()?;
                ctx.progress(i as f32 / n as f32, &format!("Artboard {} of {n}", i + 1));
                files.push(export_quick(board, path, &fmt, &prefs, cmd)?);
            }
            Ok(json!({"files": files, "format": fmt}))
        },
        |s, out| {
            crate::automate_cmds::fire_event(s, "export");
            Ok(out)
        },
    )
}

#[cfg(test)]
#[path = "quick_artboards_tests.rs"]
mod quick_artboards_tests;
#[cfg(test)]
#[path = "quick_jpg_progressive_tests.rs"]
mod quick_jpg_progressive_tests;
