//! #12902: Quick Export of a document with artboards writes one file per artboard, named after it,
//! instead of one image of the whole canvas with the boards side by side.

use super::*;

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("photocraft-quick-boards-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d.to_string_lossy().into_owned()
}

/// A document with two artboards, `Red Board` (80 × 60) and `Blue Board` (40 × 30), in that order.
fn session() -> Session {
    let mut s = crate::session();
    s.execute("file.new", json!({"width": 200, "height": 100, "name": "boards.psd"})).unwrap();
    s.execute("layer.new.artboard", json!({"rect": [0, 0, 80, 60], "name": "Red Board"})).unwrap();
    s.execute("layer.new.artboard", json!({"rect": [100, 0, 40, 30], "name": "Blue Board"})).unwrap();
    s
}

fn png_size(path: &str) -> (u32, u32) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[1..4], b"PNG", "{path} is a PNG");
    (u32::from_be_bytes(bytes[16..20].try_into().unwrap()), u32::from_be_bytes(bytes[20..24].try_into().unwrap()))
}

#[test]
fn each_artboard_becomes_a_file_of_its_own() {
    let dir = tmp("dir");
    let mut s = session();
    let r = s.execute("file.export.quickExport", json!({"dir": dir})).unwrap();
    let files: Vec<String> = r["files"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap().to_string()).collect();
    assert_eq!(files, [join(&dir, "Blue Board.png"), join(&dir, "Red Board.png")], "top board of the Layers panel first");
    assert_eq!(png_size(&files[0]), (40, 30), "each file is its board's size");
    assert_eq!(png_size(&files[1]), (80, 60));
    assert!(!std::path::Path::new(&join(&dir, "boards.png")).exists(), "no image of the whole canvas");
}

#[test]
fn a_file_named_in_the_save_dialog_gives_the_folder_and_format() {
    let dir = tmp("path");
    let mut s = session();
    let r = s.execute("file.export.quickExport", json!({"path": join(&dir, "whatever.jpg")})).unwrap();
    assert_eq!(r["format"], "jpg");
    assert!(std::path::Path::new(&join(&dir, "Red Board.jpg")).exists());
    assert!(std::path::Path::new(&join(&dir, "Blue Board.jpg")).exists());
    assert!(!std::path::Path::new(&join(&dir, "whatever.jpg")).exists());
}

#[test]
fn boards_with_one_name_are_numbered() {
    let dir = tmp("dup");
    let mut s = crate::session();
    s.execute("file.new", json!({"width": 200, "height": 100})).unwrap();
    s.execute("layer.new.artboard", json!({"rect": [0, 0, 20, 20], "name": "Board"})).unwrap();
    s.execute("layer.new.artboard", json!({"rect": [40, 0, 20, 20], "name": "board"})).unwrap();
    let r = s.execute("file.export.quickExport", json!({"dir": dir})).unwrap();
    let files: Vec<&str> = r["files"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap()).collect();
    assert_eq!(files, [join(&dir, "board.png"), join(&dir, "Board-2.png")]);
}

#[test]
fn a_document_without_artboards_still_writes_one_file() {
    let dir = tmp("plain");
    let mut s = crate::session();
    s.execute("file.new", json!({"width": 20, "height": 10})).unwrap();
    let path = join(&dir, "plain.png");
    let r = s.execute("file.export.quickExport", json!({"path": path})).unwrap();
    assert_eq!(r["path"], json!(path));
    assert_eq!(png_size(&path), (20, 10));
}

/// Quick Export runs as a job when started (a storyboard's boards took 13.7 s on the UI thread):
/// the files are written by the worker and the result arrives when the job lands. A document
/// without artboards does the same.
#[test]
fn a_started_quick_export_runs_as_a_job() {
    let land = |s: &mut Session, p: Value| -> Value {
        let id = match s.start("file.export.quickExport", p).unwrap() {
            crate::jobs::Started::Job(id) => id,
            crate::jobs::Started::Done(v) => panic!("expected a background job, ran inline: {v}"),
        };
        let t = std::time::Instant::now();
        loop {
            if let Some(e) = s.poll_jobs().into_iter().find(|e| e.id == id) {
                let crate::jobs::JobOutcome::Done(v) = e.outcome else { panic!("{:?}", e.outcome) };
                return v;
            }
            assert!(t.elapsed() < std::time::Duration::from_secs(60), "the export did not finish");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    };
    let dir = tmp("job");
    let mut s = session();
    let r = land(&mut s, json!({"dir": dir}));
    assert_eq!(r["files"].as_array().unwrap().len(), 2, "{r}");
    assert_eq!(png_size(&join(&dir, "Red Board.png")), (80, 60));
    let mut s = crate::session();
    s.execute("file.new", json!({"width": 30, "height": 20})).unwrap();
    let path = join(&dir, "flat.png");
    let r = land(&mut s, json!({"path": path}));
    assert_eq!(r["path"], json!(path));
    assert_eq!(png_size(&path), (30, 20));
}
