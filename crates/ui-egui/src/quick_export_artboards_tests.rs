//! #12902: File › Export › Quick Export as PNG on a document with artboards asks for a folder and
//! writes one PNG per artboard, named after it, instead of one image of the whole canvas.

use std::path::PathBuf;
use std::sync::Arc;

use craft::host::{PickKind, Picked, ScriptedFiles};
use serde_json::json;

use crate::PhotocraftApp;

/// A document with two artboards whose file dialogs `files` answers.
fn app_with_boards(files: &Arc<ScriptedFiles>) -> Box<PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services { files: Some(files.clone()), ..Default::default() });
    app.run("file.new", json!({"width": 200, "height": 100})).unwrap();
    app.run("layer.new.artboard", json!({"rect": [0, 0, 80, 60], "name": "Red Board"})).unwrap();
    app.run("layer.new.artboard", json!({"rect": [100, 0, 40, 30], "name": "Blue Board"})).unwrap();
    app
}

/// Quick Export as PNG from the menu, the folder dialog answered with `dir`.
fn quick_export_into(app: &mut PhotocraftApp, files: &ScriptedFiles, dir: &std::path::Path) {
    let ctx = egui::Context::default();
    crate::menus::invoke(app, &ctx, "file.export.quickExportAsPng", json!({})).unwrap();
    let asked = files.requests().pop().expect("a dialog was asked for");
    assert_eq!(asked.kind, PickKind::Folder, "a folder, not a file, is chosen");
    files.answer(Picked::Paths(vec![PathBuf::from(dir)])).expect("a dialog is open");
    app.poll_file_dialog();
}

#[test]
fn quick_export_of_artboards_asks_for_a_folder_and_writes_one_file_each() {
    let dir = tempfile::tempdir().unwrap();
    let files = Arc::new(ScriptedFiles::held());
    let mut app = app_with_boards(&files);
    quick_export_into(&mut app, &files, dir.path());
    assert!(dir.path().join("Red Board.png").exists() && dir.path().join("Blue Board.png").exists());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2, "one file per artboard, nothing else");
    assert_eq!(app.ui.status, "Exported 2 files");
}

/// With background jobs on (the desktop app) the boards are written by a worker, so the window
/// keeps answering, and the status says what was written once the job lands.
#[test]
fn quick_export_of_artboards_runs_as_a_job_and_says_what_it_wrote_when_it_lands() {
    let dir = tempfile::tempdir().unwrap();
    let files = Arc::new(ScriptedFiles::held());
    let mut app = app_with_boards(&files);
    app.background_jobs = true;
    quick_export_into(&mut app, &files, dir.path());
    assert!(app.jobs.last_started.is_some(), "Quick Export started a background job");
    let ctx = egui::Context::default();
    let t = std::time::Instant::now();
    while app.ui.status != "Exported 2 files" && t.elapsed() < std::time::Duration::from_secs(60) {
        crate::jobs_ui::tick(&mut app, &ctx);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(app.ui.status, "Exported 2 files");
    assert!(dir.path().join("Red Board.png").exists() && dir.path().join("Blue Board.png").exists());
}
