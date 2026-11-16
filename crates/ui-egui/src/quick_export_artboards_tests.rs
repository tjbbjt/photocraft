//! #12902: File › Export › Quick Export as PNG on a document with artboards asks for a folder and
//! writes one PNG per artboard, named after it, instead of one image of the whole canvas.

use serde_json::json;

use crate::PhotocraftApp;
use crate::file_dialog::{FileDialogAnswer, FileDialogRequest};

#[test]
fn quick_export_of_artboards_asks_for_a_folder_and_writes_one_file_each() {
    let dir = tempfile::tempdir().unwrap();
    let (show, asked) = crate::file_dialog::fake(vec![Some(FileDialogAnswer::FolderTo(dir.path().to_string_lossy().into_owned()))]);
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services { file_dialog: Some(show), ..Default::default() });
    app.run("file.new", json!({"width": 200, "height": 100})).unwrap();
    app.run("layer.new.artboard", json!({"rect": [0, 0, 80, 60], "name": "Red Board"})).unwrap();
    app.run("layer.new.artboard", json!({"rect": [100, 0, 40, 30], "name": "Blue Board"})).unwrap();
    let ctx = egui::Context::default();
    crate::menus::invoke(&mut app, &ctx, "file.export.quickExportAsPng", json!({})).unwrap();
    for _ in 0..3 {
        app.poll_file_dialog(&ctx, None);
    }
    assert!(matches!(asked.borrow().as_slice(), [FileDialogRequest::Folder { .. }]), "a folder, not a file, is chosen");
    assert!(dir.path().join("Red Board.png").exists() && dir.path().join("Blue Board.png").exists());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2, "one file per artboard, nothing else");
    assert_eq!(app.ui.status, "Exported 2 files");
}
