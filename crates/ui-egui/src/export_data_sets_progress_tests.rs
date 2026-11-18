//! #13768: File › Export › Data Sets as Files runs as a background job, with a progress line
//! per data set ("Exporting Data Set 2 of 2: two"), instead of freezing the window for the whole
//! export. Tests, the CLI and scripts still run it inline with the same result.

use serde_json::json;

/// A 40 × 30 document with a `Badge` layer whose visibility two data sets, `one` and `two`, set.
fn session() -> photocraft_engine::Session {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30})).unwrap();
    let badge = s.execute("layer.new.layer", json!({"name": "Badge"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("image.variables.define", json!({"defs": [{"name": "showBadge", "layer": badge, "type": "visibility"}]})).unwrap();
    s.execute(
        "image.variables.dataSets",
        json!({"dataSets": [
            {"name": "one", "values": [{"variable": "showBadge", "kind": "visibility", "value": true}]},
            {"name": "two", "values": [{"variable": "showBadge", "kind": "visibility", "value": false}]},
        ]}),
    )
    .unwrap();
    s
}

#[test]
fn data_sets_export_on_a_worker_with_a_progress_line_per_set() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().into_owned();
    let mut s = session();
    let photocraft_engine::jobs::Started::Job(job) = s.start("file.export.dataSetsAsFiles", json!({"dir": path, "format": "png"})).unwrap() else {
        panic!("the export runs on a worker");
    };
    let out = s.wait_job(job).unwrap();
    assert_eq!(out["count"], 2, "{out}");
    assert!(dir.path().join("one.png").exists() && dir.path().join("two.png").exists());
    let info = s.jobs_with_recent().into_iter().find(|j| j.id == job).expect("the job is listed");
    assert_eq!(info.message, "Exporting Data Set 2 of 2: two", "the set's number, the total and its name");
}

#[test]
fn data_sets_export_inline_gives_the_same_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().into_owned();
    let mut s = session();
    let out = s.execute("file.export.dataSetsAsFiles", json!({"dir": path, "format": "png"})).unwrap();
    assert_eq!(out["count"], 2, "{out}");
    assert!(dir.path().join("one.png").exists() && dir.path().join("two.png").exists());
}

#[test]
fn a_batch_whose_names_collide_fails_before_the_job_starts() {
    let dir = tempfile::tempdir().unwrap();
    let never = dir.path().join("never");
    let mut s = session();
    assert!(s.start("file.export.dataSetsAsFiles", json!({"dir": never.to_string_lossy(), "naming": "same"})).is_err());
    assert!(!never.exists(), "nothing is written and no folder made");
}
