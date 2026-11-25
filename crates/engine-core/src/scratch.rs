//! Native tile storage configuration, kept out of the preference data model.

use crate::prefs::Preferences;

/// A web session has no filesystem or memory-supervision thread.
#[cfg(target_arch = "wasm32")]
#[derive(Default)]
pub(crate) struct Runtime;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::Runtime;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError, Weak};
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    // Weak ownership: the process-wide cache cannot keep a session's worker alive.
    pub(crate) struct Runtime {
        _worker: Arc<Worker>,
    }

    impl Default for Runtime {
        fn default() -> Self {
            static SHARED: OnceLock<Mutex<Weak<Worker>>> = OnceLock::new();
            let mut shared = SHARED.get_or_init(Mutex::default).lock().unwrap_or_else(PoisonError::into_inner);
            let worker = shared.upgrade().unwrap_or_else(|| {
                let worker = Arc::new(Worker::start());
                *shared = Arc::downgrade(&worker);
                worker
            });
            Self { _worker: worker }
        }
    }

    struct Worker {
        stop: Arc<(Mutex<bool>, Condvar)>,
        thread: Option<JoinHandle<()>>,
    }

    impl Worker {
        fn start() -> Self {
            let stop = Arc::new((Mutex::new(false), Condvar::new()));
            let flag = stop.clone();
            let thread = std::thread::Builder::new()
                .name("pc-memory-residency".into())
                .spawn(move || {
                    let _maintenance = photocraft_raster::spill::background_maintenance();
                    let mut measured = Instant::now();
                    loop {
                        let (lock, wake) = &*flag;
                        let stopped = lock.lock().unwrap_or_else(PoisonError::into_inner);
                        let (stopped, _) = wake.wait_timeout_while(stopped, Duration::from_millis(100), |stop| !*stop).unwrap_or_else(PoisonError::into_inner);
                        if *stopped {
                            break;
                        }
                        drop(stopped);
                        if measured.elapsed() >= Duration::from_secs(2) {
                            photocraft_raster::memory::refresh();
                            measured = Instant::now();
                        } else {
                            // Between measurements the budget follows the heap's live count.
                            photocraft_raster::memory::refresh_fast();
                        }
                        photocraft_raster::spill::maintain_budget();
                    }
                })
                .ok();
            // If spawning fails, raster's synchronous fallback still enforces the budget.
            Self { stop, thread }
        }
    }

    impl Drop for Worker {
        fn drop(&mut self) {
            let (lock, wake) = &*self.stop;
            *lock.lock().unwrap_or_else(PoisonError::into_inner) = true;
            wake.notify_all();
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn workers_stop_join_and_can_restart() {
            for _ in 0..3 {
                let worker = Worker::start();
                assert!(worker.thread.is_some());
                let stop = worker.stop.clone();
                let began = Instant::now();
                drop(worker);
                assert!(*stop.0.lock().unwrap());
                assert!(began.elapsed() < Duration::from_secs(2));
            }
        }

        #[test]
        fn sessions_share_supervision_without_static_ownership() {
            let a = Runtime::default();
            let b = Runtime::default();
            assert!(Arc::ptr_eq(&a._worker, &b._worker));
            let weak = Arc::downgrade(&a._worker);
            drop((a, b));
            // Other engine tests can hold the shared runtime concurrently; a private worker
            // proves the same ownership contract without depending on their timing.
            let worker = Arc::new(Worker::start());
            let private = Arc::downgrade(&worker);
            drop(worker);
            assert!(private.upgrade().is_none());
            drop(weak);
        }

        #[test]
        fn preferences_use_fixed_scratch_capacity_in_an_isolated_process() {
            const CHILD: &str = "PHOTOCRAFT_SCRATCH_CAPACITY_TEST_CHILD";
            if std::env::var_os(CHILD).is_none() {
                let status = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "scratch::native::tests::preferences_use_fixed_scratch_capacity_in_an_isolated_process"])
                    .env(CHILD, "1")
                    .status()
                    .unwrap();
                assert!(status.success(), "isolated scratch test failed: {status}");
                return;
            }

            // If configure uses this query even once, it pauses for lack of room. This catches an
            // override that is missing or is installed after configuring the scratch directory.
            photocraft_raster::spill::set_volume_space_for_tests(|_| Some((0, 64 << 30)));
            super::super::configure(&crate::prefs::Preferences::default());
            assert!(!photocraft_raster::spill::paused());
            let room = (32_u64 << 30) - (64_u64 << 30) / 20;
            assert_eq!(photocraft_raster::spill::scratch_capacity(), Some(room - room / 64));
        }
    }
}

/// The ceiling Auto Memory Usage asks for: the most `memory::budgets` takes, so what the system
/// has free (with its own safety margin) is the limit.
const AUTO_PERCENT: u8 = 95;

pub(crate) fn configure(prefs: &Preferences) {
    #[cfg(all(test, not(target_arch = "wasm32")))]
    photocraft_raster::spill::use_test_volume_space();
    // Memory Usage left at its default means Auto: only the hardware bounds the process (the
    // RAM the system has free, less a safety margin for everything else; see `memory::budgets`),
    // not a fixed 8 GB on a machine with 32 or 128. A percentage or size the user chose is a cap
    // below that, as Photoshop's Memory Usage is.
    let auto = prefs.performance.memory_usage_percent.is_none() && prefs.performance.memory_usage_mb == crate::prefs::DEFAULT_MEMORY_USAGE_MB;
    let budget = photocraft_raster::memory::configure(photocraft_raster::memory::Policy {
        percent: if auto { Some(AUTO_PERCENT) } else { prefs.performance.memory_usage_percent },
        // The web build's cap applies here too (effective_memory_usage_mb).
        absolute_bytes: u64::from(prefs.performance.effective_memory_usage_mb()).saturating_mul(1 << 20),
    });
    #[cfg(not(target_arch = "wasm32"))]
    {
        let candidates = disk_choice::candidates(&prefs.scratch_disks.disks);
        // An unusable or full directory pauses eviction without dropping resident pixels and
        // is retried with backoff; History's byte limit applies meanwhile (sync_scratch_state).
        // The reason is exposed in ui.inspect and the application's status bar.
        let tiles = if std::env::var_os("PHOTOCRAFT_TILE_BUDGET_MB").is_some() {
            prefs.performance.tile_budget_bytes()
        } else {
            usize::try_from(budget.tile_bytes).unwrap_or(usize::MAX)
        };
        let limit = prefs.scratch_disks.size_limit_mb;
        photocraft_raster::spill::set_size_cap((limit > 0).then(|| u64::from(limit).saturating_mul(1 << 20)));
        // The first that takes the scratch file wins; when none does, the first listed keeps
        // the paused state and its reason (or spilling stays off with nothing listed).
        if !candidates.iter().any(|dir| photocraft_raster::spill::configure(Some(dir), tiles).is_ok()) {
            let _ = photocraft_raster::spill::configure(candidates.first().map(std::path::PathBuf::as_path), tiles);
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (prefs, budget);
}

/// Which folders can hold the scratch file, in the order to try them.
#[cfg(not(target_arch = "wasm32"))]
mod disk_choice {
    use std::path::{Path, PathBuf};

    use crate::prefs::ScratchDisk;

    /// The enabled scratch disks in the order listed, as Photoshop goes down its scratch disks,
    /// then the app's own cache folder; skipping any that can't help. A folder whose parent is
    /// missing (a drive that isn't mounted, or that a Flatpak can't see) is skipped rather than
    /// created on the system disk, and so is a folder held in RAM (tmpfs, as the system temp
    /// folder is on Fedora and inside a Flatpak): a scratch file there frees no memory, it only
    /// moves the tiles to another part of RAM. The cache folder is the fallback, so a machine
    /// whose only listed disk is "(system temp)" still spills to a real disk.
    pub(super) fn candidates(disks: &[ScratchDisk]) -> Vec<PathBuf> {
        candidates_with(disks, cache_base().map(|base| own_folder(&base)))
    }

    /// [`candidates`] with `cache` as the app's own folder.
    pub(super) fn candidates_with(disks: &[ScratchDisk], cache: Option<PathBuf>) -> Vec<PathBuf> {
        let mounts = std::fs::read_to_string("/proc/self/mounts").unwrap_or_default();
        let listed = disks.iter().filter(|d| d.enabled && !d.path.trim().is_empty()).map(|d| resolve(d.path.trim()));
        listed.chain(cache).filter(|dir| reachable(dir) && !ram_backed(&mounts, dir)).collect()
    }

    fn resolve(path: &str) -> PathBuf {
        if path == "(system temp)" { std::env::temp_dir() } else { PathBuf::from(path) }
    }

    /// `$XDG_CACHE_HOME` (inside a Flatpak, the app's own cache folder, on the home disk), or
    /// `~/.cache`.
    fn cache_base() -> Option<PathBuf> {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
    }

    /// `photocraft/scratch` in the cache folder `base`, made if it isn't there. It is the app's own
    /// folder, so it is created rather than taken for a drive that isn't mounted: in a fresh
    /// Flatpak only the cache folder itself exists, and without this the scratch disk was off.
    pub(super) fn own_folder(base: &Path) -> PathBuf {
        let dir = base.join("photocraft").join("scratch");
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    /// The folder exists, or its parent does (it is created when the scratch file opens).
    fn reachable(dir: &Path) -> bool {
        dir.is_dir() || dir.parent().is_some_and(Path::is_dir)
    }

    /// Whether `dir` lies on a filesystem held in memory, by the longest mount point containing
    /// it in `mounts` (`/proc/self/mounts`). Unknown (no mount table): not.
    pub(super) fn ram_backed(mounts: &str, dir: &Path) -> bool {
        let dir = std::fs::canonicalize(dir)
            .or_else(|_| dir.parent().map_or(Err(()), |p| std::fs::canonicalize(p).map_err(|_| ())))
            .unwrap_or_else(|_| dir.to_path_buf());
        mount_type(mounts, &dir).is_some_and(|t| matches!(t.as_str(), "tmpfs" | "ramfs"))
    }

    /// The filesystem type of the longest mount point in `mounts` that contains `dir`.
    pub(super) fn mount_type(mounts: &str, dir: &Path) -> Option<String> {
        mounts
            .lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let (_, point, kind) = (fields.next()?, fields.next()?, fields.next()?);
                // The table escapes spaces and other separators as octal (`\040`).
                let point = PathBuf::from(unescape(point));
                dir.starts_with(&point).then(|| (point.components().count(), kind.to_string()))
            })
            .max_by_key(|(depth, _)| *depth)
            .map(|(_, kind)| kind)
    }

    fn unescape(field: &str) -> String {
        let mut out = String::with_capacity(field.len());
        let mut rest = field;
        while let Some(i) = rest.find('\\') {
            out.push_str(&rest[..i]);
            let code = rest.get(i + 1..i + 4).and_then(|o| u8::from_str_radix(o, 8).ok());
            match code {
                Some(byte) => {
                    out.push(char::from(byte));
                    rest = &rest[i + 4..];
                }
                None => {
                    out.push('\\');
                    rest = &rest[i + 1..];
                }
            }
        }
        out.push_str(rest);
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const MOUNTS: &str = "/dev/nvme0n1p3 / btrfs rw 0 0\ntmpfs /tmp tmpfs rw 0 0\n/dev/sda1 /run/media/pce/My\\040Drive ext4 rw 0 0\n";

        #[test]
        fn the_longest_mount_point_decides() {
            assert_eq!(mount_type(MOUNTS, Path::new("/tmp/photocraft")).as_deref(), Some("tmpfs"));
            assert_eq!(mount_type(MOUNTS, Path::new("/home/pce/.cache")).as_deref(), Some("btrfs"));
            assert_eq!(mount_type(MOUNTS, Path::new("/run/media/pce/My Drive/scratch")).as_deref(), Some("ext4"));
            // A sibling whose name only starts the same isn't inside it.
            assert_eq!(mount_type(MOUNTS, Path::new("/tmpx")).as_deref(), Some("btrfs"));
        }

        #[test]
        fn ram_backed_folders_and_unmounted_drives_are_skipped() {
            let real = std::env::temp_dir().join(format!("pc-scratch-choice-{}", std::process::id()));
            std::fs::create_dir_all(&real).unwrap();
            let disks = [
                ScratchDisk { path: "/nonexistent-drive/_SCRATCH".into(), enabled: true },
                ScratchDisk { path: real.display().to_string(), enabled: false },
                ScratchDisk { path: real.display().to_string(), enabled: true },
            ];
            let got = candidates(&disks);
            assert!(!got.iter().any(|d| d.starts_with("/nonexistent-drive")), "an unmounted drive is skipped: {got:?}");
            let mounts = std::fs::read_to_string("/proc/self/mounts").unwrap_or_default();
            assert_eq!(got.first() == Some(&real), !ram_backed(&mounts, &real), "a listed disk comes first unless it's in RAM: {got:?}");
            assert!(got.iter().all(|d| !ram_backed(&mounts, d)), "nothing held in RAM: {got:?}");
            let _ = std::fs::remove_dir(&real);
        }

        /// The cache folder is used even when its `photocraft` folder doesn't exist yet (a fresh
        /// Flatpak): it is made, not skipped as an unmounted drive.
        #[test]
        fn the_cache_folder_is_made_when_missing() {
            let base = std::env::temp_dir().join(format!("pc-scratch-cache-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&base);
            std::fs::create_dir_all(&base).unwrap();
            let own = own_folder(&base);
            assert!(own.is_dir(), "made: {own:?}");
            let got = candidates_with(&[], Some(own.clone()));
            let mounts = std::fs::read_to_string("/proc/self/mounts").unwrap_or_default();
            assert_eq!(got.contains(&own), !ram_backed(&mounts, &own), "the cache folder is a candidate unless it's in RAM: {got:?}");
            let _ = std::fs::remove_dir_all(&base);
        }
    }
}
