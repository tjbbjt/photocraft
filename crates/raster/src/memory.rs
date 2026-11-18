//! Process-wide memory admission and adaptive tile residency. Native measurements are safe
//! `sysinfo` calls; web clients retain their explicit MB limit without filesystem assumptions.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

pub mod system;

const MIB: u64 = 1 << 20;

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub process_resident_bytes: u64,
    /// Address-space/virtual usage; this is not a portable measure of committed memory.
    pub process_virtual_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Policy {
    /// None keeps the pre-streaming absolute MB setting.
    pub percent: Option<u8>,
    pub absolute_bytes: u64,
}

impl Default for Policy {
    fn default() -> Self {
        Self { percent: None, absolute_bytes: 8 << 30 }
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Budgets {
    pub process_ceiling_bytes: u64,
    pub effective_process_bytes: u64,
    pub tile_bytes: u64,
    pub transient_bytes: u64,
}

/// Other applications and untiled buffers count against the same ceiling. Keep both system
/// headroom and a working reserve; virtual address-space size is deliberately not treated as RSS.
pub fn budgets(policy: Policy, sample: Snapshot, resident_tiles: u64) -> Budgets {
    let ceiling = match (policy.percent, sample.total_bytes) {
        (Some(p), total) if total > 0 => total / 100 * u64::from(p.clamp(1, 95)),
        _ => policy.absolute_bytes,
    }
    .max(MIB);
    let headroom = (sample.total_bytes / 20).max(256 * MIB);
    let effective = if sample.total_bytes == 0 {
        ceiling
    } else {
        ceiling.min(sample.process_resident_bytes.saturating_add(sample.available_bytes).saturating_sub(headroom))
    }
    .max(MIB);
    let transient = (effective / 8).clamp(MIB, 2 << 30);
    let other = sample.process_resident_bytes.saturating_sub(resident_tiles);
    let tiles = effective.saturating_sub(other).saturating_sub(transient).max(MIB);
    Budgets { process_ceiling_bytes: ceiling, effective_process_bytes: effective, tile_bytes: tiles, transient_bytes: transient }
}

static POLICY: Mutex<Policy> = Mutex::new(Policy { percent: None, absolute_bytes: 8 << 30 });
static CURRENT: Mutex<(Snapshot, Budgets)> = Mutex::new((
    Snapshot { total_bytes: 0, available_bytes: 0, process_resident_bytes: 0, process_virtual_bytes: 0 },
    Budgets { process_ceiling_bytes: 8 << 30, effective_process_bytes: 8 << 30, tile_bytes: 7 << 30, transient_bytes: 1 << 30 },
));
static RESERVED: AtomicU64 = AtomicU64::new(0);
static CONFIGURATION: AtomicU64 = AtomicU64::new(0);

pub fn configuration_generation() -> u64 {
    CONFIGURATION.load(Ordering::Relaxed)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn measure() -> Snapshot {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
    static SYSTEM: OnceLock<Mutex<Option<System>>> = OnceLock::new();
    let mut guard = SYSTEM.get_or_init(|| Mutex::new(System::new().ok())).lock().unwrap_or_else(PoisonError::into_inner);
    // No provider: an unknown total keeps the configured ceiling (see [`budgets`]).
    let Some(s) = guard.as_mut() else { return Snapshot::default() };
    s.refresh_memory();
    let pid = Pid::from_u32(std::process::id());
    s.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, ProcessRefreshKind::nothing().with_memory());
    // The shared detector bounds host memory by the process's cgroup and, on macOS, counts only
    // reclaimable pages. A constraint it could not read keeps the host figure here: the tile
    // budget only decides what to spill, and a fail-closed zero would stall every operation.
    let available = match system::current() {
        system::SystemMemory { available_ram: Some(ram), constraint_unreadable: false, .. } => ram,
        _ => s.available_memory(),
    };
    Snapshot {
        total_bytes: s.total_memory(),
        available_bytes: available,
        process_resident_bytes: s.process(pid).map_or(0, |p| p.memory()),
        process_virtual_bytes: s.process(pid).map_or(0, |p| p.virtual_memory()),
    }
}

#[cfg(target_arch = "wasm32")]
pub fn measure() -> Snapshot {
    Snapshot::default()
}

pub fn refresh() -> Budgets {
    let sample = measure();
    let heap = heap_bytes();
    *SAMPLED.lock().unwrap_or_else(PoisonError::into_inner) = (sample, heap);
    apply(sample)
}

/// [`refresh`] without measuring the system again: the last measurement, moved by how much the
/// process's heap grew or shrank since (exact and cheap with the application's counting
/// allocator, see [`set_heap_counter`]; without one this only recomputes). The residency worker
/// calls it every tick, so the tile budget follows a large buffer within about 100 ms instead of
/// the next measurement two seconds later, and tiles move to the scratch disk before the system
/// runs out.
pub fn refresh_fast() -> Budgets {
    let (sample, heap0) = *SAMPLED.lock().unwrap_or_else(PoisonError::into_inner);
    apply(live(sample, heap0, heap_bytes()))
}

/// `sample` moved by the heap's change since it was taken (`heap0` then, `heap` now).
fn live(mut sample: Snapshot, heap0: Option<u64>, heap: Option<u64>) -> Snapshot {
    if let (Some(then), Some(now)) = (heap0, heap) {
        if now >= then {
            sample.process_resident_bytes = sample.process_resident_bytes.saturating_add(now - then);
            sample.available_bytes = sample.available_bytes.saturating_sub(now - then);
        } else {
            sample.process_resident_bytes = sample.process_resident_bytes.saturating_sub(then - now);
            sample.available_bytes = sample.available_bytes.saturating_add(then - now);
        }
    }
    sample
}

fn apply(sample: Snapshot) -> Budgets {
    let policy = *POLICY.lock().unwrap_or_else(PoisonError::into_inner);
    let b = budgets(policy, sample, crate::spill::stats().resident_bytes as u64);
    *CURRENT.lock().unwrap_or_else(PoisonError::into_inner) = (sample, b);
    // The test override remains fixed even when process pressure changes.
    let tiles = std::env::var("PHOTOCRAFT_TILE_BUDGET_MB").ok().and_then(|s| s.parse::<u64>().ok()).map_or(b.tile_bytes, |mb| mb.saturating_mul(MIB));
    crate::spill::set_budget(usize::try_from(tiles).unwrap_or(usize::MAX));
    b
}

/// The last system measurement and the heap bytes when it was taken.
static SAMPLED: Mutex<(Snapshot, Option<u64>)> =
    Mutex::new((Snapshot { total_bytes: 0, available_bytes: 0, process_resident_bytes: 0, process_virtual_bytes: 0 }, None));

/// The application's live heap count (its counting global allocator), when it installed one.
static HEAP: std::sync::OnceLock<fn() -> u64> = std::sync::OnceLock::new();

/// Install the application's live heap count: every byte the process allocates from Rust code,
/// whoever allocates it. The memory budget then follows buffers outside the tile store as they
/// are allocated (see [`refresh_fast`] and [`make_room`]). Only the first call counts.
pub fn set_heap_counter(counter: fn() -> u64) {
    let _ = HEAP.set(counter);
}

/// The process's live heap bytes, when the application counts them.
pub fn heap_bytes() -> Option<u64> {
    HEAP.get().map(|count| count())
}

/// Make room for a buffer of `bytes` about to be allocated outside the tile store: tiles in
/// memory move to the scratch disk first, as many as it takes. `Err` ("not enough memory") only
/// when that can't make enough room: the buffer is larger than the memory limit leaves for it,
/// or the tiles in the way can't move (the scratch disk is off, paused after a failure or full).
pub fn make_room(bytes: u64) -> Result<(), String> {
    let refused = |why: &str| format!("not enough memory for {} MB ({why})", bytes.div_ceil(MIB));
    // Each pass moves a bounded number of tiles; a few passes cover several gigabytes.
    for _ in 0..16 {
        let b = refresh_fast();
        let (sample, _, reserved) = stats();
        let tiles = crate::spill::stats().resident_bytes as u64;
        let used = sample.process_resident_bytes.saturating_add(reserved);
        let free = b.effective_process_bytes.saturating_sub(used);
        if bytes <= free || sample.total_bytes == 0 {
            return Ok(());
        }
        // Every tile in memory moved out still wouldn't leave enough.
        if bytes > free.saturating_add(tiles) {
            return Err(refused("the memory limit, after moving every image tile to the scratch disk"));
        }
        if !crate::spill::enabled() {
            return Err(refused("no scratch disk to move image tiles to"));
        }
        if crate::spill::paused() {
            return Err(refused("the scratch disk is full or failing"));
        }
        let target = tiles.saturating_sub(bytes - free);
        crate::spill::set_budget(usize::try_from(target).unwrap_or(usize::MAX).max(1));
        if crate::spill::maintain_budget() == 0 && crate::spill::stats().resident_bytes as u64 >= tiles {
            // Nothing could move (every tile in use, say): don't spin.
            return Err(refused("no image tile could move to the scratch disk"));
        }
    }
    Err(refused("the scratch disk couldn't take the image tiles fast enough"))
}

pub fn configure(policy: Policy) -> Budgets {
    *POLICY.lock().unwrap_or_else(PoisonError::into_inner) = policy;
    CONFIGURATION.fetch_add(1, Ordering::Relaxed);
    refresh()
}

pub fn stats() -> (Snapshot, Budgets, u64) {
    let (s, b) = *CURRENT.lock().unwrap_or_else(PoisonError::into_inner);
    (s, b, RESERVED.load(Ordering::Relaxed))
}

/// A bounded temporary working set. Callers retain the guard until the allocation is released.
pub struct Reservation(u64);

pub fn try_reserve(bytes: u64) -> Result<Reservation, String> {
    let (_, b, _) = stats();
    RESERVED
        .try_update(Ordering::AcqRel, Ordering::Relaxed, |n| n.checked_add(bytes).filter(|next| *next <= b.transient_bytes))
        .map_err(|_| "not enough working memory; reduce the operation or wait for other jobs".to_string())?;
    Ok(Reservation(bytes))
}

/// Admit a legacy operation that needs a whole buffer, including concurrent working sets and
/// the current process RSS. Streaming workers keep the smaller [`try_reserve`] allowance.
/// This deliberately refuses work before an allocation rather than relying on OS paging.
pub fn try_reserve_operation(bytes: u64) -> Result<Reservation, String> {
    let b = refresh();
    let (sample, _, _) = stats();
    // Resident tiles are evictable when spilling is on, so only the untiled part of RSS is
    // charged. The transient allowance is always admissible: it is what the budget reserves
    // for working sets, so normal-sized operations are never refused by a high RSS sample.
    let evictable = if crate::spill::enabled() { crate::spill::stats().resident_bytes as u64 } else { 0 };
    let untiled = sample.process_resident_bytes.saturating_sub(evictable);
    let available = if sample.total_bytes == 0 { b.transient_bytes } else { b.effective_process_bytes.saturating_sub(untiled).max(b.transient_bytes) };
    let refused = || "not enough working memory for this operation; reduce its region or use a streaming format".to_string();
    if bytes > available {
        return Err(refused());
    }
    // It fits on its own: wait (bounded) for concurrent operations to return their working
    // sets instead of failing an operation that only overlapped another one.
    #[cfg(not(target_arch = "wasm32"))]
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        if RESERVED.try_update(Ordering::AcqRel, Ordering::Relaxed, |n| n.checked_add(bytes).filter(|next| *next <= available)).is_ok() {
            return Ok(Reservation(bytes));
        }
        #[cfg(target_arch = "wasm32")]
        return Err(refused());
        #[cfg(not(target_arch = "wasm32"))]
        {
            let now = std::time::Instant::now();
            if now >= deadline {
                return Err(refused());
            }
            let guard = RELEASED.0.lock().unwrap_or_else(PoisonError::into_inner);
            // Re-checked after the wake-up; a missed notification costs at most this slice.
            let slice = deadline.saturating_duration_since(now).min(std::time::Duration::from_millis(50));
            drop(RELEASED.1.wait_timeout(guard, slice).unwrap_or_else(PoisonError::into_inner));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
static RELEASED: (Mutex<()>, std::sync::Condvar) = (Mutex::new(()), std::sync::Condvar::new());

impl Drop for Reservation {
    fn drop(&mut self) {
        RESERVED.fetch_sub(self.0, Ordering::AcqRel);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _guard = RELEASED.0.lock().unwrap_or_else(PoisonError::into_inner);
            RELEASED.1.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// `RESERVED` is process-wide, and `cargo test` runs this module's tests on threads of one
    /// process (nextest gives each test its own), so the tests that reserve take turns: one's
    /// reservation must not show up in another's before/after count.
    static RESERVING: Mutex<()> = Mutex::new(());
    fn reserving() -> std::sync::MutexGuard<'static, ()> {
        RESERVING.lock().unwrap_or_else(PoisonError::into_inner)
    }
    #[test]
    fn ceiling_includes_non_tile_memory_and_other_apps() {
        let s = Snapshot { total_bytes: 64 << 30, available_bytes: 10 << 30, process_resident_bytes: 20 << 30, process_virtual_bytes: 100 << 30 };
        let b = budgets(Policy { percent: Some(90), absolute_bytes: 8 << 30 }, s, 15 << 30);
        assert!(b.process_ceiling_bytes > 57 << 30);
        assert!(b.effective_process_bytes < 30 << 30);
        assert!(b.tile_bytes + (5 << 30) + b.transient_bytes <= b.effective_process_bytes);
    }
    #[test]
    fn a_live_sample_follows_the_heap() {
        let s = Snapshot { total_bytes: 32 << 30, available_bytes: 10 << 30, process_resident_bytes: 4 << 30, process_virtual_bytes: 0 };
        let grown = live(s, Some(1 << 30), Some(3 << 30));
        assert_eq!(grown.process_resident_bytes, 6 << 30);
        assert_eq!(grown.available_bytes, 8 << 30);
        let shrunk = live(s, Some(3 << 30), Some(1 << 30));
        assert_eq!(shrunk.process_resident_bytes, 2 << 30);
        assert_eq!(shrunk.available_bytes, 12 << 30);
        assert_eq!(live(s, None, Some(9 << 30)).process_resident_bytes, s.process_resident_bytes, "no count when it was taken");
    }
    #[test]
    fn room_beyond_every_tile_is_refused() {
        let _turn = reserving();
        refresh();
        assert!(make_room(u64::MAX / 2).is_err(), "more than the memory limit allows");
        assert!(make_room(0).is_ok());
    }
    #[test]
    fn old_absolute_setting_and_missing_measurements_work() {
        let b = budgets(Policy { percent: None, absolute_bytes: 512 * MIB }, Snapshot::default(), 0);
        assert_eq!(b.process_ceiling_bytes, 512 * MIB);
        assert_eq!(b.tile_bytes + b.transient_bytes, b.effective_process_bytes);
    }
    #[test]
    fn hostile_percent_and_pressure_do_not_overflow() {
        let b = budgets(Policy { percent: Some(255), absolute_bytes: u64::MAX }, Snapshot { total_bytes: u64::MAX, ..Snapshot::default() }, u64::MAX);
        assert!(b.process_ceiling_bytes < u64::MAX);
        assert_eq!(b.tile_bytes, MIB);
    }
    #[test]
    fn an_operation_that_fits_alone_waits_for_a_concurrent_one() {
        let _turn = reserving();
        let available = refresh().transient_bytes;
        let first = try_reserve_operation(available / 2 + 1).unwrap();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            drop(first);
        });
        let second = try_reserve_operation(available / 2 + 1);
        releaser.join().unwrap();
        assert!(second.is_ok(), "overlapping work is serialized, not refused");
        assert!(try_reserve_operation(u64::MAX).is_err(), "work that can never fit is refused at once");
    }
    #[test]
    fn reservations_are_bounded_and_returned() {
        let _turn = reserving();
        let before = stats().2;
        let r = try_reserve(1024).unwrap();
        assert_eq!(stats().2, before + 1024);
        assert!(try_reserve(u64::MAX).is_err());
        drop(r);
        assert_eq!(stats().2, before);
    }
}
