//! Test-only heap-allocation counter.
//!
//! CB2Vec promises that a search loop built on [`IncrementalSession`] performs
//! no heap allocation after construction. That promise is only credible if it
//! is measured, so the unit-test binary installs a counting global allocator
//! and [`AllocationGuard`] asserts the delta around a block of work.
//!
//! The counter is thread-local and const-initialized, so reading it inside the
//! allocator neither allocates nor registers a destructor.
//!
//! [`IncrementalSession`]: crate::IncrementalSession

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Forwards to the system allocator and counts allocating calls.
pub struct CountingAllocator;

// SAFETY: Every method forwards its arguments unchanged to `System`, which is
// a correct `GlobalAlloc`. The counter is a thread-local `Cell<u64>` with no
// destructor, so recording cannot allocate or re-enter the allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        // SAFETY: The layout is forwarded unchanged from the caller.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        // SAFETY: The layout is forwarded unchanged from the caller.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        // SAFETY: The pointer, layout, and size are forwarded unchanged and
        // originate from a matching allocation made by this allocator.
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The pointer and layout are forwarded unchanged and originate
        // from a matching allocation made by this allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[inline]
fn record() {
    // `try_with` keeps thread teardown from panicking inside the allocator.
    let _ = ALLOCATIONS.try_with(|count| count.set(count.get().wrapping_add(1)));
}

/// Allocations performed by the current thread since it started.
pub fn thread_allocations() -> u64 {
    ALLOCATIONS.with(|count| count.get())
}

/// Snapshot used to assert that a block of work allocated nothing.
pub struct AllocationGuard {
    start: u64,
}

impl AllocationGuard {
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: thread_allocations(),
        }
    }

    /// Allocations recorded since this guard was created.
    pub fn allocations(&self) -> u64 {
        thread_allocations().wrapping_sub(self.start)
    }

    #[track_caller]
    pub fn assert_no_allocations(&self, what: &str) {
        let observed = self.allocations();
        assert_eq!(observed, 0, "{what} performed {observed} heap allocations");
    }
}

impl Default for AllocationGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_counter_observes_allocations_and_ignores_stack_work() {
        let guard = AllocationGuard::new();
        let mut total = 0u64;
        for value in 0..64u64 {
            total = total.wrapping_add(value);
        }
        assert_eq!(total, 2016);
        guard.assert_no_allocations("integer arithmetic");

        let guard = AllocationGuard::new();
        let values: Vec<u8> = Vec::with_capacity(4096);
        assert_eq!(values.capacity(), 4096);
        assert!(guard.allocations() >= 1, "Vec::with_capacity must allocate");
    }
}
