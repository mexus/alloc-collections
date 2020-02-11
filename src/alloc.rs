//! This is mostly a copy of `std::alloc::Alloc`.

use crate::layout_ext::LayoutExt;
use snafu::{ensure, OptionExt, Snafu};
use std::{
    alloc::Layout,
    cmp, fmt,
    ptr::{self, NonNull},
};

/// Represents the combination of a starting address and
/// a total capacity of the returned block.
#[derive(Debug)]
pub struct Excess(pub NonNull<u8>, pub usize);

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub")]
pub enum Error {
    #[snafu(display("Allocation of {:?} failed", layout))]
    AllocationError { layout: Layout },

    #[snafu(display("Allocation of {} items of {:?} has failed", items, layout))]
    ArrayAllocationError { layout: Layout, items: usize },

    #[snafu(display("Reallocation of {:?} to size {} failed", layout, new_size))]
    ReallocationError { layout: Layout, new_size: usize },

    #[snafu(display(
        "Reallocation of {} items of {:?} to {} items has failed",
        items,
        layout,
        new_items
    ))]
    ArrayReallocationError {
        layout: Layout,
        items: usize,
        new_items: usize,
    },

    #[snafu(display("Can't allocate space for a single zero-sized element"))]
    ZeroSize,
}

/// The `AllocErr` error indicates an allocation failure
/// that may be due to resource exhaustion or to
/// something wrong when combining the given input arguments with this
/// allocator.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AllocErr;

impl fmt::Display for AllocErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("memory allocation failed")
    }
}

/// The `CannotReallocInPlace` error is used when [`grow_in_place`] or
/// [`shrink_in_place`] were unable to reuse the given memory block for
/// a requested layout.
///
/// [`grow_in_place`]: ./trait.Alloc.html#method.grow_in_place
/// [`shrink_in_place`]: ./trait.Alloc.html#method.shrink_in_place
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CannotReallocInPlace;

impl CannotReallocInPlace {
    pub fn description(&self) -> &str {
        "cannot reallocate allocator's memory in place"
    }
}

impl fmt::Display for CannotReallocInPlace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

pub unsafe trait Alloc {
    // (Note: some existing allocators have unspecified but well-defined
    // behavior in response to a zero size allocation request ;
    // e.g., in C, `malloc` of 0 will either return a null pointer or a
    // unique pointer, but will not have arbitrary undefined
    // behavior.
    // However in jemalloc for example,
    // `mallocx(0)` is documented as undefined behavior.)

    /// Returns a pointer meeting the size and alignment guarantees of
    /// `layout`.
    ///
    /// If this method returns an `Ok(addr)`, then the `addr` returned
    /// will be non-null address pointing to a block of storage
    /// suitable for holding an instance of `layout`.
    ///
    /// The returned block of storage may or may not have its contents
    /// initialized. (Extension subtraits might restrict this
    /// behavior, e.g., to ensure initialization to particular sets of
    /// bit patterns.)
    ///
    /// # Safety
    ///
    /// This function is unsafe because undefined behavior can result
    /// if the caller does not ensure that `layout` has non-zero size.
    ///
    /// (Extension subtraits might provide more specific bounds on
    /// behavior, e.g., guarantee a sentinel address or a null pointer
    /// in response to a zero-size allocation request.)
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either memory is exhausted or
    /// `layout` does not meet allocator's size or alignment
    /// constraints.
    ///
    /// Implementations are encouraged to return `Err` on memory
    /// exhaustion rather than panicking or aborting, but this is not
    /// a strict requirement. (Specifically: it is *legal* to
    /// implement this trait atop an underlying native allocation
    /// library that aborts on memory exhaustion.)
    ///
    /// Clients wishing to abort computation in response to an
    /// allocation error are encouraged to call the [`handle_alloc_error`] function,
    /// rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, Error>;

    /// Deallocate the memory referenced by `ptr`.
    ///
    /// # Safety
    ///
    /// This function is unsafe because undefined behavior can result
    /// if the caller does not ensure all of the following:
    ///
    /// * `ptr` must denote a block of memory currently allocated via
    ///   this allocator,
    ///
    /// * `layout` must *fit* that block of memory,
    ///
    /// * In addition to fitting the block of memory `layout`, the
    ///   alignment of the `layout` must match the alignment used
    ///   to allocate that block of memory.
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout);

    // == ALLOCATOR-SPECIFIC QUANTITIES AND LIMITS ==
    // usable_size

    /// Returns bounds on the guaranteed usable size of a successful
    /// allocation created with the specified `layout`.
    ///
    /// In particular, if one has a memory block allocated via a given
    /// allocator `a` and layout `k` where `a.usable_size(k)` returns
    /// `(l, u)`, then one can pass that block to `a.dealloc()` with a
    /// layout in the size range [l, u].
    ///
    /// (All implementors of `usable_size` must ensure that
    /// `l <= k.size() <= u`)
    ///
    /// Both the lower- and upper-bounds (`l` and `u` respectively)
    /// are provided, because an allocator based on size classes could
    /// misbehave if one attempts to deallocate a block without
    /// providing a correct value for its size (i.e., one within the
    /// range `[l, u]`).
    ///
    /// Clients who wish to make use of excess capacity are encouraged
    /// to use the `alloc_excess` and `realloc_excess` instead, as
    /// this method is constrained to report conservative values that
    /// serve as valid bounds for *all possible* allocation method
    /// calls.
    ///
    /// However, for clients that do not wish to track the capacity
    /// returned by `alloc_excess` locally, this method is likely to
    /// produce useful results.
    #[inline]
    fn usable_size(&self, layout: &Layout) -> (usize, usize) {
        (layout.size(), layout.size())
    }

    // == METHODS FOR MEMORY REUSE ==
    // realloc. alloc_excess, realloc_excess

    /// Returns a pointer suitable for holding data described by
    /// a new layout with `layout`’s alignment and a size given
    /// by `new_size`. To
    /// accomplish this, this may extend or shrink the allocation
    /// referenced by `ptr` to fit the new layout.
    ///
    /// If this returns `Ok`, then ownership of the memory block
    /// referenced by `ptr` has been transferred to this
    /// allocator. The memory may or may not have been freed, and
    /// should be considered unusable (unless of course it was
    /// transferred back to the caller again via the return value of
    /// this method).
    ///
    /// If this method returns `Err`, then ownership of the memory
    /// block has not been transferred to this allocator, and the
    /// contents of the memory block are unaltered.
    ///
    /// # Safety
    ///
    /// This function is unsafe because undefined behavior can result
    /// if the caller does not ensure all of the following:
    ///
    /// * `ptr` must be currently allocated via this allocator,
    ///
    /// * `layout` must *fit* the `ptr` (see above). (The `new_size`
    ///   argument need not fit it.)
    ///
    /// * `new_size` must be greater than zero.
    ///
    /// * `new_size`, when rounded up to the nearest multiple of `layout.align()`,
    ///   must not overflow (i.e., the rounded value must be less than `usize::MAX`).
    ///
    /// (Extension subtraits might provide more specific bounds on
    /// behavior, e.g., guarantee a sentinel address or a null pointer
    /// in response to a zero-size allocation request.)
    ///
    /// # Errors
    ///
    /// Returns `Err` only if the new layout
    /// does not meet the allocator's size
    /// and alignment constraints of the allocator, or if reallocation
    /// otherwise fails.
    ///
    /// Implementations are encouraged to return `Err` on memory
    /// exhaustion rather than panicking or aborting, but this is not
    /// a strict requirement. (Specifically: it is *legal* to
    /// implement this trait atop an underlying native allocation
    /// library that aborts on memory exhaustion.)
    ///
    /// Clients wishing to abort computation in response to a
    /// reallocation error are encouraged to call the [`handle_alloc_error`] function,
    /// rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, Error> {
        let old_size = layout.size();

        if new_size >= old_size {
            if let Ok(()) = self.grow_in_place(ptr, layout, new_size) {
                return Ok(ptr);
            }
        } else if new_size < old_size {
            if let Ok(()) = self.shrink_in_place(ptr, layout, new_size) {
                return Ok(ptr);
            }
        }

        // otherwise, fall back on alloc + copy + dealloc.
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let result = self.alloc(new_layout);
        if let Ok(new_ptr) = result {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), cmp::min(old_size, new_size));
            self.dealloc(ptr, layout);
        }
        result
    }

    /// Behaves like `alloc`, but also ensures that the contents
    /// are set to zero before being returned.
    ///
    /// # Safety
    ///
    /// This function is unsafe for the same reasons that `alloc` is.
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either memory is exhausted or
    /// `layout` does not meet allocator's size or alignment
    /// constraints, just as in `alloc`.
    ///
    /// Clients wishing to abort computation in response to an
    /// allocation error are encouraged to call the [`handle_alloc_error`] function,
    /// rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<NonNull<u8>, Error> {
        let size = layout.size();
        let p = self.alloc(layout);
        if let Ok(p) = p {
            ptr::write_bytes(p.as_ptr(), 0, size);
        }
        p
    }

    /// Behaves like `alloc`, but also returns the whole size of
    /// the returned block. For some `layout` inputs, like arrays, this
    /// may include extra storage usable for additional data.
    ///
    /// # Safety
    ///
    /// This function is unsafe for the same reasons that `alloc` is.
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either memory is exhausted or
    /// `layout` does not meet allocator's size or alignment
    /// constraints, just as in `alloc`.
    ///
    /// Clients wishing to abort computation in response to an
    /// allocation error are encouraged to call the [`handle_alloc_error`] function,
    /// rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<Excess, Error> {
        let usable_size = self.usable_size(&layout);
        self.alloc(layout).map(|p| Excess(p, usable_size.1))
    }

    /// Behaves like `realloc`, but also returns the whole size of
    /// the returned block. For some `layout` inputs, like arrays, this
    /// may include extra storage usable for additional data.
    ///
    /// # Safety
    ///
    /// This function is unsafe for the same reasons that `realloc` is.
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either memory is exhausted or
    /// `layout` does not meet allocator's size or alignment
    /// constraints, just as in `realloc`.
    ///
    /// Clients wishing to abort computation in response to a
    /// reallocation error are encouraged to call the [`handle_alloc_error`] function,
    /// rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    unsafe fn realloc_excess(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<Excess, Error> {
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let usable_size = self.usable_size(&new_layout);
        self.realloc(ptr, layout, new_size)
            .map(|p| Excess(p, usable_size.1))
    }

    /// Attempts to extend the allocation referenced by `ptr` to fit `new_size`.
    ///
    /// If this returns `Ok`, then the allocator has asserted that the
    /// memory block referenced by `ptr` now fits `new_size`, and thus can
    /// be used to carry data of a layout of that size and same alignment as
    /// `layout`. (The allocator is allowed to
    /// expend effort to accomplish this, such as extending the memory block to
    /// include successor blocks, or virtual memory tricks.)
    ///
    /// Regardless of what this method returns, ownership of the
    /// memory block referenced by `ptr` has not been transferred, and
    /// the contents of the memory block are unaltered.
    ///
    /// # Safety
    ///
    /// This function is unsafe because undefined behavior can result
    /// if the caller does not ensure all of the following:
    ///
    /// * `ptr` must be currently allocated via this allocator,
    ///
    /// * `layout` must *fit* the `ptr` (see above); note the
    ///   `new_size` argument need not fit it,
    ///
    /// * `new_size` must not be less than `layout.size()`,
    ///
    /// # Errors
    ///
    /// Returns `Err(CannotReallocInPlace)` when the allocator is
    /// unable to assert that the memory block referenced by `ptr`
    /// could fit `layout`.
    ///
    /// Note that one cannot pass `CannotReallocInPlace` to the `handle_alloc_error`
    /// function; clients are expected either to be able to recover from
    /// `grow_in_place` failures without aborting, or to fall back on
    /// another reallocation method before resorting to an abort.
    unsafe fn grow_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<(), CannotReallocInPlace> {
        let _ = ptr; // this default implementation doesn't care about the actual address.
        debug_assert!(new_size >= layout.size());
        let (_l, u) = self.usable_size(&layout);
        // _l <= layout.size()                       [guaranteed by usable_size()]
        //       layout.size() <= new_layout.size()  [required by this method]
        if new_size <= u {
            Ok(())
        } else {
            Err(CannotReallocInPlace)
        }
    }

    /// Attempts to shrink the allocation referenced by `ptr` to fit `new_size`.
    ///
    /// If this returns `Ok`, then the allocator has asserted that the
    /// memory block referenced by `ptr` now fits `new_size`, and
    /// thus can only be used to carry data of that smaller
    /// layout. (The allocator is allowed to take advantage of this,
    /// carving off portions of the block for reuse elsewhere.) The
    /// truncated contents of the block within the smaller layout are
    /// unaltered, and ownership of block has not been transferred.
    ///
    /// If this returns `Err`, then the memory block is considered to
    /// still represent the original (larger) `layout`. None of the
    /// block has been carved off for reuse elsewhere, ownership of
    /// the memory block has not been transferred, and the contents of
    /// the memory block are unaltered.
    ///
    /// # Safety
    ///
    /// This function is unsafe because undefined behavior can result
    /// if the caller does not ensure all of the following:
    ///
    /// * `ptr` must be currently allocated via this allocator,
    ///
    /// * `layout` must *fit* the `ptr` (see above); note the
    ///   `new_size` argument need not fit it,
    ///
    /// * `new_size` must not be greater than `layout.size()`
    ///   (and must be greater than zero),
    ///
    /// # Errors
    ///
    /// Returns `Err(CannotReallocInPlace)` when the allocator is
    /// unable to assert that the memory block referenced by `ptr`
    /// could fit `layout`.
    ///
    /// Note that one cannot pass `CannotReallocInPlace` to the `handle_alloc_error`
    /// function; clients are expected either to be able to recover from
    /// `shrink_in_place` failures without aborting, or to fall back
    /// on another reallocation method before resorting to an abort.
    unsafe fn shrink_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<(), CannotReallocInPlace> {
        let _ = ptr; // this default implementation doesn't care about the actual address.
        debug_assert!(new_size <= layout.size());
        let (l, _u) = self.usable_size(&layout);
        //                      layout.size() <= _u  [guaranteed by usable_size()]
        // new_layout.size() <= layout.size()        [required by this method]
        if l <= new_size {
            Ok(())
        } else {
            Err(CannotReallocInPlace)
        }
    }

    // == COMMON USAGE PATTERNS ==
    // alloc_one, dealloc_one, alloc_array, realloc_array. dealloc_array

    /// Allocates a block suitable for holding an instance of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// The returned block is suitable for passing to the
    /// `realloc`/`dealloc` methods of this allocator.
    ///
    /// Note to implementors: If this returns `Ok(ptr)`, then `ptr`
    /// must be considered "currently allocated" and must be
    /// acceptable input to methods such as `realloc` or `dealloc`,
    /// *even if* `T` is a zero-sized type. In other words, if your
    /// `Alloc` implementation overrides this method in a manner
    /// that can return a zero-sized `ptr`, then all reallocation and
    /// deallocation methods need to be similarly overridden to accept
    /// such values as input.
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either memory is exhausted or
    /// `T` does not meet allocator's size or alignment constraints.
    ///
    /// For zero-sized `T`, may return either of `Ok` or `Err`, but
    /// will *not* yield undefined behavior.
    ///
    /// Clients wishing to abort computation in response to an
    /// allocation error are encouraged to call the [`handle_alloc_error`] function,
    /// rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    fn alloc_one<T>(&mut self) -> Result<NonNull<T>, Error>
    where
        Self: Sized,
    {
        let k = Layout::new::<T>();
        ensure!(k.size() > 0, ZeroSize);
        unsafe { self.alloc(k).map(|p| p.cast()) }
    }

    /// Deallocates a block suitable for holding an instance of `T`.
    ///
    /// The given block must have been produced by this allocator,
    /// and must be suitable for storing a `T` (in terms of alignment
    /// as well as minimum and maximum size); otherwise yields
    /// undefined behavior.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// # Safety
    ///
    /// This function is unsafe because undefined behavior can result
    /// if the caller does not ensure both:
    ///
    /// * `ptr` must denote a block of memory currently allocated via this allocator
    ///
    /// * the layout of `T` must *fit* that block of memory.
    unsafe fn dealloc_one<T>(&mut self, ptr: NonNull<T>)
    where
        Self: Sized,
    {
        let k = Layout::new::<T>();
        if k.size() > 0 {
            self.dealloc(ptr.cast(), k);
        }
    }

    /// Allocates a block suitable for holding `n` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// The returned block is suitable for passing to the
    /// `realloc`/`dealloc` methods of this allocator.
    ///
    /// Note to implementors: If this returns `Ok(ptr)`, then `ptr`
    /// must be considered "currently allocated" and must be
    /// acceptable input to methods such as `realloc` or `dealloc`,
    /// *even if* `T` is a zero-sized type. In other words, if your
    /// `Alloc` implementation overrides this method in a manner
    /// that can return a zero-sized `ptr`, then all reallocation and
    /// deallocation methods need to be similarly overridden to accept
    /// such values as input.
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either memory is exhausted or
    /// `[T; n]` does not meet allocator's size or alignment
    /// constraints.
    ///
    /// For zero-sized `T` or `n == 0`, may return either of `Ok` or
    /// `Err`, but will *not* yield undefined behavior.
    ///
    /// Always returns `Err` on arithmetic overflow.
    ///
    /// Clients wishing to abort computation in response to an
    /// allocation error are encouraged to call the [`handle_alloc_error`] function,
    /// rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    fn alloc_array<T>(&mut self, n: usize) -> Result<NonNull<T>, Error>
    where
        Self: Sized,
    {
        match Layout::array_ext::<T>(n) {
            Ok(layout) if layout.size() > 0 => unsafe { self.alloc(layout).map(|p| p.cast()) },
            _ => Err(Error::ArrayAllocationError {
                layout: Layout::new::<T>(),
                items: n,
            }),
        }
    }

    /// Reallocates a block previously suitable for holding `n_old`
    /// instances of `T`, returning a block suitable for holding
    /// `n_new` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// The returned block is suitable for passing to the
    /// `realloc`/`dealloc` methods of this allocator.
    ///
    /// # Safety
    ///
    /// This function is unsafe because undefined behavior can result
    /// if the caller does not ensure all of the following:
    ///
    /// * `ptr` must be currently allocated via this allocator,
    ///
    /// * the layout of `[T; n_old]` must *fit* that block of memory.
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either memory is exhausted or
    /// `[T; n_new]` does not meet allocator's size or alignment
    /// constraints.
    ///
    /// For zero-sized `T` or `n_new == 0`, may return either of `Ok` or
    /// `Err`, but will *not* yield undefined behavior.
    ///
    /// Always returns `Err` on arithmetic overflow.
    ///
    /// Clients wishing to abort computation in response to a
    /// reallocation error are encouraged to call the [`handle_alloc_error`] function,
    /// rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    unsafe fn realloc_array<T>(
        &mut self,
        ptr: NonNull<T>,
        n_old: usize,
        n_new: usize,
    ) -> Result<NonNull<T>, Error>
    where
        Self: Sized,
    {
        match (Layout::array_ext::<T>(n_old), Layout::array_ext::<T>(n_new)) {
            (Ok(k_old), Ok(k_new)) if k_old.size() > 0 && k_new.size() > 0 => {
                debug_assert!(k_old.align() == k_new.align());
                self.realloc(ptr.cast(), k_old, k_new.size())
                    .map(NonNull::cast)
            }
            _ => Err(Error::ArrayReallocationError {
                layout: Layout::new::<T>(),
                items: n_old,
                new_items: n_new,
            }),
        }
    }

    /// Deallocates a block suitable for holding `n` instances of `T`.
    ///
    /// Captures a common usage pattern for allocators.
    ///
    /// # Safety
    ///
    /// This function is unsafe because undefined behavior can result
    /// if the caller does not ensure both:
    ///
    /// * `ptr` must denote a block of memory currently allocated via this allocator
    ///
    /// * the layout of `[T; n]` must *fit* that block of memory.
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either `[T; n]` or the given
    /// memory block does not meet allocator's size or alignment
    /// constraints.
    ///
    /// Always returns `Err` on arithmetic overflow.
    unsafe fn dealloc_array<T>(&mut self, ptr: NonNull<T>, n: usize) -> Result<(), AllocErr>
    where
        Self: Sized,
    {
        match Layout::array_ext::<T>(n) {
            Ok(k) if k.size() > 0 => Ok(self.dealloc(ptr.cast(), k)),
            _ => Err(AllocErr),
        }
    }
}

unsafe impl<A: Alloc> Alloc for &mut A {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, Error> {
        A::alloc(self, layout)
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        A::dealloc(self, ptr, layout)
    }

    #[inline]
    fn usable_size(&self, layout: &Layout) -> (usize, usize) {
        A::usable_size(self, layout)
    }

    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, Error> {
        A::realloc(self, ptr, layout, new_size)
    }

    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<NonNull<u8>, Error> {
        A::alloc_zeroed(self, layout)
    }

    unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<Excess, Error> {
        A::alloc_excess(self, layout)
    }

    unsafe fn realloc_excess(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<Excess, Error> {
        A::realloc_excess(self, ptr, layout, new_size)
    }

    unsafe fn grow_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<(), CannotReallocInPlace> {
        A::grow_in_place(self, ptr, layout, new_size)
    }

    unsafe fn shrink_in_place(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<(), CannotReallocInPlace> {
        A::shrink_in_place(self, ptr, layout, new_size)
    }

    fn alloc_one<T>(&mut self) -> Result<NonNull<T>, Error> {
        A::alloc_one(self)
    }

    unsafe fn dealloc_one<T>(&mut self, ptr: NonNull<T>) {
        A::dealloc_one(self, ptr)
    }

    fn alloc_array<T>(&mut self, n: usize) -> Result<NonNull<T>, Error> {
        A::alloc_array(self, n)
    }

    unsafe fn realloc_array<T>(
        &mut self,
        ptr: NonNull<T>,
        n_old: usize,
        n_new: usize,
    ) -> Result<NonNull<T>, Error> {
        A::realloc_array(self, ptr, n_old, n_new)
    }

    unsafe fn dealloc_array<T>(&mut self, ptr: NonNull<T>, n: usize) -> Result<(), AllocErr> {
        A::dealloc_array(self, ptr, n)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GlobalAlloc<A>(A);

use ::std::alloc::GlobalAlloc as StdGlobalAlloc;

unsafe impl<A: StdGlobalAlloc> Alloc for GlobalAlloc<A> {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, Error> {
        let ptr = StdGlobalAlloc::alloc(&mut self.0, layout);
        NonNull::new(ptr).context(AllocationError { layout })
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        StdGlobalAlloc::dealloc(&mut self.0, ptr.as_ptr(), layout)
    }

    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, Error> {
        let ptr = StdGlobalAlloc::realloc(&mut self.0, ptr.as_ptr(), layout, new_size);
        NonNull::new(ptr).context(ReallocationError { layout, new_size })
    }
}

#[derive(Debug, Default, Clone)]
pub struct Global;

unsafe impl Alloc for Global {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, Error> {
        let ptr = std::alloc::alloc(layout);
        NonNull::new(ptr).context(AllocationError { layout })
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        std::alloc::dealloc(ptr.as_ptr(), layout)
    }

    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, Error> {
        let ptr = std::alloc::realloc(ptr.as_ptr(), layout, new_size);
        NonNull::new(ptr).context(ReallocationError { layout, new_size })
    }
}

unsafe impl Alloc for &Global {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, Error> {
        Global.alloc(layout)
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        Global.dealloc(ptr, layout)
    }

    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, Error> {
        Global.realloc(ptr, layout, new_size)
    }
}

pub type System = GlobalAlloc<::std::alloc::System>;

unsafe impl Alloc for &System {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, Error> {
        GlobalAlloc(::std::alloc::System).alloc(layout)
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        GlobalAlloc(::std::alloc::System).dealloc(ptr, layout)
    }

    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, Error> {
        GlobalAlloc(::std::alloc::System).realloc(ptr, layout, new_size)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NoOp;

unsafe impl Alloc for NoOp {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, Error> {
        Err(Error::AllocationError { layout })
    }

    unsafe fn dealloc(&mut self, _ptr: NonNull<u8>, _layout: Layout) {
        /* No op */
    }
}

unsafe impl Alloc for &NoOp {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, Error> {
        NoOp.alloc(layout)
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        NoOp.dealloc(ptr, layout)
    }
}
