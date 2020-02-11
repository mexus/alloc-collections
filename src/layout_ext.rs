//! Layout helper extension.
//!
//! This is mostly a copy of `std::alloc::Layout`.

use std::{alloc::Layout, fmt, mem};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayoutErrExt {
    private: (),
}

impl fmt::Display for LayoutErrExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid parameters to Layout::from_size_align")
    }
}

pub trait LayoutExt: Sized {
    fn array_ext<T>(n: usize) -> Result<Self, LayoutErrExt>;
    fn repeat_ext(&self, n: usize) -> Result<(Self, usize), LayoutErrExt>;
    fn padding_needed_for_ext(&self, align: usize) -> usize;
}

impl LayoutExt for Layout {
    #[inline]
    fn array_ext<T>(n: usize) -> Result<Self, LayoutErrExt> {
        Layout::new::<T>().repeat_ext(n).map(|(k, offs)| {
            debug_assert!(offs == mem::size_of::<T>());
            k
        })
    }

    #[inline]
    fn repeat_ext(&self, n: usize) -> Result<(Self, usize), LayoutErrExt> {
        // This cannot overflow. Quoting from the invariant of Layout:
        // > `size`, when rounded up to the nearest multiple of `align`,
        // > must not overflow (i.e., the rounded value must be less than
        // > `usize::MAX`)
        let padded_size = self.size() + self.padding_needed_for_ext(self.align());
        let alloc_size = padded_size
            .checked_mul(n)
            .ok_or(LayoutErrExt { private: () })?;

        unsafe {
            // self.align is already known to be valid and alloc_size has been
            // padded already.
            Ok((
                Layout::from_size_align_unchecked(alloc_size, self.align()),
                padded_size,
            ))
        }
    }

    #[inline]
    fn padding_needed_for_ext(&self, align: usize) -> usize {
        let len = self.size();

        // Rounded up value is:
        //   len_rounded_up = (len + align - 1) & !(align - 1);
        // and then we return the padding difference: `len_rounded_up - len`.
        //
        // We use modular arithmetic throughout:
        //
        // 1. align is guaranteed to be > 0, so align - 1 is always
        //    valid.
        //
        // 2. `len + align - 1` can overflow by at most `align - 1`,
        //    so the &-mask with `!(align - 1)` will ensure that in the
        //    case of overflow, `len_rounded_up` will itself be 0.
        //    Thus the returned padding, when added to `len`, yields 0,
        //    which trivially satisfies the alignment `align`.
        //
        // (Of course, attempts to allocate blocks of memory whose
        // size and padding overflow in the above manner should cause
        // the allocator to yield an error anyway.)

        let len_rounded_up = len.wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1);
        len_rounded_up.wrapping_sub(len)
    }
}
