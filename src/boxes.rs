//! Various Box-like types.

use crate::{
    alloc::{Alloc, Error as AllocErr, Global},
    vec::Vec,
};
use core::{
    alloc::Layout,
    convert::TryFrom,
    marker::PhantomData,
    mem::{align_of, align_of_val, size_of, size_of_val, ManuallyDrop},
    ops::{Deref, DerefMut},
    ptr::{null_mut, NonNull},
    slice,
};

/// Custom-allocated `T`.
pub struct CustomBox<T: ?Sized, A: Alloc = Global> {
    ptr: NonNull<u8>,
    allocator: ManuallyDrop<A>,
    layout: Layout,
    _pd: PhantomData<T>,
}

impl<T: ?Sized> CustomBox<T> {
    /// Puts given `item` into a new memory chunk, allocated by the global allocator.
    pub fn new_unsized(item: &T) -> Result<Self, AllocErr>
    where
        for<'a> &'a T: Copy,
    {
        Self::new_unsized_in(item, <_>::default())
    }
}

impl<T> CustomBox<T> {
    /// Puts given `item` into a new memory chunk, allocated by the global allocator.
    pub fn new(item: T) -> Result<Self, AllocErr> {
        Self::new_in(item, <_>::default())
    }
}

impl<T, A: Alloc> CustomBox<[T], A> {
    /// Represents boxed item as a slice.
    pub fn as_slice(&self) -> &[T] {
        let len = self.layout.size() / size_of::<T>();
        unsafe { slice::from_raw_parts(self.ptr.as_ptr().cast(), len) }
    }

    /// Represents boxed item as a mutable slice.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        let len = self.layout.size() / size_of::<T>();
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), len) }
    }

    /// Tries to clone a slice.
    pub fn try_clone_slice(&self) -> Result<Self, AllocErr>
    where
        T: Clone,
        A: Clone,
    {
        let mut allocator = A::clone(&self.allocator);
        let new_ptr;

        if self.layout.size() == 0 {
            new_ptr = NonNull::<T>::dangling();
        } else {
            unsafe {
                new_ptr = allocator.alloc(self.layout)?.cast::<T>();
                for (id, item) in self.as_slice().iter().enumerate() {
                    new_ptr.as_ptr().add(id).write(item.clone());
                }
            }
        };

        Ok(CustomBox {
            allocator: ManuallyDrop::new(allocator),
            layout: self.layout,
            ptr: new_ptr.cast(),
            _pd: PhantomData,
        })
    }

    /// Tries to clone a slice into a given allocator.
    pub fn try_clone_slice_into<NewAlloc>(
        &self,
        mut new_alloc: NewAlloc,
    ) -> Result<CustomBox<T, NewAlloc>, AllocErr>
    where
        T: Clone,
        NewAlloc: Alloc,
    {
        let new_ptr;
        if self.layout.size() == 0 {
            new_ptr = NonNull::<T>::dangling();
        } else {
            unsafe {
                new_ptr = new_alloc.alloc(self.layout)?.cast::<T>();
                for (id, item) in self.as_slice().iter().enumerate() {
                    new_ptr.as_ptr().add(id).write(item.clone());
                }
            }
        }
        Ok(CustomBox {
            allocator: ManuallyDrop::new(new_alloc),
            layout: self.layout,
            ptr: new_ptr.cast(),
            _pd: PhantomData,
        })
    }

    /// Convets the owned slice into a vector. No allocations involved!
    pub fn into_vec(self) -> Vec<T, A> {
        Vec::from_boxed_slice(self)
    }
}

impl<T: ?Sized, A: Alloc> CustomBox<T, A> {
    /// Puts given `item` into a new memory chunk, allocated by a given allocator.
    pub fn new_unsized_in(item: &T, mut allocator: A) -> Result<Self, AllocErr>
    where
        for<'a> &'a T: Copy,
    {
        let alignment = align_of_val(item);
        let size = size_of_val(item);
        let layout = unsafe { Layout::from_size_align_unchecked(size, alignment) };

        let heap_ptr = if size == 0 {
            unsafe { NonNull::new_unchecked(null_mut::<u8>().add(alignment)) }
        } else {
            unsafe { allocator.alloc(layout)? }
        };
        unsafe {
            heap_ptr
                .as_ptr()
                .copy_from_nonoverlapping(item as *const T as *const u8, size)
        }
        Ok(CustomBox {
            ptr: heap_ptr,
            allocator: ManuallyDrop::new(allocator),
            layout,
            _pd: PhantomData,
        })
    }

    /// Constructs a boxed object from a pointer, layout and an allocator that was used to obtain
    /// the aforementioned parameters.
    ///
    /// # Safety
    ///
    /// The method is safe if and only if the pointer was obtained by the given allocator with the
    /// given layout and there are no other owners.
    pub unsafe fn from_raw_parts(ptr: NonNull<u8>, layout: Layout, allocator: A) -> Self {
        debug_assert!(ptr.as_ptr() as usize % layout.align() == 0);
        CustomBox {
            ptr,
            allocator: ManuallyDrop::new(allocator),
            layout,
            _pd: PhantomData,
        }
    }

    /// Breaks the box into raw parts. Could be reconstructed back using
    /// `CustomBox::from_raw_parts`.
    pub fn into_raw_parts(self) -> (NonNull<u8>, Layout, A) {
        let mut boxed = ManuallyDrop::new(self);
        let allocator = unsafe { ManuallyDrop::take(&mut boxed.allocator) };
        (boxed.ptr, boxed.layout, allocator)
    }

    /// Tries to copy self.
    pub fn try_copy(&self) -> Result<Self, AllocErr>
    where
        T: Copy,
        A: Clone,
    {
        self.try_copy_with(A::clone(&self.allocator))
    }

    /// Tries to copy self into another allocator.
    pub fn try_copy_with<NewAlloc>(
        &self,
        mut new_allocator: NewAlloc,
    ) -> Result<CustomBox<T, NewAlloc>, AllocErr>
    where
        T: Copy,
        NewAlloc: Alloc,
    {
        let layout = self.layout;
        let new_ptr;

        unsafe {
            if layout.size() == 0 {
                new_ptr = NonNull::new_unchecked(null_mut::<u8>().add(layout.align()));
            } else {
                new_ptr = new_allocator.alloc(layout)?;
                new_ptr
                    .as_ptr()
                    .copy_from_nonoverlapping(self.ptr.as_ptr(), layout.size());
            }
        }

        Ok(CustomBox {
            ptr: new_ptr,
            layout,
            allocator: ManuallyDrop::new(new_allocator),
            _pd: PhantomData,
        })
    }
}

impl<T, A: Alloc> CustomBox<T, A> {
    /// Puts given `item` into a new memory chunk, allocated by a given allocator.
    pub fn new_in(item: T, mut allocator: A) -> Result<Self, AllocErr> {
        let alignment = align_of::<T>();
        let size = size_of::<T>();
        let layout = unsafe { Layout::from_size_align_unchecked(size, alignment) };

        let heap_ptr = if size == 0 {
            NonNull::<T>::dangling().cast()
        } else {
            unsafe { allocator.alloc(layout)? }
        };
        unsafe { heap_ptr.as_ptr().cast::<T>().write(item) }
        Ok(CustomBox {
            ptr: heap_ptr,
            allocator: ManuallyDrop::new(allocator),
            layout,
            _pd: PhantomData,
        })
    }

    /// Tries to clone self.
    pub fn try_clone(&self) -> Result<Self, AllocErr>
    where
        T: Clone,
        A: Clone,
    {
        let allocator = A::clone(&self.allocator);
        self.try_clone_with(allocator)
    }

    /// Tries to clone self into another allocator.
    pub fn try_clone_with<NewAlloc>(
        &self,
        mut new_allocator: NewAlloc,
    ) -> Result<CustomBox<T, NewAlloc>, AllocErr>
    where
        T: Clone,
        NewAlloc: Alloc,
    {
        let layout = self.layout;

        let new_ptr;
        if layout.size() == 0 {
            new_ptr = NonNull::<T>::dangling();
        } else {
            new_ptr = new_allocator.alloc_one::<T>()?;
            unsafe {
                new_ptr
                    .as_ptr()
                    .write(self.ptr.cast::<T>().as_ref().clone());
            }
        }

        Ok(CustomBox {
            ptr: new_ptr.cast(),
            layout,
            allocator: ManuallyDrop::new(new_allocator),
            _pd: PhantomData,
        })
    }
}

impl<T: ?Sized, A: Alloc> Drop for CustomBox<T, A> {
    fn drop(&mut self) {
        unsafe {
            self.ptr.as_ptr().drop_in_place();
            let mut allocator = ManuallyDrop::take(&mut self.allocator);
            if self.layout.size() != 0 {
                let _ = allocator.dealloc(self.ptr, self.layout);
            }
        }
    }
}

impl<T, A: Alloc> Deref for CustomBox<T, A> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.ptr.as_ptr() as *const T) }
    }
}

impl<T, A: Alloc> Deref for CustomBox<[T], A> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, A: Alloc> DerefMut for CustomBox<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self.ptr.as_ptr() as *mut T) }
    }
}

impl<T, A: Alloc> DerefMut for CustomBox<[T], A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

impl<T> TryFrom<std::vec::Vec<T>> for CustomBox<[T]> {
    type Error = AllocErr;

    fn try_from(vector: std::vec::Vec<T>) -> Result<Self, AllocErr> {
        let slice = vector.as_slice();
        CustomBox::new_unsized(slice)
    }
}

impl<T: Clone, A: Alloc + Clone> Clone for CustomBox<T, A> {
    fn clone(&self) -> Self {
        self.try_clone().expect("Clone failed")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn check() {
        let vec = vec!["l", "o", "l"];
        let mut boxed = CustomBox::new_unsized(vec.as_slice()).unwrap();
        assert_eq!(boxed.as_slice(), &["l", "o", "l"]);

        boxed.as_slice_mut()[0] = "ahaha";
        assert_eq!(boxed.as_slice(), &["ahaha", "o", "l"]);
    }
}
