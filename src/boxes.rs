//! Various Box-like types.

use crate::{
    alloc::{Alloc, Error as AllocErr, Global},
    Vec,
};
use core::{
    mem::{size_of, ManuallyDrop},
    ops::{Deref, DerefMut},
    ptr::{self, copy_nonoverlapping, NonNull},
    slice,
};

/// Custom-allocated slice of `T`.
pub struct BoxedSlice<T, A: Alloc = Global> {
    ptr: NonNull<T>,
    allocator: ManuallyDrop<A>,
    count: usize,
}

impl<T> BoxedSlice<T> {
    /// Creates a boxed slice from a contigous array.
    pub fn new<Array>(array: Array) -> Result<Self, AllocErr>
    where
        Array: AsRef<[T]>,
    {
        Self::new_in(array, <_>::default())
    }
}

impl<T, A: Alloc> BoxedSlice<T, A> {
    /// Creates a boxed slice from a contigous array.
    pub fn new_in<Array>(array: Array, mut allocator: A) -> Result<Self, AllocErr>
    where
        Array: AsRef<[T]>,
    {
        let count = array.as_ref().len();
        let first = array.as_ref().as_ptr();

        let ptr;
        if size_of::<T>() == 0 || count == 0 {
            ptr = NonNull::dangling()
        } else {
            ptr = allocator.alloc_array::<T>(count)?;
            unsafe {
                copy_nonoverlapping(first, ptr.as_ptr(), count);
            }
        };
        Ok(BoxedSlice {
            ptr,
            allocator: ManuallyDrop::new(allocator),
            count,
        })
    }

    /// Tries to copy a slice.
    pub fn try_copy(&self) -> Result<Self, AllocErr>
    where
        A: Clone,
        T: Copy,
    {
        self.try_copy_in(A::clone(&self.allocator))
    }

    /// Tries to copy a slice.
    pub fn try_copy_in<NewAlloc>(
        &self,
        allocator: NewAlloc,
    ) -> Result<BoxedSlice<T, NewAlloc>, AllocErr>
    where
        T: Copy,
        NewAlloc: Alloc,
    {
        BoxedSlice::new_in(&self[..], allocator)
    }

    /// Tries to clone a slice.
    pub fn try_clone(&self) -> Result<Self, AllocErr>
    where
        A: Clone,
        T: Clone,
    {
        self.try_clone_in(A::clone(&self.allocator))
    }

    /// Tries to clone a slice.
    pub fn try_clone_in<NewAlloc>(
        &self,
        mut allocator: NewAlloc,
    ) -> Result<BoxedSlice<T, NewAlloc>, AllocErr>
    where
        T: Clone,
        NewAlloc: Alloc,
    {
        let count = self.len();

        let ptr;

        if size_of::<T>() == 0 || count == 0 {
            ptr = NonNull::dangling()
        } else {
            ptr = allocator.alloc_array::<T>(count)?;
            for (idx, item) in self.iter().enumerate() {
                let new_item = T::clone(item);
                unsafe {
                    ptr.as_ptr().add(idx).write(new_item);
                }
            }
        };

        Ok(BoxedSlice {
            allocator: ManuallyDrop::new(allocator),
            count,
            ptr,
        })
    }

    /// Breaks the box into raw parts.
    pub fn into_raw_parts(self) -> (NonNull<T>, usize, A) {
        let mut this = ManuallyDrop::new(self);
        let ptr = this.ptr;
        let allocator = unsafe { ManuallyDrop::take(&mut this.allocator) };
        (ptr, this.count, allocator)
    }

    /// Constructs BoxedSlice from its raw parts.
    pub unsafe fn from_raw_parts(ptr: NonNull<T>, count: usize, allocator: A) -> Self {
        BoxedSlice {
            ptr,
            count,
            allocator: ManuallyDrop::new(allocator),
        }
    }

    /// Transforms the box into a vector.
    pub fn into_vec(self) -> Vec<T, A> {
        Vec::from_boxed_slice(self)
    }
}

impl<T, A: Alloc> Deref for BoxedSlice<T, A> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr() as *const _, self.count) }
    }
}

impl<T, A: Alloc> DerefMut for BoxedSlice<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.count) }
    }
}

impl<T, A: Alloc> Drop for BoxedSlice<T, A> {
    fn drop(&mut self) {
        unsafe {
            // use drop for [T]
            ptr::drop_in_place(&mut self[..]);

            if size_of::<T>() != 0 && self.count != 0 {
                let _ = self.allocator.dealloc_array(self.ptr, self.count);
            }
            let _ = ManuallyDrop::take(&mut self.allocator);
        }
    }
}

impl<'a, T, A: Alloc> IntoIterator for &'a BoxedSlice<T, A> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        (self as &[T]).iter()
    }
}

impl<'a, T, A: Alloc> IntoIterator for &'a mut BoxedSlice<T, A> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        (self as &mut [T]).iter_mut()
    }
}

/// Custom-allocated `T`.
pub struct CustomBox<T, A: Alloc = Global> {
    ptr: NonNull<T>,
    allocator: ManuallyDrop<A>,
}

impl<T> CustomBox<T> {
    /// Puts given `item` into a new memory chunk, allocated by the global allocator.
    pub fn new(item: T) -> Result<Self, AllocErr> {
        Self::new_in(item, <_>::default())
    }
}

impl<T, A: Alloc> CustomBox<T, A> {
    /// Puts given `item` into a new memory chunk, allocated by a given allocator.
    pub fn new_in(item: T, mut allocator: A) -> Result<Self, AllocErr> {
        let heap_ptr = if size_of::<T>() == 0 {
            NonNull::<T>::dangling().cast()
        } else {
            allocator.alloc_one::<T>()?
        };
        unsafe { heap_ptr.as_ptr().write(item) }
        Ok(CustomBox {
            ptr: heap_ptr,
            allocator: ManuallyDrop::new(allocator),
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
        let new_ptr;
        if size_of::<T>() == 0 {
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
            ptr: new_ptr,
            allocator: ManuallyDrop::new(new_allocator),
        })
    }
}

impl<T, A: Alloc> Drop for CustomBox<T, A> {
    fn drop(&mut self) {
        unsafe {
            self.ptr.as_ptr().drop_in_place();
            let mut allocator = ManuallyDrop::take(&mut self.allocator);
            if size_of::<T>() != 0 {
                let _ = allocator.dealloc_one(self.ptr);
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

impl<T, A: Alloc> DerefMut for CustomBox<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self.ptr.as_ptr() as *mut T) }
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
        let mut boxed = BoxedSlice::new(vec).unwrap();
        assert_eq!(&boxed[..], &["l", "o", "l"]);

        boxed[0] = "ahaha";
        assert_eq!(&boxed[..], &["ahaha", "o", "l"]);
    }
}
