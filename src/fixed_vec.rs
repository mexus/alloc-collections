use crate::{
    alloc::{self, Alloc},
    raw_vec,
    vec::Vec,
};
use std::{
    fmt, mem,
    ops::{Deref, DerefMut},
};

/// Manually-destroyed fixed-sized vector.
pub struct FixedVec<T>(mem::ManuallyDrop<Vec<T, alloc::NoOp>>);

impl<T> FixedVec<T> {
    pub fn repeat_in<A: Alloc>(value: T, count: usize, alloc: A) -> Result<Self, raw_vec::Error>
    where
        T: Clone,
    {
        let vec = Vec::repeat_in(value, count, alloc)?;
        Ok(vec.into())
    }

    pub fn clone_with<A: Alloc>(&self, alloc: A) -> Result<Self, raw_vec::Error>
    where
        T: Clone,
    {
        let mut v = Vec::with_capacity_in(self.len(), alloc)?;
        v.extend_from_slice(&self[..])?;
        Ok(v.into())
    }

    pub unsafe fn destroy<A: Alloc>(&mut self, alloc: A) {
        let p = self.0.as_mut_ptr();
        let len = self.0.len();
        let cap = self.0.capacity();
        let _ = Vec::from_raw_parts(p, len, cap, alloc);
    }

    pub fn as_vec(&self) -> &Vec<T, alloc::NoOp> {
        &self.0
    }
}

impl<T, A: Alloc> From<Vec<T, A>> for FixedVec<T> {
    fn from(mut vec: Vec<T, A>) -> Self {
        let p = vec.as_mut_ptr();
        let len = vec.len();
        let cap = vec.capacity();
        mem::forget(vec);
        let v = unsafe { Vec::from_raw_parts(p, len, cap, alloc::NoOp) };
        Self(mem::ManuallyDrop::new(v))
    }
}

impl<T: fmt::Debug> fmt::Debug for FixedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<T> Deref for FixedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for FixedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn empty() {
        let mut f = FixedVec::repeat_in(0, 0, alloc::NoOp).unwrap();
        unsafe { f.destroy(alloc::Global) };
    }

    #[test]
    fn short() {
        let mut f = FixedVec::repeat_in(0, 100, alloc::Global).unwrap();
        unsafe { f.destroy(alloc::Global) };
    }
}
