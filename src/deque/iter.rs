//! VecDeque iterators.

use super::VecDeque;
use crate::Alloc;
use core::iter::ExactSizeIterator;
use std::marker::PhantomData;

pub struct Iter<'a, T> {
    buf: *const T,
    head: usize,
    length: usize,
    capacity: usize,
    _pd: PhantomData<&'a ()>,
}

impl<'a, T> Iter<'a, T> {
    pub(super) fn new<A: Alloc>(queue: &'a VecDeque<T, A>) -> Self {
        Self {
            buf: queue.buf.as_ptr(),
            head: queue.head,
            length: queue.length,
            capacity: queue.capacity(),
            _pd: PhantomData,
        }
    }
}

impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.length == 0 {
            None
        } else {
            let current = self.head;
            let next = current.wrapping_add(1) % self.capacity;
            self.head = next;
            self.length -= 1;
            unsafe { Some(&*self.buf.add(current)) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, T: 'a> ExactSizeIterator for Iter<'a, T> {
    fn len(&self) -> usize {
        self.length
    }
}

pub struct IterMut<'a, T> {
    buf: *mut T,
    head: usize,
    length: usize,
    capacity: usize,
    _pd: PhantomData<&'a mut ()>,
}

impl<'a, T> IterMut<'a, T> {
    pub(super) fn new<A: Alloc>(queue: &'a mut VecDeque<T, A>) -> Self {
        Self {
            buf: queue.buf.as_ptr(),
            head: queue.head,
            length: queue.length,
            capacity: queue.capacity(),
            _pd: PhantomData,
        }
    }
}

impl<'a, T: 'a> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.length == 0 {
            return None;
        }
        let current = self.head;
        let next = current.wrapping_add(1) % self.capacity;
        self.head = next;
        self.length -= 1;

        let next = unsafe { &mut *self.buf.add(current) };
        Some(next)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, T: 'a> ExactSizeIterator for IterMut<'a, T> {
    fn len(&self) -> usize {
        self.length
    }
}

#[derive(Clone)]
pub struct IntoIter<T, A: Alloc> {
    inner: VecDeque<T, A>,
}

impl<T, A: Alloc> IntoIter<T, A> {
    pub(super) fn new(queue: VecDeque<T, A>) -> Self {
        Self { inner: queue }
    }
}

impl<T, A: Alloc> Iterator for IntoIter<T, A> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.pop_back()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<T, A: Alloc> ExactSizeIterator for IntoIter<T, A> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn check() {
        let mut v = VecDeque::<u16>::with_capacity(10).unwrap();
        v.head = 5;
        for i in 0..10 {
            v.push_back(i).unwrap();
        }
        assert_ne!(v.head, 0);

        assert_eq!(v.iter().len(), 10);
        for (idx, value) in v.iter().enumerate() {
            assert_eq!(idx, usize::from(*value));
        }

        assert_eq!(v.iter_mut().len(), 10);
        for (idx, value) in v.iter_mut().enumerate() {
            assert_eq!(idx, usize::from(*value));
        }
    }
}
