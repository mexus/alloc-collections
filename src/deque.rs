//! Double-ended queue

use crate::{
    alloc::Global,
    raw_vec::{self, RawVec},
    Alloc,
};
use core::{fmt, ptr, slice};

pub mod iter;

/// Double-ended queue.
///
/// # Using in shared memory regions.
///
/// This queue aims to provide the following guarantee:
///
/// If a process that pushing data to the queue crashes in the middle of the action,
/// e.g. before the data is fully written, the internal counters are increased in such an order,
/// so reading partially-initialized data by other processes becomes impossible.
///
/// This behaviour is achieved by using volatile writes, so it won't safe you if you don't provide
/// some kind of a memory barrier (by using mutex, for example).
pub struct VecDeque<T, A: Alloc = Global> {
    buf: RawVec<T, A>,

    // First item in the queue.
    head: usize,

    length: usize,
}

unsafe impl<T: Send, A: Alloc + Send> Send for VecDeque<T, A> {}
unsafe impl<T: Sync, A: Alloc + Sync> Sync for VecDeque<T, A> {}

impl<T> Default for VecDeque<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> VecDeque<T> {
    /// Creates a VecDeque with a zero capacity.
    pub fn new() -> Self {
        Self::new_in(<_>::default())
    }

    /// Creates a VecDeque with a given capacity.
    pub fn with_capacity(capacity: usize) -> Result<Self, raw_vec::Error> {
        Self::with_capacity_in(capacity, <_>::default())
    }

    /// Tries to construct a queue from a given iterator.
    pub fn try_from_iter<I>(i: I) -> Result<Self, raw_vec::Error>
    where
        I: IntoIterator<Item = T>,
    {
        Self::try_from_iter_in(i, <_>::default())
    }
}

impl<T, A: Alloc> VecDeque<T, A> {
    /// Creates a VecDeque with a zero capacity using a given allocator.
    pub fn new_in(alloc: A) -> Self {
        Self::with_capacity_in(0, alloc).expect("Should not allocate")
    }

    /// Creates a VecDeque with a given capacity using a given allocator.
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Result<Self, raw_vec::Error> {
        let buf = RawVec::with_capacity_in(capacity, alloc)?;
        Ok(Self {
            buf,
            head: 0,
            length: 0,
        })
    }

    /// Tries to construct a queue from a given iterator.
    pub fn try_from_iter_in<I>(i: I, allocator: A) -> Result<Self, raw_vec::Error>
    where
        I: IntoIterator<Item = T>,
    {
        let mut vec = VecDeque::new_in(allocator);
        vec.try_extend(i)?;
        Ok(vec)
    }

    /// Tries to extend the queue with a given iterator.
    pub fn try_extend<I>(&mut self, i: I) -> Result<(), raw_vec::Error>
    where
        I: IntoIterator<Item = T>,
    {
        let iter = i.into_iter();
        self.grow_if_necessary(estimation(&iter))?;
        for item in iter {
            self.push_back(item)?;
        }
        Ok(())
    }

    /// Returns iterator over stored items.
    pub fn iter(&self) -> iter::Iter<T> {
        iter::Iter::new(self)
    }

    /// Returns iterator over stored items.
    pub fn iter_mut(&mut self) -> iter::IterMut<T> {
        iter::IterMut::new(self)
    }

    /// Returns a reference to the `n`th item, if any.
    pub fn get(&self, n: usize) -> Option<&T> {
        if n < self.len() {
            unsafe { Some(self.get_unchecked(n)) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to the `n`th item, if any.
    pub fn get_mut(&mut self, n: usize) -> Option<&mut T> {
        if n < self.len() {
            unsafe { Some(self.get_mut_unchecked(n)) }
        } else {
            None
        }
    }

    /// Returns a reference to the `n`th item.
    ///
    /// # Safety
    ///
    /// Behaviour is undefined if `n` is out of bounds.
    pub unsafe fn get_unchecked(&self, n: usize) -> &T {
        &*self.nth_unchecked(n)
    }

    /// Returns a mutable reference to the `n`th item.
    ///
    /// # Safety
    ///
    /// Behaviour is undefined if `n` is out of bounds.
    pub unsafe fn get_mut_unchecked(&mut self, n: usize) -> &mut T {
        &mut *self.nth_unchecked_mut(n)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns length of the queue.
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns a pair of slices that represents the queue.
    pub fn as_slices(&self) -> (&[T], &[T]) {
        unsafe {
            if self.is_empty() {
                (&[], &[])
            } else if self.is_contiguous() {
                (
                    slice::from_raw_parts(self.nth_unchecked(0), self.len()),
                    &[],
                )
            } else {
                let first =
                    slice::from_raw_parts(self.nth_unchecked(0), self.buf.capacity() - self.head);
                let second = slice::from_raw_parts(
                    self.buf.as_ptr(),
                    (self.head + self.length) % self.capacity(),
                );
                (first, second)
            }
        }
    }

    /// Returns a pair of slices that represents the queue.
    pub fn as_slices_mut(&mut self) -> (&mut [T], &mut [T]) {
        unsafe {
            if self.is_contiguous() {
                (
                    slice::from_raw_parts_mut(self.buf.as_ptr().add(self.head), self.len()),
                    &mut [],
                )
            } else {
                let first = slice::from_raw_parts_mut(
                    self.buf.as_ptr().add(self.head),
                    self.buf.capacity() - self.head,
                );
                let second = slice::from_raw_parts_mut(
                    self.buf.as_ptr(),
                    (self.head + self.length) % self.capacity(),
                );
                (first, second)
            }
        }
    }

    /// Tries to push an item to the end (back) of the queue.
    pub fn push_back(&mut self, item: T) -> Result<(), raw_vec::Error> {
        self.grow_if_necessary(1)?;

        let old_length = self.length;
        let new_length = old_length
            .checked_add(1)
            .ok_or(raw_vec::Error::CapacityOverflow)?;

        unsafe {
            self.buf
                .as_ptr()
                .add((self.head + old_length) % self.buf.capacity())
                .write_volatile(item)
        };

        // Update length only after we've finished writing data.
        unsafe { (&mut self.length as *mut usize).write_volatile(new_length) };
        Ok(())
    }

    /// Tries to push an item to the beginning (front) of the queue.
    pub fn push_front(&mut self, item: T) -> Result<(), raw_vec::Error> {
        self.grow_if_necessary(1)?;
        let old_length = self.length;
        let old_head = self.head;

        let new_length = old_length
            .checked_add(1)
            .ok_or(raw_vec::Error::CapacityOverflow)?;

        let new_head = old_head
            .checked_sub(1)
            .unwrap_or_else(|| self.buf.capacity() - 1);

        // Update head before we start writing data.
        unsafe { (&mut self.head as *mut usize).write_volatile(new_head) };

        // Now, write data.
        unsafe { self.buf.as_ptr().add(new_head).write_volatile(item) };

        // Update length only after we've updated head.
        unsafe { (&mut self.length as *mut usize).write_volatile(new_length) };
        Ok(())
    }

    /// Tries to pop an item from the front of the queue.
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let item = unsafe { self.nth_unchecked_mut(0).read() };
            self.head = (self.head + 1) % self.capacity();
            self.length -= 1;
            Some(item)
        }
    }

    /// Tries to pop an item from the back of the queue.
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let item = unsafe { self.nth_unchecked_mut(self.length - 1).read() };
            self.length -= 1;
            Some(item)
        }
    }

    /// Full capacity of the queue.
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// How many items can be inserted to the queue without reallocation.
    pub fn remaining_capacity(&self) -> usize {
        self.capacity() - self.len()
    }

    /// Tries to reserve space to insert additional items.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), raw_vec::Error> {
        self.grow_if_necessary(additional)
    }

    /// Allows to insert more `new_items` to the queue.
    fn grow_if_necessary(&mut self, new_items: usize) -> Result<(), raw_vec::Error> {
        match new_items.checked_sub(self.remaining_capacity()) {
            None | Some(0) => { /* No op */ }
            Some(shortage) if self.capacity() == 0 => self.buf.try_reserve(0, shortage)?,
            Some(shortage) => {
                let used_capacity = if self.is_contiguous() {
                    self.head + self.length
                } else {
                    self.capacity()
                };

                let old_capacity = self.capacity();
                self.buf.try_reserve(used_capacity, shortage)?;
                let new_capacity = self.capacity();
                debug_assert!(new_capacity >= self.len() + shortage);
                debug_assert!(new_capacity >= old_capacity * 2);

                let old_tail_length = (self.head + self.length) % old_capacity;
                let old_head_length = self.length - old_tail_length;

                // Move the shortest contiguous section of the ring buffer
                //
                // A: tail is empty.
                //    H
                //   [| | | | | | | - ]
                //    H
                //   [| | | | | | | - - - - - - - - - ]
                //
                // B: tail is shorter than head
                //          H
                //   [| | - | | | | | ]  cap = 8, len = 7, head = 3, head_len = 5, tail_len = 2
                //          H
                //   [- - - | | | | | | |- - - - - - ]
                //
                // C: tail is longer than head
                //                H
                //   [| | | | | - | | ] cap = 8, len = 7, head = 6, head_len = 2, tail_len = 5
                //                                H
                //   [| | | | | - - - - - - - - - | | ]

                if old_tail_length == 0 {
                    // Case A.
                    // No op.
                } else if old_tail_length < old_head_length {
                    // Case B.
                    // debug_assert!();
                    unsafe {
                        ptr::copy_nonoverlapping(
                            self.buf.as_ptr(),
                            self.buf.as_ptr().add(self.head + old_head_length),
                            old_tail_length,
                        );
                    }
                    debug_assert!(self.is_contiguous());
                } else {
                    // Case C.
                    let new_head = new_capacity - old_head_length;
                    debug_assert!(new_head > self.head + old_head_length);
                    debug_assert!(new_head + old_head_length >= new_capacity);
                    unsafe {
                        ptr::copy_nonoverlapping(
                            self.buf.as_ptr().add(self.head),
                            self.buf.as_ptr().add(new_head),
                            old_head_length,
                        );
                    }
                    self.head = new_head;
                    debug_assert!(!self.is_contiguous());
                }
            }
        };
        Ok(())
    }

    /// Whether or not the queue can be represented as a single slice.
    #[inline]
    fn is_contiguous(&self) -> bool {
        self.head + self.length <= self.capacity()
    }

    /// Returns a raw pointer to the `n`-th item in the queue without performing any bounds checks.
    unsafe fn nth_unchecked(&self, n: usize) -> *const T {
        debug_assert!(n < self.len());
        let wrapped = (self.head + n) % self.capacity();
        self.buf.as_ptr().add(wrapped)
    }

    /// Returns a raw pointer to the `n`-th item in the queue without performing any bounds checks.
    unsafe fn nth_unchecked_mut(&mut self, n: usize) -> *mut T {
        debug_assert!(n < self.len());
        let wrapped = (self.head + n) % self.capacity();
        self.buf.as_ptr().add(wrapped)
    }

    /// Drops all the items in the queue.
    pub fn clear(&mut self) {
        let (head, tail) = self.as_slices_mut();
        unsafe {
            // use drop for [T]
            ptr::drop_in_place(head);
            ptr::drop_in_place(tail);
        }
        self.head = 0;
        self.length = 0;
    }

    /// Tries to copy the queue.
    pub fn try_copy(&self) -> Result<Self, raw_vec::Error>
    where
        A: Clone,
        T: Copy,
    {
        let allocator = A::clone(self.buf.alloc());
        self.try_copy_in(allocator)
    }

    /// Tries to copy the queue into another allocator.
    pub fn try_copy_in<NewAlloc>(
        &self,
        alloc: NewAlloc,
    ) -> Result<VecDeque<T, NewAlloc>, raw_vec::Error>
    where
        NewAlloc: Alloc,
        T: Copy,
    {
        let mut new_queue = VecDeque::<T, _>::with_capacity_in(self.len(), alloc)?;
        let (head, tail) = self.as_slices();
        unsafe {
            new_queue
                .buf
                .as_ptr()
                .copy_from_nonoverlapping(head.as_ptr(), head.len());
            new_queue
                .buf
                .as_ptr()
                .add(head.len())
                .copy_from_nonoverlapping(tail.as_ptr(), tail.len());
        }
        new_queue.head = 0;
        new_queue.length = self.len();
        Ok(new_queue)
    }

    /// Tries to clone the queue.
    pub fn try_clone(&self) -> Result<Self, raw_vec::Error>
    where
        A: Clone,
        T: Clone,
    {
        let alloc = A::clone(self.buf.alloc());
        self.try_clone_in(alloc)
    }

    /// Tries to clone the queue into a given allocator.
    pub fn try_clone_in<NewAlloc>(
        &self,
        alloc: NewAlloc,
    ) -> Result<VecDeque<T, NewAlloc>, raw_vec::Error>
    where
        NewAlloc: Alloc,
        T: Clone,
    {
        let mut new_vec = VecDeque::<T, _>::with_capacity_in(self.len(), alloc)?;
        for item in self.iter() {
            new_vec.push_back(T::clone(item))?;
        }
        Ok(new_vec)
    }
}

impl<T, A: Alloc> Drop for VecDeque<T, A> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T, A: Alloc> std::ops::Index<usize> for VecDeque<T, A> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
            .unwrap_or_else(|| panic!("Index {} is out of bounds (length = {})", index, self.len()))
    }
}

impl<'a, T, A: Alloc> IntoIterator for &'a VecDeque<T, A> {
    type IntoIter = iter::Iter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, A: Alloc> IntoIterator for &'a mut VecDeque<T, A> {
    type IntoIter = iter::IterMut<'a, T>;
    type Item = &'a mut T;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: fmt::Debug, A: Alloc> fmt::Debug for VecDeque<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        let mut iter = self.iter();
        if let Some(first) = iter.next() {
            write!(f, "{:?}", first)?;
        }
        for item in iter {
            write!(f, ", {:?}", item)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<T1, T2, A> PartialEq<[T1]> for VecDeque<T2, A>
where
    T2: PartialEq<T1>,
    A: Alloc,
{
    fn eq(&self, other: &[T1]) -> bool {
        if self.len() != other.len() {
            return false;
        }
        let (head, tail) = self.as_slices();
        let (other_head, other_tail) = other.split_at(head.len());
        head == other_head && tail == other_tail
    }
}

impl<T1, T2, A> PartialEq<VecDeque<T2, A>> for [T1]
where
    T2: PartialEq<T1>,
    A: Alloc,
{
    fn eq(&self, other: &VecDeque<T2, A>) -> bool {
        PartialEq::eq(other, self)
    }
}

impl<T1, T2, A1, A2> PartialEq<VecDeque<T1, A1>> for VecDeque<T2, A2>
where
    T2: PartialEq<T1>,
    A1: Alloc,
    A2: Alloc,
{
    fn eq(&self, other: &VecDeque<T1, A1>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        let (sa, sb) = self.as_slices();
        let (oa, ob) = other.as_slices();
        if sa.len() == oa.len() {
            sa == oa && sb == ob
        } else if sa.len() < oa.len() {
            // Always divisible in three sections, for example:
            // self:  [a b c|d e f]
            // other: [0 1 2 3|4 5]
            // front = 3, mid = 1,
            // [a b c] == [0 1 2] && [d] == [3] && [e f] == [4 5]
            let front = sa.len();
            let mid = oa.len() - front;

            let (oa_front, oa_mid) = oa.split_at(front);
            let (sb_mid, sb_back) = sb.split_at(mid);
            debug_assert_eq!(sa.len(), oa_front.len());
            debug_assert_eq!(sb_mid.len(), oa_mid.len());
            debug_assert_eq!(sb_back.len(), ob.len());
            sa == oa_front && sb_mid == oa_mid && sb_back == ob
        } else {
            let front = oa.len();
            let mid = sa.len() - front;

            let (sa_front, sa_mid) = sa.split_at(front);
            let (ob_mid, ob_back) = ob.split_at(mid);
            debug_assert_eq!(sa_front.len(), oa.len());
            debug_assert_eq!(sa_mid.len(), ob_mid.len());
            debug_assert_eq!(sb.len(), ob_back.len());
            sa_front == oa && sa_mid == ob_mid && sb == ob_back
        }
    }
}

fn estimation<I: Iterator>(iter: &I) -> usize {
    let (lower, maybe_upper) = iter.size_hint();
    if let Some(upper) = maybe_upper {
        upper
    } else {
        lower
    }
}

impl<T, A: Alloc> IntoIterator for VecDeque<T, A> {
    type Item = T;
    type IntoIter = iter::IntoIter<T, A>;

    fn into_iter(self) -> Self::IntoIter {
        iter::IntoIter::new(self)
    }
}

impl<T: Clone, A: Alloc + Clone> Clone for VecDeque<T, A> {
    fn clone(&self) -> Self {
        self.try_clone().expect("Unable to clone a queue")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use core::fmt::Debug;
    use quickcheck::{Arbitrary, TestResult};
    use quickcheck_macros::quickcheck;
    use rand::Rng;
    use std::rc::Rc;

    #[test]
    fn check_len() {
        let mut q = VecDeque::<usize>::with_capacity(8).unwrap();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);

        q.push_back(5).unwrap();
        assert_eq!(q.len(), 1);
    }

    fn check_growth_impl<T: Clone + Debug + PartialEq>(
        original: &[T],
        initial_shift: usize,
        add: usize,
    ) -> TestResult {
        if original
            .len()
            .checked_add(initial_shift)
            .and_then(|len| len.checked_add(add))
            .is_none()
        {
            return TestResult::discard();
        }

        let mut vec = VecDeque::<T>::with_capacity(original.len() + initial_shift)
            .expect("Unable to allocate");
        vec.head += initial_shift;
        vec.try_extend(original.iter().cloned())
            .expect("Unable to populate queue");
        vec.try_reserve(add).unwrap_or_else(|e| {
            panic!(
                "Unable to reserve additional space for `{}` items: {}",
                add, e
            )
        });
        assert_eq!(&vec, original);
        TestResult::passed()
    }

    #[quickcheck]
    fn check_growth_u8(original: Vec<u8>, initial_shift: usize, add: usize) -> TestResult {
        check_growth_impl(&original, initial_shift, add)
    }

    #[quickcheck]
    fn check_growth_string(original: Vec<String>, initial_shift: usize, add: usize) -> TestResult {
        if original
            .len()
            .checked_add(initial_shift)
            .and_then(|len| len.checked_add(add))
            .is_none()
        {
            return TestResult::discard();
        }
        check_growth_impl(&original, initial_shift, add)
    }

    fn check_push<T: Debug + Clone + PartialEq>(
        original: &[T],
        init_items: usize,
        front: bool,
    ) -> TestResult {
        if init_items > original.len() {
            return TestResult::discard();
        }
        let mut v = VecDeque::with_capacity(init_items).unwrap();
        v.try_extend(original.iter().take(init_items).cloned())
            .unwrap();

        let mut reference: std::collections::VecDeque<T> =
            original.iter().take(init_items).cloned().collect();

        if front {
            for item in original.iter().skip(init_items) {
                v.push_front(item.clone()).expect("Unable to push front");
                reference.push_front(item.clone());
            }
        } else {
            for item in original.iter().skip(init_items) {
                v.push_back(item.clone()).expect("Unable to push back");
                reference.push_back(item.clone());
            }
        }

        let result: std::vec::Vec<T> = v.iter().cloned().collect();
        let reference: std::vec::Vec<T> = reference.into_iter().collect();
        assert_eq!(result, reference);

        TestResult::passed()
    }

    #[quickcheck]
    fn check_push_u8(original: Vec<u8>, init_items: usize, front: bool) -> TestResult {
        check_push(&original, init_items, front)
    }

    #[quickcheck]
    fn check_push_usize(original: Vec<usize>, init_items: usize, front: bool) -> TestResult {
        check_push(&original, init_items, front)
    }

    #[quickcheck]
    fn check_push_string(original: Vec<String>, init_items: usize, front: bool) -> TestResult {
        check_push(&original, init_items, front)
    }

    #[derive(Debug, Clone, PartialEq)]
    enum Action<T> {
        PopBack,
        PushBack(T),
        PopFront,
        PushFront(T),
        Clear,
        Get(usize),
    }

    impl<T: Arbitrary> Arbitrary for Action<T> {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            match g.gen_range(0, 6) {
                0 => Action::PopBack,
                1 => {
                    let item = T::arbitrary(g);
                    Action::PushBack(item)
                }
                2 => Action::PopFront,
                3 => {
                    let item = T::arbitrary(g);
                    Action::PushFront(item)
                }
                4 => Action::Clear,
                5 => {
                    let n = g.gen_range(0, 1000);
                    Action::Get(n)
                }
                _ => unreachable!(),
            }
        }
    }

    fn check_actions<T, I>(initial: &[T], actions: I)
    where
        T: Clone + Debug + PartialEq,
        I: IntoIterator<Item = Action<T>>,
    {
        let mut result = VecDeque::try_from_iter(initial.iter().cloned()).expect("Unable to init");
        let mut reference: std::collections::VecDeque<T> = initial.iter().cloned().collect();

        for action in actions {
            match action {
                Action::PopBack => {
                    let result = result.pop_back();
                    let reference = reference.pop_back();
                    assert_eq!(result, reference);
                }
                Action::PopFront => {
                    let result = result.pop_front();
                    let reference = reference.pop_front();
                    assert_eq!(result, reference);
                }
                Action::PushBack(item) => {
                    result.push_back(item.clone()).expect("Push back failed");
                    reference.push_back(item);
                }
                Action::PushFront(item) => {
                    result.push_front(item.clone()).expect("Push front failed");
                    reference.push_front(item);
                }
                Action::Clear => {
                    result.clear();
                    reference.clear();
                }
                Action::Get(idx) => {
                    let result = result.get(idx);
                    let reference = reference.get(idx);
                    assert_eq!(result, reference);
                }
            }
        }

        let result: std::vec::Vec<T> = result.iter().cloned().collect();
        let reference: std::vec::Vec<T> = reference.into_iter().collect();
        assert_eq!(result, reference);
    }

    #[quickcheck]
    fn check_actions_u8(initial: Vec<u8>, actions: Vec<Action<u8>>) {
        check_actions(&initial, actions)
    }

    #[quickcheck]
    fn check_actions_usize(initial: Vec<usize>, actions: Vec<Action<usize>>) {
        check_actions(&initial, actions)
    }

    #[quickcheck]
    fn check_actions_string(initial: Vec<String>, actions: Vec<Action<String>>) {
        check_actions(&initial, actions)
    }

    #[test]
    fn check_drops() {
        let r1 = Rc::new(123);
        let r2 = Rc::new(456);

        {
            let mut v = VecDeque::new();
            v.push_back(Rc::clone(&r1)).unwrap();
            v.push_front(r2.clone()).unwrap();

            let (head, tail) = v.as_slices();
            assert_eq!(head, &[r2.clone()]);
            assert_eq!(tail, &[r1.clone()]);
            assert_eq!(Rc::strong_count(&r1), 2);
            assert_eq!(Rc::strong_count(&r2), 2);
        }

        assert_eq!(Rc::strong_count(&r1), 1);
        assert_eq!(Rc::strong_count(&r2), 1);
    }

    #[quickcheck]
    fn check_clone(front: Vec<String>, back: Vec<String>) {
        let mut v = VecDeque::with_capacity(front.len() + back.len()).unwrap();
        for item in back {
            v.push_back(item).unwrap();
        }
        for item in front {
            v.push_front(item).unwrap();
        }

        let clone = v.clone();
        assert_eq!(v, clone);
    }

    #[quickcheck]
    fn check_copy(front: Vec<usize>, back: Vec<usize>) {
        let mut v = VecDeque::with_capacity(front.len() + back.len()).unwrap();
        for item in back {
            v.push_back(item).unwrap();
        }
        for item in front {
            v.push_front(item).unwrap();
        }

        let clone = v.try_copy().unwrap();
        assert_eq!(v, clone);
    }
}
