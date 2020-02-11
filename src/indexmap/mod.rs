#[macro_use]
pub mod macros;

mod equivalent;
mod mutable_keys;
mod util;
pub mod map;

use crate::{vec::Vec, alloc::Alloc};
pub use equivalent::Equivalent;
pub use map::IndexMap;

/// Hash value newtype. Not larger than usize, since anything larger
/// isn't used for selecting position anyway.
#[derive(Copy, Debug)]
struct HashValue(usize);

impl HashValue {
    #[inline(always)]
    fn get(self) -> usize {
        self.0
    }
}

impl Clone for HashValue {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl PartialEq for HashValue {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Bucket<K, V> {
    hash: HashValue,
    key: K,
    value: V,
}

impl<K, V> Bucket<K, V> {
    // field accessors -- used for `f` instead of closures in `.map(f)`
    fn key_ref(&self) -> &K {
        &self.key
    }
    fn value_ref(&self) -> &V {
        &self.value
    }
    fn value_mut(&mut self) -> &mut V {
        &mut self.value
    }
    fn key_value(self) -> (K, V) {
        (self.key, self.value)
    }
    fn refs(&self) -> (&K, &V) {
        (&self.key, &self.value)
    }
    fn ref_mut(&mut self) -> (&K, &mut V) {
        (&self.key, &mut self.value)
    }
    fn muts(&mut self) -> (&mut K, &mut V) {
        (&mut self.key, &mut self.value)
    }
}

trait Entries {
    type Entry;
    type Alloc: Alloc;

    fn into_entries(self) -> Vec<Self::Entry, Self::Alloc>;
    fn as_entries(&self) -> &[Self::Entry];
    fn as_entries_mut(&mut self) -> &mut [Self::Entry];
    fn with_entries<F>(&mut self, f: F)
    where
        F: FnOnce(&mut [Self::Entry]);
}
