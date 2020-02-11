//! A crate with collections that do not rely on global allocator.

pub mod alloc;
pub mod indexmap;
pub mod layout_ext;
pub mod raw_vec;
pub mod unique;
pub mod vec;
pub mod fixed_vec;

pub use alloc::Alloc;
pub use indexmap::IndexMap;
pub use layout_ext::LayoutExt;
pub use vec::Vec;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_works() {
        let alloc = crate::alloc::System::default();
        let mut v = Vec::new_in(alloc);
        v.push("Wow!").unwrap();
        assert_eq!(&v[..], &["Wow!"]);
        v.pop();
        assert!(v.is_empty());
    }
}
