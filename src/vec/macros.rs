//! Macro for Vec

/// Vector initialization macros.
///
/// ```rust
/// use alloc_collections::vec;
///
/// let v = vec![1; 4].unwrap();
/// assert_eq!(v.as_slice(), &[1, 1, 1, 1]);
///
/// let v = vec![1, 2, 3, 4].unwrap();
/// assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
/// ```
#[macro_export]
macro_rules! vec {
    (@single $($x:tt)*) => (());
    (@count $($rest:expr),*) => (<[()]>::len(&[$($crate::vec!(@single $rest)),*]));

    ($item:expr; $count:expr) => {{
        let init = || -> Result<_, $crate::raw_vec::Error> {
            let mut v = $crate::Vec::with_capacity($count)?;
            v.resize($count, $item)?;
            Ok(v)
        };
        init()
    }};
    ($($x:expr),*) => {{
        let init = || -> Result<_, $crate::raw_vec::Error> {
            let cnt = $crate::vec!(@count $($x),*);
            let mut v = $crate::Vec::with_capacity(cnt)?;
            $(
                v.push($x)?;
            )*
            Ok(v)
        };
        init()
    }};
    ($($x:expr,)*) => (crate::vec![$($x),*])
}
