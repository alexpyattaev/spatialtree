// macro_rules! show_size {
//     (header) => (
//         println!("{:<28} {:>4}    {}", "Type", "T", "Option<T>");
//     );
//     ($t:ty) => (
//         println!("{:<28} {:4} {:4}", stringify!($t), size_of::<$t>(), size_of::<Option<$t>>())
//     )
// }

pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
}
