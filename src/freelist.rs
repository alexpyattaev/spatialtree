use std::{
    cmp::Eq,
    mem::ManuallyDrop,
    ops::{Deref, Index, IndexMut},
};

pub trait Indexer: Eq + Clone + Copy + core::fmt::Debug {
    /// Cast to usize
    fn usize(self) -> usize;

    /// Generate valid reference from index x (does not check if x is actually valid!)
    fn valid(x: usize) -> Self;

    /// Generate an invalid reference
    fn invalid() -> Self;

    /// Check if index is valid
    fn is_valid(self) -> bool;

    // fn used(&self)->bool{
    //
    // }
    //
    // fn set_used(& mut self, val:bool)->bool{
    //
    // }
}

impl Indexer for isize {
    #[inline]
    fn usize(self) -> usize {
        return (self - 1) as usize;
    }
    #[inline]
    fn valid(x: usize) -> Self {
        return (x + 1) as isize;
    }

    /// Generate an invalid reference
    #[inline]
    fn invalid() -> Self {
        return 0;
    }

    /// Check if index is valid
    #[inline]
    fn is_valid(self) -> bool {
        return self != 0;
    }
}

impl Indexer for i32 {
    #[inline]
    fn usize(self) -> usize {
        return (self - 1) as usize;
    }
    #[inline]
    fn valid(x: usize) -> Self {
        return (x + 1) as i32;
    }

    /// Generate an invalid reference
    #[inline]
    fn invalid() -> Self {
        return 0;
    }

    /// Check if index is valid
    #[inline]
    fn is_valid(self) -> bool {
        return self != 0;
    }
}

#[derive(Debug)]
struct Slot<T, DT: Indexer> {
    data: ManuallyDrop<T>,
    next_free: DT,
}

impl<T: Clone, DT: Indexer> Clone for Slot<T, DT> {
    fn clone(&self) -> Self {
        return Self {
            data: self.data.clone(),
            next_free: self.next_free,
        };
    }
}

impl<T, DT: Indexer> Slot<T, DT> {
    /// Overwrite self with new data T
    #[inline]
    fn overwrite(&mut self, data: T) {
        unsafe { ManuallyDrop::drop(&mut self.data) };
        self.data = ManuallyDrop::new(data);
    }
    // #[inline]
    // fn empty()->Self {
    //     Slot { data: unsafe { MaybeUninit::uninit().assume_init() }, next_free: None }
    // }
}

#[derive(Debug)]
pub struct FreeList<T, DT = i32>
where
    DT: Indexer,
{
    /// Actually hold slots with data
    entries: Vec<Slot<T, DT>>,

    ///SAFETY: this references a free Slot in entries vector. If this can not be maintained, panic!
    next_free: DT,

    /// number of used entries (typically smaller than entries length).
    /// This is important for Drop implementation so should be always correct.
    num_entries: usize,
    _last_used_idx: usize,
}

impl<T, DT: Indexer> Default for FreeList<T, DT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, DT: Indexer> Drop for FreeList<T, DT> {
    fn drop(&mut self) {
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.entries.as_mut_ptr(),
                self.entries.len(),
            ));
            self.entries.set_len(0);
        }
    }
}

impl<T, DT: Indexer> FreeList<T, DT> {
    pub fn new() -> Self {
        FreeList::with_capacity(0)
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: Vec::with_capacity(cap),
            next_free: DT::invalid(),
            num_entries: 0,
            _last_used_idx: 0,
        }
    }

    #[inline]
    pub fn add(&mut self, data: T) -> usize {
        let rv = if self.next_free.is_valid() {
            let i = self.next_free.usize();
            //SAFETY: we know for sure that next_free is a valid index
            let slot = unsafe { self.entries.get_unchecked_mut(i) };
            unsafe { ManuallyDrop::drop(&mut slot.data) };
            // Place new data in the slot
            slot.data = ManuallyDrop::new(data);
            // fix the invariants
            self.next_free = slot.next_free;
            slot.next_free = DT::invalid();
            i
        } else {
            // add more capacity (this may allocate)
            self.entries.push(Slot {
                data: ManuallyDrop::new(data),
                next_free: DT::invalid(),
            });

            self.entries.len() - 1
        };
        self.num_entries += 1;
        rv
    }
    #[inline]
    pub fn remove(&mut self, idx: usize) {
        //let entry = self.entries.get_mut(idx).expect("Provided index is not valid!");

        // We do not run Drop for the contained T, as we are just marking slot empty.
        self.entries[idx].next_free = self.next_free;
        self.next_free = DT::valid(idx);
        self.num_entries -= 1;
    }
    #[inline]
    pub fn remove_replace(&mut self, idx: usize, new_data: T) {
        let entry = self
            .entries
            .get_mut(idx)
            .expect("Provided index is not valid!");
        entry.overwrite(new_data);
    }

    /// An iterator over entire collection
    pub unsafe fn iter_raw(&self) -> impl Iterator<Item = &T> {
        self.entries.iter().map(|v| v.data.deref())
    }

    // /// An iterator over all filled slots
    // pub unsafe fn iter(&self) -> impl Iterator<Item = &T> {
    //     self.entries.iter().filter_map(|v| v.data.as_ref())
    // }

    /// Number of allocated slots
    #[inline]
    pub fn allocated_slots(&self) -> usize {
        self.entries.capacity()
    }

    /// Number of actually used slots
    #[inline]
    pub fn used_slots(&self) -> usize {
        self.num_entries
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, idx: DT) -> &T {
        self.entries.get_unchecked(idx.usize()).data.deref()
    }

    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, idx: DT) -> &mut T {
        &mut self.entries.get_unchecked_mut(idx.usize()).data
    }

    /// Drop contigous block of unused memory from the tail of freelist.
    /// Will not rearrange items, so this does nothing to remove holes.
    pub fn shrink_to_fit(&mut self) {
        todo!()
    }
}

const PANIC_MSG: &str = "Invalid index into FreeList";
impl<T, DT: Indexer> Index<usize> for FreeList<T, DT> {
    type Output = T;
    /// Index into the datastructure. If index points to an invalid location, this will panic.
    fn index(&self, index: usize) -> &Self::Output {
        self.entries.get(index).expect(PANIC_MSG).data.deref()
    }
}

impl<T, DT: Indexer> IndexMut<usize> for FreeList<T, DT> {
    /// Index into the datastructure. If index points to an invalid location, this will panic.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.entries.get_mut(index).expect(PANIC_MSG).data
    }
}

impl<T: Clone, DT: Indexer> Clone for FreeList<T, DT> {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            next_free: self.next_free,
            num_entries: self.num_entries,
            _last_used_idx: self._last_used_idx,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[derive(Debug)]
    struct Chunk {
        x: usize,
    }

    impl Drop for Chunk {
        fn drop(&mut self) {
            println!("Dropping {}", self.x)
        }
    }

    #[test]
    fn freelist_basic_ops() {
        let mut fl: FreeList<Chunk, i32> = FreeList::new();
        println!("Allocated memory for {}", fl.allocated_slots());
        let idx1 = fl.add(Chunk { x: 1 });
        fl[idx1].x += 1;
        assert_eq!(fl[idx1].x, 2);
        let _idx2 = fl.add(Chunk { x: 3 });
        fl.remove(idx1);
        assert_eq!(fl.used_slots(), 1);

        let idx3 = fl.add(Chunk { x: 4 });
        assert_eq!(idx1, idx3);
        println!("{:?},{:?}", fl.allocated_slots(), fl.used_slots());
    }

    #[test]
    fn freelist_sizes() {
        use crate::util_funcs::*;
        let s: Slot<u32, i32> = Slot {
            data: ManuallyDrop::new(0),
            next_free: 0,
        };
        let sl = unsafe { any_as_u8_slice(&s) };
        println!("{:?}", sl);
    }

    #[test]
    #[should_panic(expected = "Invalid index into FreeList")]
    fn freelist_index_error_panic() {
        let fl: FreeList<u32, i32> = FreeList::new();
        let _z = fl[42];
    }
    #[test]
    #[should_panic(expected = "Invalid index into FreeList")]
    fn freelist_emptied_slot_error_panic() {
        let mut fl: FreeList<u32, i32> = FreeList::new();
        fl.add(16 as u32);
        fl.remove(0);
        let _z = fl[0];
    }
}
