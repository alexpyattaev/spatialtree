/* Generic tree structures for storage of spatial data.
 * Copyright (C) 2023  Alexander Pyattaev
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

use std::num::NonZeroU32;
use crate::coords::*;

/// Utility function to cast random structures into arrays of bytes
/// This is mostly for debugging purposes
/// # Safety
/// This is safe since returned slice is readonly (as long as you do not modify thing it is pointing into)
pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
}



/// Type for relative pointers to nodes in the tree. Kept 32bit for cache locality during lookups.
/// Should you need > 4 billion nodes in the tree do let me know who sells you the RAM.
pub type NodePtr = Option<NonZeroU32>;

/// Type for relative pointers to chunks in the tree. Kept 32bit for cache locality during lookups.
/// Encodes "None" variant as -1, and Some(idx) as positive numbers
/// This is not as fast as NonZeroU32, but close enough for our needs
/// Should you need > 2 billion chunks in the tree do let me know who sells you the RAM.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ChunkPtr(i32);
impl core::fmt::Display for ChunkPtr {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("ChunkPtr({:?})", self.get()))
    }
}

impl ChunkPtr {
    #[inline]
    pub(crate) fn get(self) -> Option<usize> {
        match self.0 {
            -1 => None,
            _ => Some(self.0 as usize),
        }
    }

    #[inline]
    pub(crate) fn take(&mut self) -> Option<usize> {
        let rv = match self.0 {
            -1 => None,
            _ => Some(self.0 as usize),
        };
        *self = Self::None;
        rv
    }

    #[inline]
    pub(crate) fn from(x: Option<usize>) -> Self {
        match x {
            Some(v) => Self ( v as i32 ),
            None => Self::None,
        }
    }
    // Cheat for "compatibility" with option.
    #[allow(non_upper_case_globals)]
    pub const None: Self = ChunkPtr( -1 );
}

//TODO: impl Try once stable
/*impl std::ops::Try for ChunkPtr{
 *    type Output = usize;
 *
 *    type Residual;
 *
 *    fn from_output(output: Self::Output) -> Self {
 *        todo!()
 *    }
 *
 *    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
 *        todo!()
 *    }
 * }*/


//TODO - use struct of arrays?
/// Tree node that encompasses multiple children at once. This just barely fits into one cache line for octree.
/// For each possible child, the node has two relative pointers:
///  * children will point to the TreeNode in a given branch direction
///  * chunk will point to the data chunk in a given branch direction
/// both pointers may be "None", indicating either no children, or no data
#[derive(Clone, Debug)]
pub struct TreeNode<const B: usize> {
    /// children, these can't be the root (index 0), so we can use Some and Nonzero for slightly more compact memory
    pub children: [NodePtr; B],

    /// where the chunks for particular children is stored (if any)
    pub chunk: [ChunkPtr; B],
}

impl<const B: usize> TreeNode<B> {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            children: [None; B],
            chunk: [ChunkPtr::None; B],
        }
    }

    #[inline]
    pub fn iter_existing_chunks(
        &self,
    ) -> impl Iterator<Item = (usize, usize)>  +'_ {
        self.chunk.iter().filter_map(|c| c.get()).enumerate()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
         self.children.iter().all(|c| c.is_none()) && self.chunk.iter().all(|c| *c== ChunkPtr::None)
    }
}

#[inline]
pub fn iter_treenode_children< const N: usize>(
    children: &[NodePtr; N],
) -> impl Iterator<Item = (usize, usize)> + '_ {
    children
    .iter()
    .filter_map(|c| Some((*c)?.get() as usize))
    .enumerate()
}

// utility struct for holding actual chunks and the node that owns them
#[derive(Clone, Debug)]
pub struct ChunkContainer<const N: usize, C: Sized, L: LodVec<N>> {
    /// actual data inside the chunk
    pub chunk: C,
    // where the chunk is (as this can not be easily recovered from node tree).
    pub(crate) position: L,
    // index of the node that holds this chunk. Do not modify unless you know what you are doing!
    pub(crate) node_idx: u32,
    // index of the child in the node. Do not modify unless you know what you are doing!
    pub(crate) child_idx: u8,
}

impl<const N: usize, C: Sized, L: LodVec<N>> ChunkContainer<N, C, L> {
    /// get an mutable pointer to chunk which is not tied to the lifetime of the container
    /// this is only needed for iterators.
    #[inline(always)]
    pub fn chunk_ptr(&mut self) -> *mut C {
        &mut self.chunk as *mut C
    }
    /// where the chunk is. Modifying this
    /// will not move the chunk to a new position, so this is readonly
    #[inline(always)]
    pub fn position(&self) -> L {
        self.position
    }
}

/// utility struct for holding locations in the tree.
#[derive(Clone, Debug, Copy)]
pub struct TreePos<const N: usize, L: LodVec<N>> {
    /// node or chunk index
    pub idx: usize,
    /// and it's position
    pub pos: L,
}
