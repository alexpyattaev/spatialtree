/* Generic tree structures for storage of spatial data.
Copyright (C) 2023  Alexander Pyattaev

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

//! Iterators over tree data and over coordinates
use crate::coords::*;
use crate::tree::*;

/// iterator for all coordinates that are inside given bounds
pub struct CoordsInBoundsIter<const N: usize, L: LodVec<N>> {
    // internal stack for which coordinates are next
    stack: Vec<L>,

    // and maximum depth to go to
    max_depth: u8,

    // and the min of the bound
    bound_min: L,

    // and max of the bound
    bound_max: L,
}

impl<const N: usize, L: LodVec<N>> CoordsInBoundsIter<N, L> {
    /// Returns the amount of heap allocation to run this iterator.
    #[inline]
    pub fn stack_size(cv: L) -> usize {
        (cv.depth() as usize * L::MAX_CHILDREN) - (cv.depth() as usize - 1)
    }
}

impl<const N: usize, L: LodVec<N>> Iterator for CoordsInBoundsIter<N, L> {
    type Item = L;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let current = self.stack.pop()?;
        if current.depth() != self.max_depth {
            // go over all child nodes
            for i in 0..L::MAX_CHILDREN {
                let position = current.get_child(i);
                // if they are in bounds, add them to the stack
                if position.is_inside_bounds(self.bound_min, self.bound_max, self.max_depth) {
                    //We really do not want this to EVER allocate...
                    debug_assert_ne!(self.stack.capacity(), self.stack.len());
                    self.stack.push(position);
                }
            }
        }
        // and return this item from the stack
        Some(current)
    }
}

///Iterator over positions and indices of chunks in a given AABB.
pub struct ChunkIdxInAABBIter<'a, const N: usize, const B: usize, L>
where
    L: LodVec<N>,
{
    /// the reference to tree's nodes
    nodes: &'a NodeStorage<B>,

    /// internal stack for tree traverse
    to_visit: Vec<TreePos<N, L>>,

    /// index of child to return
    to_return: arrayvec::ArrayVec<TreePos<N, L>, B>,
    /// and maximum depth to go to
    max_depth: u8,

    /// the min of the bound box
    bound_min: L,

    /// max of the bound box
    bound_max: L,
}

impl<'a, const N: usize, const B: usize, L> ChunkIdxInAABBIter<'a, N, B, L>
where
    L: LodVec<N>,
{
    pub fn new(nodes: &'a NodeStorage<B>, bound_min: L, bound_max: L) -> Self {
        debug_assert_eq!(bound_min.depth(), bound_max.depth());

        // TODO: Smallvec?
        let mut to_visit = Vec::with_capacity(Self::stack_size(bound_min));

        to_visit.push(TreePos {
            idx: 0,
            pos: L::root(),
        });

        #[cfg(test)]
        unsafe {
            MAX_STACK = 1;
        }

        ChunkIdxInAABBIter {
            to_visit,
            to_return: arrayvec::ArrayVec::new(),
            nodes,
            max_depth: bound_min.depth(),
            bound_min,
            bound_max,
        }
    }
    #[inline]
    pub fn stack_size(cv: L) -> usize {
        (cv.depth().saturating_sub(1) as usize) * (L::MAX_CHILDREN - 1) + 1
    }
}

impl<'a, const N: usize, const B: usize, L> Iterator for ChunkIdxInAABBIter<'a, N, B, L>
where
    L: LodVec<N>,
{
    type Item = TreePos<N, L>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // if we have nothing in to_return stack, traverse the tree.
        while self.to_return.is_empty() {
            // try to traverse more nodes, if nothing to traverse we are done
            let current = self.to_visit.pop()?;
            // dbg!(current);
            //if we are allowed to dive deeper, do so
            if current.pos.depth() != self.max_depth {
                let cur_node = &self.nodes[current.idx];
                // go over all child nodes
                for i in 0..L::MAX_CHILDREN {
                    let child_position = current.pos.get_child(i);

                    if !child_position.is_inside_bounds(
                        self.bound_min,
                        self.bound_max,
                        self.max_depth,
                    ) {
                        //dbg!(self.bound_min,self.bound_max);
                        //println!("Tried child {i} pos {child_position:?} got OOB!");
                        continue;
                    }
                    //println!("Use child {i} pos {child_position:?}");
                    // if child is present at given location add it to the visit list
                    //dbg!(cur_node.children);
                    if let Some(child_idx) = cur_node.children[i] {
                        // make sure this push never allocates
                        debug_assert!(self.to_visit.capacity() > 0);
                        self.to_visit.push(TreePos {
                            pos: child_position,
                            idx: child_idx.get() as usize,
                        });
                        #[cfg(test)]
                        unsafe {
                            MAX_STACK = MAX_STACK.max(self.to_visit.len());
                        }
                    }
                    if let Some(chunk_idx) = cur_node.chunk[i].get() {
                        //println!("Return chunk {}",chunk_idx);
                        self.to_return.push(TreePos {
                            pos: child_position,
                            idx: chunk_idx,
                        });
                    }
                }
                //dbg!(&self.to_visit);
            }
        }
        match self.to_return.pop() {
            Some(rv) => Some(rv),
            // SAFETY: the only way to get here is by having stuf in the to_return vec
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }
}

duplicate::duplicate! {
    [
        StructName              reference(lt, type)     getter(p);
        [ChunksInAABBIter]      [& 'lt type]             [ &self.chunks[p.idx].chunk ];
        // SAFETY: we attach the lifetime of the iterator to this when returning so
        // nobody can destroy the tree when we are not looking.
        [ChunksInAABBIterMut]   [& 'lt mut type]         [unsafe{self.chunks[p.idx].chunk_ptr().as_mut().unwrap_unchecked()}];
    ]

    ///Iterator over positions and chunks in the AABB
    pub struct StructName<'a, const N:usize, const B:usize, C, L>
    where
    L:LodVec<N>,
    C:Sized,
    {
        // the chunks storage reference
        chunks: reference([a],[ChunkStorage<N,C,L>]),
        // iterator over indices in chunk storage
        chunk_idx_iter: ChunkIdxInAABBIter<'a, N,B,L>,
    }
    impl  <'a, const N:usize, const B:usize, C, L> Iterator for StructName<'a,N,B, C, L> where
    L:LodVec<N>,
    C:Sized,
    {
        type Item = (TreePos<N, L>, reference([a], [C]));

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            // fetch next position from position iterator
            let pos = self.chunk_idx_iter.next()?;
            // return appropriate reference to the chunk
            Some((pos, getter([pos])))
        }
    }

}

/// Iterate over all positions inside a certain AABB.
/// Important - this returns all intermediate depths, but does not return root node.
#[inline]
pub fn iter_all_positions_in_bounds<const N: usize, L: LodVec<N>>(
    bound_min: L,
    bound_max: L,
) -> CoordsInBoundsIter<N, L> {
    debug_assert!(
        bound_min < bound_max,
        "Bounds must select a non-empty region"
    );

    let mut stack = Vec::with_capacity(CoordsInBoundsIter::<N, L>::stack_size(bound_min));
    stack.push(L::root());
    let mut ite = CoordsInBoundsIter {
        stack,
        max_depth: bound_min.depth(),
        bound_min,
        bound_max,
    };
    //discard root node as we never want it.
    ite.next();
    ite
}

#[cfg(test)]
static mut MAX_STACK: usize = 0;

impl<'a, const N: usize, const B: usize, C, L> Tree<N, B, C, L>
where
    C: Sized,
    L: LodVec<N>,
    Self: 'a,
{
    /// Iterate over references to all chunks of the tree in the bounding box. Also returns chunk positions.
    #[inline(always)]
    pub fn iter_chunk_indices_in_aabb(
        &'a self,
        bound_min: L,
        bound_max: L,
    ) -> ChunkIdxInAABBIter<N, B, L> {
        ChunkIdxInAABBIter::new(&self.nodes, bound_min, bound_max)
    }

    /// Iterate over references to all chunks of the tree in the bounding box. Also returns chunk positions.
    #[inline(always)]
    pub fn iter_chunks_in_aabb(
        &'a self,
        bound_min: L,
        bound_max: L,
    ) -> ChunksInAABBIter<N, B, C, L> {
        ChunksInAABBIter {
            chunks: &self.chunks,
            chunk_idx_iter: ChunkIdxInAABBIter::new(&self.nodes, bound_min, bound_max),
        }
    }
    /// Iterate over mutable references to all chunks of the tree in the bounding box. Also returns chunk positions.
    #[inline(always)]
    pub fn iter_chunks_in_aabb_mut(
        &'a mut self,
        bound_min: L,
        bound_max: L,
    ) -> ChunksInAABBIterMut<N, B, C, L> {
        ChunksInAABBIterMut {
            chunks: &mut self.chunks,
            chunk_idx_iter: ChunkIdxInAABBIter::new(&self.nodes, bound_min, bound_max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    const NUM_QUERIES: usize = 1;

    struct Chunk {
        visible: bool,
    }

    ///Tests generation of coordinates in bounds over QuadTree
    #[test]
    fn bounds_quadtree() {
        let mut rng = SmallRng::seed_from_u64(42);

        for _i in 0..NUM_QUERIES {
            let depth = rng.gen_range(4u8..8u8);
            let cmax = 1 << depth;
            let min = rand_cv(
                &mut rng,
                QuadVec::new([0, 0], depth),
                QuadVec::new([cmax - 2, cmax - 2], depth),
            );
            let max = rand_cv(
                &mut rng,
                min + QuadVec::new([1, 1], depth),
                QuadVec::new([cmax - 1, cmax - 1], depth),
            );

            println!("Generated min  {:?}", min);
            println!("Generated max {:?}", max);

            let mut count = 0;
            for pos in iter_all_positions_in_bounds(min, max) {
                //println!("{:?}", pos);

                if pos.depth == depth {
                    count += 1;
                }
            }
            assert_eq!(count, get_chunk_count_at_max_depth(min, max));
        }
    }

    ///Tests generation of coordinates in bounds over OctTree
    #[test]
    fn bounds_octree() {
        let mut rng = SmallRng::seed_from_u64(42);

        for _i in 0..NUM_QUERIES {
            let depth = rng.gen_range(4u8..8u8);
            let cmax = 1 << depth;
            let min = rand_cv(
                &mut rng,
                OctVec::new([0, 0, 0], depth),
                OctVec::new([cmax - 2, cmax - 2, cmax - 2], depth),
            );
            let max = rand_cv(
                &mut rng,
                min + OctVec::new([1, 1, 1], depth),
                OctVec::new([cmax - 1, cmax - 1, cmax - 1], depth),
            );
            let mut count = 0;
            for pos in iter_all_positions_in_bounds(min, max) {
                //println!("{:?}", pos);

                if pos.depth == depth {
                    count += 1;
                }
            }
            assert_eq!(count, get_chunk_count_at_max_depth(min, max));
        }
    }
    #[test]
    fn aabb_iterator_stack_size() {
        println!("Testing OctTree");
        for _ in 0..50 {
            for d in 1..6 {
                println!("Depth {d}");
                let mut tree = OctTree::<Chunk, OctVec>::new();
                let cmax = (1u8 << d) - 1;
                let min = OctVec::new([0u8, 0, 0], d);
                let max = OctVec::new([cmax, cmax, cmax], d);
                let pos_iter = iter_all_positions_in_bounds(min, max);

                tree.insert_many(pos_iter, |_| Chunk { visible: false });
                for (_l, _c) in tree.iter_chunks_in_aabb(min, max) {}

                let expected_maxstack = ChunkIdxInAABBIter::<3, 8, OctVec>::stack_size(min);
                unsafe {
                    assert_eq!(MAX_STACK, expected_maxstack);
                    MAX_STACK = 0;
                }
            }

            println!("Testing QuadTree");
            for d in 1..10 {
                println!("Depth {d}");
                let mut tree = QuadTree::<Chunk, QuadVec<u16>>::new();
                let cmax = (1u16 << d) - 1;
                let min = QuadVec::new([0u16, 0], d);
                let max = QuadVec::new([cmax, cmax], d);
                let pos_iter = iter_all_positions_in_bounds(min, max);

                tree.insert_many(pos_iter, |_| Chunk { visible: false });
                for (_l, _c) in tree.iter_chunks_in_aabb(min, max) {}
                let expected_maxstack = ChunkIdxInAABBIter::<2, 4, QuadVec<u16>>::stack_size(min);
                unsafe {
                    assert_eq!(MAX_STACK, expected_maxstack);
                    MAX_STACK = 0;
                }
            }
        }
    }
    #[test]
    fn iterate_over_chunks_in_aabb() {
        const D: u8 = 4;
        const R: u8 = 3;

        let mut counter: usize = 0;
        let mut counter_created: usize = 0;
        let mut chunk_creator = |position: OctVec| -> Chunk {
            let r = (R * R) as i32 - 2;

            let visible = match position.depth {
                D => position
                    .pos
                    .iter()
                    .fold(true, |acc, e| acc & ((*e as i32 - R as i32).pow(2) < r)),
                _ => false,
            };
            counter += visible as usize;
            counter_created += 1;
            //println!("create {:?} {:?}", position, visible);
            Chunk { visible }
        };
        let cmax = 2u8 * R;
        let mut tree = OctTree::<Chunk, OctVec>::new();
        let pos_iter = iter_all_positions_in_bounds(
            OctVec::new([0u8, 0, 0], D),
            OctVec::new([cmax, cmax, cmax], D),
        )
        .filter(|p| p.depth == D);

        tree.insert_many(pos_iter, &mut chunk_creator);

        // query the whole region for filled voxels
        let mut filled_voxels: usize = 0;
        let mut total_voxels: usize = 0;
        let min = OctVec::new([0, 0, 0], D);
        let max = OctVec::new([cmax, cmax, cmax], D);
        for (l, c) in tree.iter_chunks_in_aabb(min, max) {
            filled_voxels += c.visible as usize;
            total_voxels += 1;
            //println!("visit {:?} {:?}", l, c.visible);
            assert_eq!(
                l.pos.depth, D,
                "All chunks must be at max depth (as we did not insert any others)"
            );
        }
        assert_eq!(
            filled_voxels, counter,
            " we should have found all voxels that were inserted and marked visible"
        );
        assert_eq!(
            total_voxels, counter_created,
            "we should have found all voxels that were inserted"
        );
        assert_eq!(
            total_voxels,
            get_chunk_count_at_max_depth(min, max),
            "we should have found all voxels in AABB"
        );

        // query a bunch of random regions
        let mut rng = SmallRng::seed_from_u64(42);
        for _ite in 0..NUM_QUERIES {
            let cmax = 2u8 * R;
            let min = rand_cv(
                &mut rng,
                OctVec::new([0, 0, 0], D),
                OctVec::new([cmax - 2, cmax - 2, cmax - 2], D),
            );
            let max = rand_cv(
                &mut rng,
                min + OctVec::new([1, 1, 1], D),
                OctVec::new([cmax, cmax, cmax], D),
            );
            // println!("Generated min  {:?}", min);
            // println!("Generated max {:?}", max);
            assert!(max > min);

            let mut filled_voxels: usize = 0;
            //println!("{:?}  {:?} {:?}", _ite, min, max);
            for (_l, c) in tree.iter_chunks_in_aabb(min, max) {
                if c.visible {
                    //println!(" Sphere chunk {:?}", _l);
                    filled_voxels += 1;
                }
            }
            //println!("  filled {:?}", filled_voxels);
            assert!(
                filled_voxels <= counter,
                "No way we see more voxels than were inserted!"
            )
        }
    }

    #[test]
    fn edit_chunks_in_aabb() {
        const D: u8 = 4;
        const R: u8 = 3;
        let mut counter: usize = 0;
        let mut chunk_creator = |position: OctVec| -> Chunk {
            let r = (R * R) as i32 - 2;

            let visible = match position.depth {
                D => position
                    .pos
                    .iter()
                    .fold(true, |acc, e| acc & ((*e as i32 - R as i32).pow(2) < r)),
                _ => false,
            };
            counter += visible as usize;
            println!("create {:?} {:?}", position, visible);
            Chunk { visible }
        };

        let mut tree = OctTree::<Chunk, OctVec>::new();
        let pos_iter = iter_all_positions_in_bounds(
            OctVec::build(0, 0, 0, D),
            OctVec::build(2 * R, 2 * R, 2 * R, D),
        )
        .filter(|p| p.depth == D);
        tree.insert_many(pos_iter, &mut chunk_creator);
        // query the whole region for filled voxels
        let cmax = 2u8 * R;
        let min = OctVec::build(0, 0, 0, D);
        let max = OctVec::new([cmax, cmax, cmax], D);
        let mut filled_voxels: usize = 0;
        for (l, c) in tree.iter_chunks_in_aabb_mut(min, max) {
            if c.visible {
                filled_voxels += 1;
                c.visible = false;
            }

            assert_eq!(
                l.pos.depth, D,
                "All chunks must be at max depth (as we did not insert any others)"
            );
        }

        assert_eq!(
            filled_voxels, counter,
            " we should have found all voxels that were inserted and marked visible"
        );

        let mut rng = SmallRng::seed_from_u64(42);

        for _ite in 0..NUM_QUERIES {
            let cmax = 2u8 * R;
            let min = rand_cv(
                &mut rng,
                OctVec::new([0, 0, 0], D),
                OctVec::new([cmax - 2, cmax - 2, cmax - 2], D),
            );
            let max = rand_cv(
                &mut rng,
                min + OctVec::new([1, 1, 1], D),
                OctVec::new([cmax, cmax, cmax], D),
            );

            for (l, c) in tree.iter_chunks_in_aabb_mut(min, max) {
                assert_eq!(c.visible, false, "no way any voxel is still visible");
                assert_eq!(
                    l.pos.depth, D,
                    "All chunks must be at max depth (as we did not insert any others)"
                );
            }
        }
    }
}
