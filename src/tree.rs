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

//! Contains the tree struct, which is used to hold all chunks

use crate::coords::*;
use crate::util_funcs::*;
use slab::Slab;
use std::fmt::Debug;
use std::num::NonZeroU32;
use std::ops::ControlFlow;

// type aliases to make iterators more readable
pub(crate) type ChunkStorage<const N: usize, C, L> = Slab<ChunkContainer<N, C, L>>;
pub(crate) type NodeStorage<const B: usize> = Slab<TreeNode<B>>;

/// Tree holding the actual data permanently in memory.
/// This is arguably "too generic", and one should use provided OctTree and QuadTree types when possible.
///
///
/// Template parameters are:
/// * N is the number bits needed to encode B, i.e. 3 for octrees and 4 for quadtrees.
/// * B is the branch count per node, i.e. 8 for octrees and 4 for quadtrees. Has to be power of two.
///
/// An invariant between the two has to be ensured where 1<<N == B, else tree will fail to construct.
/// This will look nicer once const generics fully stabilize.
#[derive(Clone, Debug)]
pub struct Tree<const N: usize, const B: usize, C: Sized, L: LodVec<N>> {
    /// All data chunks in the tree
    pub(crate) chunks: ChunkStorage<N, C, L>,
    /// All nodes of the Tree
    pub(crate) nodes: NodeStorage<B>,
    /// Temporary buffer for nodes used during rebuilds
    new_nodes: Slab<TreeNode<B>>,
}

pub enum Entry<'a, C: Sized> {
    Occupied(&'a mut C),
    Vacant(&'a mut C),
}

impl<const N: usize, const B: usize, C, L> Tree<N, B, C, L>
where
    C: Sized,
    L: LodVec<N>,
{
    /// create a tree with preallocated memory for chunks and nodes
    /// NOTE this function does runtime asserts to make sure Tree is templated correctly.
    /// this runtime check will become compiletime check once generic_const_exprs matures.
    pub fn with_capacity_unsafe(nodes_capacity: usize, chunks_capacity: usize) -> Self {
        // TODO: once feature(generic_const_exprs) is mature, make this look nicer
        assert_eq!(1<<N, B, "Relation between N and B is not met, once feature(generic_const_exprs) is mature this will be statically checked");

        debug_assert!(nodes_capacity >= 1);
        debug_assert!(chunks_capacity >= 1);
        let mut nodes = Slab::with_capacity(nodes_capacity);
        // create root node right away, no point having a tree without a root.
        let r = nodes.insert(TreeNode::new());
        //Slab always returns 0 for index of first inserted element, right? Better check...
        debug_assert_eq!(r, 0);

        Self {
            chunks: Slab::with_capacity(chunks_capacity),
            nodes,
            new_nodes: Slab::new(),
        }
    }
    //TODO: use duplicate! on this

    /// Gets the node "controlling" the desired position. This means node that is one depth level above target.
    /// Returns the index of child entry and mutable reference to the node.
    /// If exact match is found, returns Ok variant, else Err variant with nearest match (at lower depth).
    fn follow_nodes_to_position_mut(
        &mut self,
        position: L,
    ) -> Result<(usize, &mut TreeNode<B>), (usize, &mut TreeNode<B>)> {
        // start in root
        let mut addr = TreePos {
            idx: 0,
            pos: L::root(),
        };
        // make sure target is not root (else we will be stuck here)
        debug_assert_ne!(position, addr.pos);
        // then loop
        loop {
            // SAFETY: the node hierarchy should be sound. If it is not we are doomed.
            let current = unsafe {
                (self.nodes.get_unchecked_mut(addr.idx) as *mut TreeNode<B>)
                    .as_mut()
                    .unwrap_unchecked()
            };
            // compute child index & position towards target
            let child_idx = addr.pos.get_child_index(position);
            let child_pos = addr.pos.get_child(child_idx);
            // if the current node is the one we are looking for, return it
            if child_pos == position {
                return Ok((child_idx, current));
            }

            // assuming the child index points to an existing child node, follow it.
            addr = match current.children[child_idx] {
                Some(idx) => TreePos {
                    idx: idx.get() as usize,
                    pos: addr.pos.get_child(child_idx),
                },
                None => {
                    return Err((child_idx, current));
                }
            };
        }
    }

    /// Gets the node "controlling" the desired position. This means node that is one depth level above target.
    /// Returns the index of child entry and mutable reference to the node.
    /// If exact match is found, returns Ok variant, else Err variant with nearest match (at lower depth).
    fn follow_nodes_to_position(
        &self,
        position: L,
    ) -> Result<(usize, &TreeNode<B>), (usize, &TreeNode<B>)> {
        // start in root
        let mut addr = TreePos {
            idx: 0,
            pos: L::root(),
        };
        // make sure target is not root (else we will be stuck here)
        debug_assert_ne!(position, addr.pos);

        // then loop
        loop {
            // SAFETY: the node hierarchy should be sound. If it is not we are doomed.
            let current = unsafe { self.nodes.get_unchecked(addr.idx) };
            // compute child index & position towards target
            let child_idx = addr.pos.get_child_index(position);
            let child_pos = addr.pos.get_child(child_idx);
            // if the current node is the one we are looking for, return it
            if child_pos == position {
                return Ok((child_idx, current));
            }

            // assuming the child index points to an existing child node, follow it.
            addr = match current.children[child_idx] {
                Some(idx) => TreePos {
                    idx: idx.get() as usize,
                    pos: addr.pos.get_child(child_idx),
                },
                None => {
                    return Err((child_idx, current));
                }
            };
        }
    }

    /// get the number of chunks in the tree
    #[inline]
    pub fn get_num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// get a reference to chunk by index
    #[inline]
    pub fn get_chunk(&self, index: usize) -> &ChunkContainer<N, C, L> {
        &self.chunks[index]
    }

    /// get a mutable reference to chunk container by index
    #[inline]
    pub fn get_chunk_mut(&mut self, index: usize) -> &mut ChunkContainer<N, C, L> {
        &mut self.chunks[index]
    }

    /// get a chunk by position if it's in the tree
    #[inline]
    pub fn get_chunk_by_position(&self, position: L) -> Option<&C> {
        // get the index of the chunk
        let (idx, node) = self.follow_nodes_to_position(position).ok()?;
        let chunk_index = node.chunk[idx].get()?;
        // and return the chunk
        Some(&self.chunks[chunk_index].chunk)
    }

    /// get a mutable chunk by position if it's in the tree
    #[inline]
    pub fn get_chunk_by_position_mut(&mut self, position: L) -> Option<&mut C> {
        // get the index of the chunk
        let (idx, node) = self.follow_nodes_to_position(position).ok()?;
        let chunk_index = node.chunk[idx].get()?;
        // and return the chunk
        Some(&mut self.chunks[chunk_index].chunk)
    }

    /// get an Entry handle to modify existing or insert a new chunk.
    /// this is WIP
    #[inline]
    pub fn entry<V>(&mut self, _position: L, mut _chunk_creator: V) -> Entry<C>
    where
        V: FnMut(L) -> C,
    {
        todo!()
        /*match self.follow_nodes_to_position(position){
            Ok((idx, node))=>{(idx, node)}
            Err((idx, node))=>{todo!()}
        }

        let chunk_index = node.chunk[idx].get()?;
        // and return the chunk
        Some(&mut self.chunks[chunk_index].chunk)*/
    }

    /// get the position of a chunk, if it exists
    #[inline]
    pub fn get_chunk_position(&self, index: usize) -> Option<L> {
        Some(self.chunks.get(index)?.position)
    }

    /// Inserts/replaces chunks at specified locations.
    /// This operation will create necessary intermediate nodes to meet datastructure
    /// constraints.
    /// This operation may allocate memory more than once if targets is long enough
    ///
    /// If targets are nearby, we could save quite a bit of resources by
    /// reducing walking needed to insert data.
    pub fn insert_many<T, V>(&mut self, mut targets: T, mut chunk_creator: V)
    where
        T: Iterator<Item = L>,
        V: FnMut(L) -> C,
    {
        debug_assert!(!self.nodes.is_empty());

        // Internal queue for batch processing, it will be as long as maximal depth of the tree.
        // Stack allocated since it is only ~200 bytes, and this keeps things cache-friendly.
        let mut queue = arrayvec::ArrayVec::<TreePos<N, L>, { MAX_DEPTH as usize }>::new();

        // start at the root node
        let mut addr = TreePos {
            pos: L::root(),
            idx: 0,
        };

        let mut tgt = targets.next().expect("Expected at least one target");
        loop {
            debug_assert_ne!(tgt, L::root(), "Root node is not a valid target!");
            //println!("===Inserting target {tgt:?}===");

            loop {
                addr = match self.insert_inner(addr, tgt, &mut chunk_creator) {
                    ControlFlow::Break(_) => {
                        break;
                    }
                    ControlFlow::Continue(a) => {
                        queue.push(addr);
                        a
                    }
                };
            }
            // insert done, fetch us the next target
            tgt = match targets.next() {
                Some(t) => t,
                None => {
                    break;
                }
            };
            debug_assert_ne!(tgt, L::root(), "Root node is not a valid target!");
            // walk up the tree until we are high enough to work on next target
            //dbg!(&self.processing_queue);
            while !addr.pos.contains_child_node(tgt) {
                addr = queue
                    .pop()
                    .expect("This should not happen, as we keep root in the stack");
                //println!("Going up the stack to {addr:?} for target {tgt:?}");
            }
        }
    }

    /// Removes chunk at specified position, and returns its content (if any)
    #[inline]
    pub fn pop_chunk_by_position(&mut self, pos: L) -> Option<C> {
        let (child, node) = self.follow_nodes_to_position_mut(pos).ok()?;
        let chunk_idx = node.chunk[child].take()?;

        let chunk_rec = self.chunks.remove(chunk_idx);
        Some(chunk_rec.chunk)
    }

    // Common part of various insert operations
    #[inline]
    fn insert_inner<V>(
        &mut self,
        addr: TreePos<N, L>,
        tgt: L,
        chunk_creator: &mut V,
    ) -> ControlFlow<usize, TreePos<N, L>>
    where
        V: FnMut(L) -> C,
    {
        //dbg!(addr, tgt);
        let child_idx = addr.pos.get_child_index(tgt);
        let child_pos = addr.pos.get_child(child_idx);
        let current_node = self.nodes.get_mut(addr.idx).expect("Node index broken!");
        //println!("Current node {addr:?}");
        if child_pos == tgt {
            //println!("Found child {child_pos:?}, id {child_idx:?}");
            let chunk = chunk_creator(tgt);
            //perform actual insertion at this location
            let inserted = match current_node.chunk[child_idx].get() {
                Some(ci) => {
                    self.chunks[ci].chunk = chunk;
                    //println!("Found target, replacing existing chunk at {}", ci.get());
                    ci
                }
                None => {
                    let chunk_idx = self.chunks.insert(ChunkContainer {
                        chunk,
                        position: child_pos,
                        node_idx: addr.idx as u32,
                        child_idx: child_idx as u8,
                    });
                    //println!("Found target, inserting chunk at new index {chunk_idx}");
                    current_node.chunk[child_idx] = ChunkPtr::from(Some(chunk_idx));
                    chunk_idx
                }
            };
            return ControlFlow::Break(inserted);
        }

        let idx = match current_node.children[child_idx] {
            Some(idx) => idx.get() as usize,
            None => {
                //println!("Inserting new node");
                // modify nodes slab
                let idx = self.nodes.insert(TreeNode::new());
                // update pointer in parent node
                self.nodes[addr.idx].children[child_idx] = NonZeroU32::new(idx as u32);
                idx
            }
        };
        ControlFlow::Continue(TreePos {
            idx,
            pos: child_pos,
        })
    }

    /// Inserts/replaces a single chunk at specified location.
    /// This operation will create necessary intermediate nodes to meet datastructure
    /// constraints.
    /// This operation may allocate memory.
    ///
    /// If you need to insert lots of chunks, use insert_many instead, it will probably be faster on deep trees.
    /// returns index of inserted chunk.
    pub fn insert<V>(&mut self, tgt: L, mut chunk_creator: V) -> usize
    where
        V: FnMut(L) -> C,
    {
        debug_assert!(!self.nodes.is_empty());
        debug_assert_ne!(tgt, L::root(), "Root node is not a valid target!");

        // start at the root node
        let mut addr = TreePos {
            pos: L::root(),
            idx: 0,
        };

        //println!("===Inserting target {tgt:?}===");
        loop {
            addr = match self.insert_inner(addr, tgt, &mut chunk_creator) {
                ControlFlow::Continue(a) => a,
                ControlFlow::Break(idx) => {
                    break idx;
                }
            };
        }
    }

    #[inline]
    pub fn iter_chunks_mut(&mut self) -> slab::IterMut<ChunkContainer<N, C, L>> {
        self.chunks.iter_mut()
    }

    #[inline]
    pub fn iter_chunks(&mut self) -> slab::Iter<ChunkContainer<N, C, L>> {
        self.chunks.iter()
    }

    /// clears the tree, removing all nodes, chunks and internal buffers
    #[inline]
    pub fn clear(&mut self) {
        self.chunks.clear();
        let root = self.nodes.remove(0);
        self.nodes.clear();
        self.nodes.insert(root);
    }

    /// Defragments the chunks array to enable fast iteration.
    /// This will have zero cost if array does not need defragmentation.
    /// The next update might take longer due to memory allocations.
    #[inline]
    pub fn defragment_chunks(&mut self) {
        let nodes = &mut self.nodes;
        self.chunks.compact(|chunk, cur, new| {
            assert_eq!(
                nodes[chunk.node_idx as usize].chunk[chunk.child_idx as usize]
                    .get()
                    .unwrap(),
                cur
            );
            nodes[chunk.node_idx as usize].chunk[chunk.child_idx as usize] =
                ChunkPtr::from(Some(new));
            true
        });
    }

    /// Prunes the nodes array to delete all nodes that have no chunks.
    /// This requires nodes to be traversed in a depth-first manner, so this is somewhat slow on larger trees
    /// You only really need this if you have deleted a whole bunch of chunks and really need the nodes memory back
    pub fn prune_nodes(&mut self) {
        todo!()
    }

    /// Defragments the nodes array to enable faster operation and prune dead leaves.
    /// This requires nodes to be copied, so this will allocate. Many unsafes would be needed otherwise.
    /// The next update might take longer due to memory allocations.
    pub fn defragment_nodes(&mut self) {
        let num_nodes = self.nodes.len();
        // allocate for all nodes since we know their exact number
        self.new_nodes.reserve(num_nodes);

        // move the root node to kick things off
        self.new_nodes.insert(self.nodes.remove(0));

        // for every index in new slab, move its children immediately after itself, keep doing that until all are moved.
        // this will produce a breadth-first traverse of original nodes laid out in new memory, which should keep
        // nearby nodes close in memory locations.
        for n in 0..num_nodes {
            // clone children array to keep it safe while we mess with it
            let children = self.new_nodes[n].children;
            // now go over node's children and move them over
            for (i, old_idx) in iter_treenode_children(&children) {
                let old_node = self.nodes.remove(old_idx);
                //eliminate empty nodes
                if old_node.is_empty() {
                    self.new_nodes[n].children[i] = None;
                    continue;
                }
                // move the child into new slab
                let new_idx = self.new_nodes.insert(old_node);
                // ensure slab is not doing anything fishy, and actually gives us correct indices
                debug_assert_eq!(new_idx, n + i + 1);
                // fix our reference to that child
                self.new_nodes[n].children[i] = Some(NonZeroU32::new(new_idx as u32).unwrap());

                // if the position of the node has changed
                if new_idx != old_idx {
                    // update node index of all chunks we are referring to
                    for (_, chunk_idx) in self.new_nodes[new_idx].iter_existing_chunks() {
                        self.chunks[chunk_idx].node_idx = new_idx as u32;
                    }
                }
            }
        }
        std::mem::swap(&mut self.nodes, &mut self.new_nodes);
        self.new_nodes.clear();
    }

    /// Prepares the tree for an LOD update. This operation reorganizes the nodes and
    /// adds chunks around specified locations (targets) while also erasing all other chunks.
    /// A side-effect of this is that nodes are defragmented.
    /// # Params
    /// * `targets` the target positions to generate the maximal level of detail around, e.g. players
    /// * `detail` the size of the region which will be filled with max level of detail
    /// * `chunk_creator` function to create a new chunk from a given position
    /// * `evict_callback` function to dispose of unneeded chunks (can move them into cache or whatever)
    pub fn lod_update<V, W>(
        &mut self,
        targets: &[L],
        detail: u32,
        mut chunk_creator: V,
        mut evict_callback: W,
    ) where
        V: FnMut(L) -> C,
        W: FnMut(L, C),
    {
        let num_nodes = self.nodes.len();
        // allocate room for new nodes (assuming it is about same amount as before update)
        // better to overallocate here than to allocate twice.
        //let mut new_nodes = Slab::with_capacity(num_nodes);
        self.new_nodes.reserve(num_nodes);
        let mut new_positions = std::collections::VecDeque::with_capacity(B);

        // move the root node to kick things off
        self.new_nodes.insert(self.nodes.remove(0));
        new_positions.push_back(L::root());

        // for every index in new slab, move its children immediately after itself, keep doing that until all are moved.
        // this will produce a breadth-first traverse of original nodes laid out in new memory, which should keep
        // nearby nodes close in memory locations.
        for n in 0..usize::MAX {
            let pos = match new_positions.pop_front() {
                Some(p) => p,
                None => break,
            };
            // copy children array to keep it safe while we mess with it
            let children = self.new_nodes[n].children;

            // now go over node's children
            for (b, maybe_child) in children.iter().enumerate() {
                // figure out position of child node
                let child_pos = pos.get_child(b);
                // figure if any of the targets needs it subdivided
                let subdivide = targets.iter().any(|x| x.can_subdivide(child_pos, detail));
                //println!("{child_pos:?}, {subdivide:?}");

                // if child is subdivided we do not want a chunk there,
                // and in other case we need one, so we make one if necessary
                match (self.new_nodes[n].chunk[b].get(), subdivide) {
                    (None, true) => {
                        //println!("No chunks present");
                    }
                    (Some(chunk_idx), true) => {
                        let cont = self.chunks.remove(chunk_idx);
                        debug_assert_eq!(cont.position, child_pos);
                        evict_callback(child_pos, cont.chunk);
                        self.new_nodes[n].chunk[b] = ChunkPtr::None;
                    }
                    (None, false) => {
                        let chunk_idx = self.chunks.insert(ChunkContainer {
                            chunk: chunk_creator(child_pos),
                            position: child_pos,
                            node_idx: n as u32,
                            child_idx: b as u8,
                        });

                        self.new_nodes[n].chunk[b] = ChunkPtr::from(Some(chunk_idx));
                    }
                    (Some(chunk_idx), false) => {
                        //println!("Preserve chunk at index {chunk_idx}");
                        self.chunks[chunk_idx].node_idx = n as u32;
                    }
                }

                // make sure a child is present if we are going to subdivide
                match (maybe_child, subdivide) {
                    // no node present but we need one
                    (None, true) => {
                        let new_idx = self.new_nodes.insert(TreeNode::new());
                        // keep track of positions
                        new_positions.push_back(child_pos);
                        self.new_nodes[n].children[b] =
                            Some(NonZeroU32::new(new_idx as u32).unwrap());
                    }
                    // no node present and we do not need one
                    (None, false) => {}
                    //existing node needed, keep it (same logic as in defragment_nodes)
                    (Some(child_idx), true) => {
                        // move the child into new slab
                        let new_idx = self
                            .new_nodes
                            .insert(self.nodes.remove(child_idx.get() as usize));
                        // keep track of positions
                        new_positions.push_back(child_pos);
                        // fix our reference to that child
                        self.new_nodes[n].children[b] =
                            Some(NonZeroU32::new(new_idx as u32).unwrap());
                    }
                    // existing node not needed, delete its chunks and do not copy over the node itself
                    (Some(child_idx), false) => {
                        for node in traverse(&self.nodes, &self.nodes[child_idx.get() as usize]) {
                            for (_, cid) in node.iter_existing_chunks() {
                                let cont = self.chunks.remove(cid);
                                debug_assert!(child_pos.contains_child_node(cont.position));
                                evict_callback(cont.position, cont.chunk);
                            }
                        }
                        self.new_nodes[n].children[b] = None;
                    }
                };
            }
        }
        std::mem::swap(&mut self.nodes, &mut self.new_nodes);
        self.new_nodes.clear();
    }

    /// Shrinks all internal buffers to fit actual need, reducing memory usage.
    /// Calling defragment_chunks() and defragment_nodes() before this is advisable.
    /// The next update might take longer due to memory allocations.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.chunks.shrink_to_fit();
        self.nodes.shrink_to_fit();
        self.new_nodes.shrink_to_fit();
    }
}

/// Construct an itreator that traverses a subtree in nodes that begins in start (including start itself).
#[inline]
pub fn traverse<'a, const B: usize>(
    nodes: &'a NodeStorage<B>,
    start: &'a TreeNode<B>,
) -> TraverseIter<'a, B> {
    //TODO use better logic here (DFS)!
    let mut to_visit = Vec::with_capacity(8 * B); //arrayvec::ArrayVec::new();
    to_visit.push(start);
    TraverseIter { nodes, to_visit }
}

///Helper to perform breadth-first traverse of tree's nodes.
pub struct TraverseIter<'a, const B: usize> {
    nodes: &'a NodeStorage<B>,
    to_visit: Vec<&'a TreeNode<B>>,
    //to_visit: arrayvec::ArrayVec<&'a TreeNode<B>, B>,
}

impl<'a, const B: usize> Iterator for TraverseIter<'a, B> {
    type Item = &'a TreeNode<B>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let current = self.to_visit.pop()?;
        for (_, c) in iter_treenode_children(&current.children) {
            self.to_visit.push(&self.nodes[c]);
        }
        Some(current)
    }
}

pub type OctTree<C, L> = Tree<3, 8, C, L>;
impl<C, L> OctTree<C, L>
where
    C: Sized,
    L: LodVec<3>,
{
    /// creates a new, empty OctTree, with no cache
    pub fn new() -> Self {
        Self::with_capacity(1, 1)
    }
    /// Creates a OctTree with given capacity for nodes and chunks. Capacities should be > 1 both.
    pub fn with_capacity(nodes_capacity: usize, chunks_capacity: usize) -> Self {
        Tree::with_capacity_unsafe(nodes_capacity, chunks_capacity)
    }
}

pub type QuadTree<C, L> = Tree<2, 4, C, L>;
impl<C, L> QuadTree<C, L>
where
    C: Sized,
    L: LodVec<2>,
{
    /// creates a new, empty QuadTree, with no cache
    pub fn new() -> Self {
        Self::with_capacity(1, 1)
    }
    /// Creates a QuadTree with given capacity for nodes and chunks. Capacities should be > 1 both.
    pub fn with_capacity(nodes_capacity: usize, chunks_capacity: usize) -> Self {
        Tree::with_capacity_unsafe(nodes_capacity, chunks_capacity)
    }
}

impl<C, L> Default for OctTree<C, L>
where
    C: Sized,
    L: LodVec<3>,
{
    /// creates a new, empty tree
    fn default() -> Self {
        Self::new()
    }
}

impl<C, L> Default for QuadTree<C, L>
where
    C: Sized,
    L: LodVec<2>,
{
    /// creates a new, empty tree
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    struct TestChunk;

    #[test]
    fn lod_update() {
        // make a tree
        let mut tree = QuadTree::<TestChunk, QuadVec>::new();
        // as long as we need to update, do so
        //let targets = [QuadVec::build(1, 1, 2), QuadVec::build(2, 3, 2)];
        let targets = [QuadVec::build((1 << 2) - 1, (1 << 2) - 1, 2)];
        println!("=====> Update with targets {targets:?}");
        tree.lod_update(
            &targets,
            0,
            |p| {
                println!("Creating chunk at {p:?}");
                TestChunk {}
            },
            |p, _| {
                println!("Evicting chunk at {p:?}");
            },
        );

        let targets = [QuadVec::build(0, 0, 2)];
        println!("=====> Update with targets {targets:?}");
        tree.lod_update(
            &targets,
            0,
            |p| {
                println!("Creating chunk at {p:?}");
                TestChunk {}
            },
            |p, _| {
                println!("Evicting chunk at {p:?}");
            },
        );
    }
    #[test]
    fn insert_into_tree() {
        // make a tree
        let mut tree = QuadTree::<TestChunk, QuadVec>::new();
        // as long as we need to update, do so
        let targets = [
            QuadVec::build(1u8, 1u8, 1u8),
            QuadVec::build(2u8, 3u8, 2u8),
            QuadVec::build(2u8, 2u8, 2u8),
            QuadVec::build(0u8, 1u8, 1u8),
        ];

        tree.insert_many(targets.iter().copied(), |_| TestChunk {});

        let mut tree2 = QuadTree::<TestChunk, QuadVec>::new();
        for t in targets {
            tree2.insert(t, |_| TestChunk {});
        }
    }

    #[test]
    pub fn defragment() {
        let mut tree = QuadTree::<TestChunk, QuadVec>::new();
        let targets = [
            QuadVec::build(0u8, 0u8, 2u8),
            QuadVec::build(0u8, 1u8, 2u8),
            QuadVec::build(3u8, 3u8, 2u8),
            QuadVec::build(2u8, 2u8, 2u8),
        ];
        tree.insert_many(targets.iter().copied(), |_| TestChunk {});
        // slab is a Vec, so it should have binary exponential growth. With 4 entires it should have capacity = len.
        assert_eq!(tree.chunks.capacity(), tree.chunks.len());
        tree.pop_chunk_by_position(targets[1]);
        assert_eq!(tree.chunks.capacity(), 4);
        assert_eq!(tree.chunks.len(), 3);
        tree.defragment_chunks();
        tree.shrink_to_fit();
        assert_eq!(tree.chunks.capacity(), tree.chunks.len());
        dbg!(tree.nodes.len());
        tree.pop_chunk_by_position(targets[0]);
        //TODO!
        //tree.prune();
        dbg!(tree.nodes.len());
    }

    #[test]
    pub fn alignment() {
        assert_eq!(
            std::mem::size_of::<TreeNode<8>>(),
            64,
            "Octree node should be 64 bytes"
        );
        assert_eq!(
            std::mem::size_of::<TreeNode<4>>(),
            32,
            "Quadtree node should be 32 bytes"
        );
    }
}
