//! Contains the tree struct, which is used to hold all chunks

use crate::coords::*;
use slab::Slab;
use std::fmt::Debug;
use std::num::NonZeroU32;
use std::ops::ControlFlow;

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
            Some(v) => Self { 0: v as i32 },
            None => Self::None,
        }
    }
    // Cheat for "compatibility" with option.
    #[allow(non_upper_case_globals)]
    const None: Self = ChunkPtr { 0: -1 };
}

//TODO: impl Try once stable
/*impl std::ops::Try for ChunkPtr{
    type Output = usize;

    type Residual;

    fn from_output(output: Self::Output) -> Self {
        todo!()
    }

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        todo!()
    }
}*/

//TODO - use struct of arrays?
/// Tree node that encompasses multiple children at once. This just barely fits into one cache line for octree.
/// For each possible child, the node has two relative pointers:
///  - children[i] will point to the TreeNode in a given branch direction
///  - chunk[i] will point to the data chunk.
/// both pointers may be "None", indicating either no children, or no data
#[derive(Clone, Debug)]
pub struct TreeNode<const N: usize> {
    /// children, these can't be the root (index 0), so we can use Some and Nonzero for slightly more compact memory
    pub children: [NodePtr; N],

    /// where the chunks for particular children is stored (if any)
    pub chunk: [ChunkPtr; N],
}

impl<const N: usize> TreeNode<N> {
    fn new() -> Self {
        Self {
            children: [NodePtr::None; N],
            chunk: [ChunkPtr::None; N],
        }
    }

    pub fn iter_existing_chunks<'a>(
        &'a self,
    ) -> impl core::iter::Iterator<Item = (usize, usize)> + 'a {
        self.chunk.iter().filter_map(|c| c.get()).enumerate()
    }
}

pub fn iter_treenode_children<'a, const N: usize>(
    children: &'a [NodePtr; N],
) -> impl core::iter::Iterator<Item = (usize, usize)> + 'a {
    children
        .iter()
        .filter_map(|c| Some((*c)?.get() as usize))
        .enumerate()
}

// utility struct for holding actual chunks and the node that owns them
#[derive(Clone, Debug)]
pub struct ChunkContainer<const N: usize, C: Sized, L: LodVec<N>> {
    pub chunk: C,      // actual data inside the chunk
    pub position: L,   // where the chunk is (as this can not be easily recovered from node tree)
    pub node_idx: u32, // index of the node that holds this chunk
    pub child_idx: u8, // index of the child in the node
}

impl<const N: usize, C: Sized, L: LodVec<N>> ChunkContainer<N, C, L> {
    /// get an mutable pointer to chunk which is not tied to the lifetime of the container
    /// this is only needed for iterators.
    #[inline]
    pub(crate) fn chunk_ptr(&mut self) -> *mut C {
        &mut self.chunk as *mut C
    }
}

/// utility struct for holding locations in the tree.
#[derive(Clone, Debug, Copy)]
pub struct TreePos<const N: usize, L: LodVec<N>> {
    pub idx: usize, // node index
    pub pos: L,     // and it's position
}

/// Tree holding the actual data permanently in memory.
/// This is "too generic", and one should use provided OctTree and QuadTree types when possible.
///
/// Internals are partially based on:
///  * https://stackoverflow.com/questions/41946007/efficient-and-well-explained-implementation-of-a-quadtree-for-2d-collision-det
///  * https://github.com/Dimev/lodtree
///
/// Template parameters are:
/// * N is the number bits needed to encode B, i.e. 3 for octrees and 4 for quadtrees.
/// * B is the branch count per node, i.e. 8 for octrees and 4 for quadtrees. Has to be power of two.
/// An invariant between the two has to be ensured where 1<<N == B, else tree will fail to construct.
/// This will look nicer once const generics fully stabilize.
#[derive(Clone, Debug)]
pub struct Tree<const N: usize, const B: usize, C: Sized, L: LodVec<N>> {
    /// All data chunks in the tree
    pub(crate) chunks: Slab<ChunkContainer<N, C, L>>,
    /// All nodes of the Tree
    pub(crate) nodes: Slab<TreeNode<B>>,

    /// Internal queue for batch processing, it will be as long as maximal depth of the tree.
    /// Single inserts and erases do not operate with this.
    processing_queue: Vec<TreePos<N, L>>,
}

impl<const N: usize, const B: usize, C, L> Tree<N, B, C, L>
where
    C: Sized,
    L: LodVec<N>,
{
    //TODO: use duplicate! on this
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
            processing_queue: Vec::with_capacity(4),
        }
    }

    /// Gets the node "controlling" the desired position. This means node that is one depth level above target.
    /// Returns the index of child entry and mutable reference to the node.
    /// If exact match is found, returns Ok variant, else Err variant with nearest match (at lower depth).
    fn follow_nodes_to_position_mut<'a>(
        &'a mut self,
        position: L,
    ) -> Result<(usize, &'a mut TreeNode<B>), (usize, &'a mut TreeNode<B>)> {
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
                    .unwrap()
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
    fn follow_nodes_to_position<'a>(
        &'a self,
        position: L,
    ) -> Result<(usize, &'a TreeNode<B>), (usize, &'a TreeNode<B>)> {
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

    /// get a chunk by index
    #[inline]
    pub fn get_chunk(&self, index: usize) -> &C {
        &self.chunks[index].chunk
    }

    /// get a chunk as mutable
    #[inline]
    pub fn get_chunk_mut(&mut self, index: usize) -> &mut C {
        &mut self.chunks[index].chunk
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
    pub fn get_chunk_from_position_mut(&mut self, position: L) -> Option<&mut C> {
        // get the index of the chunk
        let (idx, node) = self.follow_nodes_to_position(position).ok()?;
        let chunk_index = node.chunk[idx].get()?;
        // and return the chunk
        Some(&mut self.chunks[chunk_index].chunk)
    }

    /// get the position of a chunk, if it exists
    #[inline]
    pub fn get_chunk_position(&self, index: usize) -> Option<L> {
        Some(self.chunks.get(index)?.position)
    }

    /// Inserts/replaces chunks at specified locations.
    /// This operation will create necessary intermediate nodes to meet datastructure
    /// constraints.
    /// This operation may allocate memory.
    ///
    /// If targets are nearby, we could save quite a bit of resources by
    /// reducing walking needed to insert data.
    /// Insert many does not
    pub fn insert_many<T, V>(&mut self, mut targets: T, mut chunk_creator: V)
    where
        T: Iterator<Item = L>,
        V: FnMut(L) -> C,
    {
        debug_assert!(self.nodes.len() > 0);
        // clear the processing queue from any previous updates
        self.processing_queue.clear();

        // start at the root node
        let mut addr = TreePos {
            pos: L::root(),
            idx: 0,
        };

        let mut tgt = targets.next().expect("Expected at least one target");
        loop {
            //println!("===Inserting target {tgt:?}===");

            loop {
                addr = match self.insert_inner(addr, tgt, &mut chunk_creator) {
                    ControlFlow::Break(_) => {
                        break;
                    }
                    ControlFlow::Continue(a) => {
                        self.processing_queue.push(addr);
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
            // walk up the tree until we are high enough to work on next target
            //dbg!(&self.processing_queue);
            while !addr.pos.contains_child_node(tgt) {
                addr = self
                    .processing_queue
                    .pop()
                    .expect("This should not happen, as we keep root in the stack");
                //println!("Going up the stack to {addr:?} for target {tgt:?}");
            }
        }
    }

    pub fn pop_chunk_by_position(&mut self, pos: L) -> Option<C> {
        let (child, node) = self.follow_nodes_to_position_mut(pos).ok()?;
        let chunk_idx = node.chunk[child].take()?;
        let chunk_rec = self.chunks.remove(chunk_idx);
        Some(chunk_rec.chunk)
    }

    // Common part of various insert operations
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
        let mut current_node = self.nodes.get_mut(addr.idx).expect("Node index broken!");
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
                // drop reference to current node to keep borrow checker happy
                drop(current_node);
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
        debug_assert!(self.nodes.len() > 0);

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

    pub fn iter_chunks_mut(&mut self) -> slab::IterMut<ChunkContainer<N, C, L>> {
        self.chunks.iter_mut()
    }

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
        self.processing_queue.clear();
    }

    /// Defragments the chunks array to enable fast iteration.
    /// This will have zero cost if array does not need defragmentation.
    /// The next update might take longer due to memory allocations.
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

    /// Defragments the nodes array to enable faster operation.
    /// This requires nodes to be copied, so this will allocate. Many unsafes would be needed otherwise.
    /// The next update might take longer due to memory allocations.
    pub fn defragment_nodes(&mut self) {
        let num_nodes = self.nodes.len();
        // allocate a fresh slab for nodes that will not allocate unnecessarily
        let mut new_nodes = Slab::with_capacity(num_nodes);

        // move the root node to kick things off
        new_nodes.insert(self.nodes.remove(0));

        // for every index in new slab, move its children immediately after itself, keep doing that until all are moved.
        // this will produce a breadth-first traverse of original nodes laid out in new memory, which should keep
        // nearby nodes close in memory locations.
        for n in 0..num_nodes {
            // clone children array to keep it safe while we mess with it
            let children = new_nodes[n].children.clone();
            // now go over node's children and move them over
            for (i, old_idx) in iter_treenode_children(&children) {
                // move the child into new slab
                let new_idx = new_nodes.insert(self.nodes.remove(old_idx));
                // ensure slab is not doing anything fishy, and actually gives us correct indices
                debug_assert_eq!(new_idx, n + i + 1);
                // fix our reference to that child
                new_nodes[n].children[i] = Some(NonZeroU32::new(new_idx as u32).unwrap());

                // if the position of the node has changed
                if new_idx != old_idx {
                    // update node index of all chunks we are referring to
                    for (_, chunk_idx) in new_nodes[new_idx].iter_existing_chunks() {
                        self.chunks[chunk_idx].node_idx = new_idx as u32;
                    }
                }
            }


        }
        self.nodes = new_nodes;
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
        let mut new_nodes = Slab::with_capacity(num_nodes);
        let mut new_positions = std::collections::VecDeque::with_capacity(B);

        // move the root node to kick things off
        new_nodes.insert(self.nodes.remove(0));
        new_positions.push_back(L::root());

        // for every index in new slab, move its children immediately after itself, keep doing that until all are moved.
        // this will produce a breadth-first traverse of original nodes laid out in new memory, which should keep
        // nearby nodes close in memory locations.
        for n in 0..usize::MAX {
            let pos = match new_positions.pop_front(){
                Some(p) => p,
                None=>break
            };
            // clone children array to keep it safe while we mess with it
            let children = new_nodes[n].children.clone();

            // now go over node's children
            for (b, maybe_child) in children.iter().enumerate() {
                // figure out position of child node
                let child_pos = pos.get_child(b);
                // figure if any of the targets needs it subdivided
                let subdivide = targets.iter().any(|x| x.can_subdivide(child_pos, detail));
                //println!("{child_pos:?}, {subdivide:?}");

                // if child is subdivided we do not want a chunk there,
                // and in other case we need one, so we make one if necessary
                match (new_nodes[n].chunk[b].get(),subdivide) {
                    ( None,true) => {
                        //println!("No chunks present");
                    },
                    ( Some(chunk_idx),true) => {
                        let cont = self.chunks.remove(chunk_idx);
                        debug_assert_eq!(cont.position, child_pos);
                        evict_callback(child_pos, cont.chunk);
                        new_nodes[n].chunk[b] = ChunkPtr::None;
                    },
                    (None,false) => {
                        let chunk_idx = self.chunks.insert(ChunkContainer {
                            chunk: chunk_creator(child_pos),
                            position: child_pos,
                            node_idx: n as u32,
                            child_idx: b as u8,
                        });

                        new_nodes[n].chunk[b] = ChunkPtr::from(Some(chunk_idx));
                    },
                    (Some(chunk_idx),false) => {
                        //println!("Preserve chunk at index {chunk_idx}");
                        self.chunks[chunk_idx].node_idx = n as u32;
                    },
                }

                // make sure a child is present if we are going to subdivide
                match (maybe_child, subdivide) {
                    // no node present but we need one
                    (None, true) => {
                        let new_idx = new_nodes.insert(TreeNode::new());
                        // keep track of positions
                        new_positions.push_back(child_pos);
                        new_nodes[n].children[b] = Some(NonZeroU32::new(new_idx as u32).unwrap());
                    }
                    // no node present and we do not need one
                    (None, false) => {}
                    //existing node needed, keep it (same logic as in defragment_nodes)
                    (Some(child_idx), true) => {
                        // move the child into new slab
                        let new_idx = new_nodes.insert(self.nodes.remove(child_idx.get() as usize));
                        // keep track of positions
                        new_positions.push_back(child_pos);
                        // fix our reference to that child
                        new_nodes[n].children[b] = Some(NonZeroU32::new(new_idx as u32).unwrap());
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
                        new_nodes[n].children[b] = None;
                    }
                };
            }
        }
        self.nodes = new_nodes;

    }

    /// Shrinks all internal buffers to fit actual need, reducing fragmentation and memory usage.
    ///
    /// The next update might take longer due to memory allocations.
    #[inline]
    pub fn shrink(&mut self) {
        self.defragment_chunks();
        self.defragment_nodes();

        self.chunks.shrink_to_fit();
        self.nodes.shrink_to_fit();
    }
}

/// Construct an itreator that traverses a subtree in nodes that begins in start (including start itself).
pub fn traverse<'a, const B: usize>(
    nodes: &'a Slab<TreeNode<B>>,
    start: &'a TreeNode<B>,
) -> TraverseIter<'a, B> {
    let mut to_visit = arrayvec::ArrayVec::new();
    to_visit.push(start);
    TraverseIter { nodes, to_visit }
}

///Helper to perform breadth-first traverse of tree's nodes.
pub struct TraverseIter<'a, const B: usize> {
    nodes: &'a Slab<TreeNode<B>>,
    //to_visit: Vec<&'a TreeNode<B>>,
    to_visit: arrayvec::ArrayVec<&'a TreeNode<B>, B>,
}

impl<'a, const B: usize> Iterator for TraverseIter<'a, B> {
    type Item = &'a TreeNode<B>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let current = self.to_visit.pop()?;
        for c in current.children {
            if let Some(c) = c {
                self.to_visit.push(&self.nodes[c.get() as usize]);
            }
        }
        return Some(current);
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
        let targets = [QuadVec::build((1<<2)-1, (1<<2)-1, 2)];
        println!("=====> Update with targets {targets:?}");
        tree.lod_update(&targets, 0, |p| {
            println!("Creating chunk at {p:?}");
            TestChunk {}
        }, |p, _| {
            println!("Evicting chunk at {p:?}");
        });

        let targets = [QuadVec::build(0, 0, 2)];
        println!("=====> Update with targets {targets:?}");
        tree.lod_update(&targets, 0, |p| {
            println!("Creating chunk at {p:?}");
            TestChunk {}
        }, |p, _| {
            println!("Evicting chunk at {p:?}");
        });
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
    pub fn things() {
        //
        // // and move the target
        // while tree.prepare_update(&[QuadVec::new(16, 8, 16)], 8, |_| TestChunk {}) {
        //     // and actually update
        //     tree.do_update();
        // }
        //
        // // get the resulting chunk from a search
        // let found_chunk = tree.get_chunk_from_position(QuadVec::new(16, 8, 16));
        //
        // // and find the resulting chunk
        // println!("{:?}", found_chunk.is_some());
        //
        // // and make the tree have no items
        // while tree.prepare_update(&[], 8, |_| TestChunk {}) {
        //     // and actually update
        //     tree.do_update();
        // }
        //
        // // and do the same for an octree
        // let mut tree = Tree::<TestChunk, OctVec>::new(64);
        //
        // // as long as we need to update, do so
        // while tree.prepare_update(&[OctVec::new(128, 128, 128, 32)], 8, |_| TestChunk {}) {
        //     // and actually update
        //     tree.do_update();
        // }
        //
        // // and move the target
        // while tree.prepare_update(&[OctVec::new(16, 8, 32, 16)], 8, |_| TestChunk {}) {
        //     // and actually update
        //     tree.do_update();
        // }
        //
        // // and make the tree have no items
        // while tree.prepare_update(&[], 8, |_| TestChunk {}) {
        //     // and actually update
        //     tree.do_update();
        // }
    }
    #[test]
    pub fn defragment_chunks() {}
    #[test]
    pub fn defragment_nodes() {}

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
