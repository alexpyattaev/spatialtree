//! Contains the tree struct, which is used to hold all chunks

use crate::coords::*;

use std::fmt::Debug;
use std::num::NonZeroU32;



/// Type for relative pointers in the tree. Kept 32bit for cache locality during lookups.
/// Should you need > 4 billion nodes in the tree do let me know.
pub type RelPtr=Option<NonZeroU32>;

//TODO - use struct of arrays derive
/// Tree node that encompasses multiple children at once. This just barely fits into one cache line for octree.
/// For each possible child, the node has two relative pointers:
///  - children[i] will point to the TreeNode in a given branch direction
///  - chunk[i] will point to the data chunk.
/// both pointers may be "None", indicating either no children, or no data
#[derive(Clone, Debug)]
pub(crate) struct TreeNode<const N:usize, L: LodVec<N>>  {
    // children, these can't be the root (index 0), so we can use Some and Nonzero for slightly more compact memory
    pub(crate) children: [RelPtr; NUM_CHILDREN],

    // where the chunks for particular children is stored (if any)
    pub(crate) chunk: [RelPtr; NUM_CHILDREN],
}

impl <const N:usize, L: LodVec<N>>TreeNode<N, L>{
    fn new()->Self{
        Self{
            children: [RelPtr::None;N],
            chunk:[RelPtr::None;N],
        }
    }
}


// utility struct for holding actual chunks and the node that owns them
#[derive(Clone, Debug)]
pub(crate) struct ChunkContainer<const N:usize, C: Sized, L: LodVec<N>> {
    pub(crate) chunk: C,    // actual data inside the chunk
    pub(crate) node_idx: u32,  // index of the node that holds this chunk
    pub(crate) position: L, // where the chunk is (as this can not be recovered from node tree)
}

/// holds a descriptor of chunk to be added
/// modifying the position will break things badly.
#[derive(Clone, Debug)]
pub struct ToAddContainer<const N:usize, L: LodVec<N>> {
    /// Position of the chunk to add
    pub position: L,
    chunk_index: u32,  // chunk array index
    /// Index of the parent node
    parent_node_index: u32,
}

// utility struct for holding chunks to remove
#[derive(Clone, Debug)]
struct ToRemoveContainer {
    chunk: u32,  // chunk index
    parent: u32, // parent index
}



// utility struct for holding chunks in the queue
#[derive(Clone, Debug)]
struct QueueContainer<const N:usize,L: LodVec<N>> {
    node: u32,   // chunk index
    position: L, // and it's position
}
use freelist::{FreeList, Idx};

// Tree holding the actual data permanently in memory.
// partially based on: https://stackoverflow.com/questions/41946007/efficient-and-well-explained-implementation-of-a-quadtree-for-2d-collision-det
#[derive(Clone, Debug)]
pub struct Tree<const N:usize, C: Sized, L: LodVec<N>>{
    /// All chunks in the tree
    pub(crate) chunks: FreeList<ChunkContainer<N, C, L>>,
    /// nodes in the Tree
    pub(crate) nodes: FreeList<TreeNode<N,L>>,

    /// indices of the chunks that need to be activated (i.e. the chunks that have just lost children)
    //chunks_to_activate: Vec<u32>,

    /// indices of the chunks that need to be deactivated (i.e. chunks that have been subdivided in this iteration)
    //chunks_to_deactivate: Vec<u32>,

    /// internal queue for processing, that way we won't need to reallocate it
    processing_queue: Vec<QueueContainer<N, L>>,
}

impl<const N:usize, C, L> Tree<N, C, L>
where
    C: Sized,
    L: LodVec<N>,
{
    /// Create a new, empty tree
    pub fn new() -> Self {

        // create root node right away, no point having a tree without a root.
        let mut nodes = FreeList::new();
        nodes.add(TreeNode::new());
        Self {
            /*chunks_to_add: Vec::new(),
             *            chunks_to_remove: Vec::new(),
             *            chunks_to_activate: Vec::new(),
             *            chunks_to_deactivate: Vec::new(),*/
            chunks: FreeList::new(),
            nodes,
            processing_queue: Vec::new(),
        }
    }

    /// create a tree with preallocated memory for chunks and nodes
    pub fn with_capacity(_capacity: usize) -> Self {
        // TODO: make sure freelist has with_capacity
        // TODO: make capacity make sense
        Self::new()
    }


    /// Gets the node vector from a position.
    /// If position is not pointing to a node in tree, None is returned.
    fn get_node_by_position(&self, position: L) -> Option<&mut TreeNode<N,L>> {
        // the current node
        let mut current = unsafe{self.nodes.get_unchecked_mut(0)};

        // and position
        let mut current_position = L::root();

        // then loop
        loop {
            // if the current node is the one we are looking for, return
            if current_position == position {
                return Some(current.chunk as usize);
            }

            // if the current node does not have children, stop
            // this works according to clippy
            current.children?;

            // if not, go over the node children
            if let Some((index, found_position)) = (0..L::NUM_CHILDREN as u32)
                .map(|i| (i, current_position.get_child(i)))
                .find(|(_, x)| x.contains_child_node(position))
            {
                // we found the position to go to
                current_position = found_position;

                // and the node is at the index of the child nodes + index
                current = self.nodes[(current.children.unwrap().get() + index) as usize];
            } else {
                // if no child got found that matched the item, return none
                return None;
            }
        }
    }


    /// get the number of chunks in the tree
    #[inline]
    pub fn get_num_chunks(&self) -> usize {
        todo!()
        //self.chunks.len()
    }

    /// get a chunk
    #[inline]
    pub fn get_chunk(&self, index: usize) -> &C {
        &self.chunks[Idx::new(index)].chunk
    }

    /// get a chunk as mutable
    #[inline]
    pub fn get_chunk_mut(&mut self, index: usize) -> &mut C {
        &mut self.chunks[Idx::new(index)].chunk
    }


    /// get a chunk by position, or none if it's not in the tree
    #[inline]
    pub fn get_chunk_by_position(&self, position: L) -> Option<&C> {
        // get the index of the chunk
        let chunk_index = self.get_node_index_from_position(position)?;

        // and return the chunk
        Some(&self.chunks[Idx::new(chunk_index)].chunk)
    }

    /// get a mutable chunk by position, or none if it's not in the tree
    #[inline]
    pub fn get_chunk_from_position_mut(&mut self, position: L) -> Option<&mut C> {
        // get the index of the chunk
        let chunk_index = self.get_node_index_from_position(position)?;

        // and return the chunk
        Some(&mut self.chunks[Idx::new(chunk_index)].chunk)
    }


    /// gets a mutable pointer to a chunk
    /// This casts get_chunk_mut to a pointer underneath the hood
    #[inline]
    pub fn get_chunk_pointer_mut(&mut self, index: usize) -> *mut C {
        self.get_chunk_mut(index)
    }

    /// get the position of a chunk
    #[inline]
    pub fn get_chunk_position(&self, index: usize) -> L {
        self.chunks[index].position
    }



    /// Adds chunks at and around specified locations.
    /// This operation will also add chunks at other locations around the target to fullfill the
    /// datastructure constraints (such that no partially filled nodes exist).
    pub fn prepare_insert(
        &mut self,
        targets: &[L],
        detail: u32,
        chunk_creator: &mut dyn FnMut(L) -> C,
    ) -> bool {
        //FIXME: this function currently will dry-run once for every update to make sure
        // there is nothing left to update. This is a waste of CPU time, especially for many targets

        // first, clear the previous arrays
        // self.chunks_to_add.clear();
        // self.chunks_to_remove.clear();
        // self.chunks_to_activate.clear();
        // self.chunks_to_deactivate.clear();

        // if we don't have a root, make one pending for creation
        if self.nodes.is_empty() {
            // chunk to add
            let chunk_to_add = chunk_creator(L::root());

            // we need to add the root as pending
            // self.chunks_to_add.push(ToAddContainer {
            //     position: L::root(),
            //     chunk: chunk_to_add,
            //     parent_node_index:0,
            // });

            // and an update is needed
            return true;
        }

        // clear the processing queue from any previous updates
        self.processing_queue.clear();

        // add the root node (always at 0, if there is no root we would have returned earlier) to the processing queue
        self.processing_queue.push(QueueContainer {
            position: L::root(),
            node: 0,
        });

        // then, traverse the tree, as long as something is inside the queue
        while let Some(QueueContainer {
            position: current_position,
            node: current_node_index,
        }) = self.processing_queue.pop()
        {
            // fetch the current node
            let current_node = self.nodes[current_node_index as usize];
            //dbg!(current_node_index, current_node);
            // if we can subdivide, and the current node does not have children, subdivide the current node
            if current_node.children.is_none() {
                //println!("adding children");
                // add children to be added
                for i in 0..L::NUM_CHILDREN as u32 {
                    // chunk to add
                    let chunk_to_add = chunk_creator(current_position.get_child(i));

                    // add the new chunk to be added
                    // self.chunks_to_add.push(ToAddContainer {
                    //     position: current_position.get_child(i),
                    //     chunk: chunk_to_add,
                    //     parent_node_index:current_node_index,
                    // });

                }

                // and add ourselves for deactivation
                self.chunks_to_deactivate.push(current_node_index);
            } else if let Some(index) = current_node.children {
                //println!("has children at {index:?}");
                // queue child nodes for processing
                for i in 0..L::NUM_CHILDREN as u32 {
                    // wether we can subdivide
                    let child_pos = current_position.get_child(i);
                    //dbg!(child_pos);
                    for t in targets {
                        if *t == child_pos {
                            //println!("Found match for target {t:?}");
                            self.chunks[(index.get() + i) as usize].chunk =
                                chunk_creator(child_pos);
                            continue;
                        }
                        if t.can_subdivide(child_pos, detail) {
                            self.processing_queue.push(QueueContainer {
                                position: child_pos,
                                node: index.get() + i,
                            });
                            break;
                        }
                    }
                }
            }
        }

        // and return whether an update needs to be done
        !self.chunks_to_add.is_empty()
    }


    /// Runs the update that's stored in the internal lists.
    /// This adds and removes chunks based on that, however this assumes that chunks in the to_activate and to_deactivate list were manually activated or deactivated.
    /// This also assumes that the chunks in to_add had proper initialization, as they are added to the tree.
    /// After this, it's needed to clean un nodes in the chunk_to_delete list and call the function complete_update(), in order to properly clear the cache
    /*pub fn do_update(&mut self) {
        // no need to do anything with chunks that needed to be (de)activated, as we assume that has been handled beforehand

        // first, get the iterator for chunks that will be added
        // this becomes useful later
        let mut chunks_to_add_iter = self.chunks_to_add.drain(..);

        // then, remove old chunks, or cache them
        // we'll drain the vector, as we don't need it anymore afterward
        for ToRemoveContainer {
            chunk: index,
            parent: parent_index,
        } in self.chunks_to_remove.drain(..)
        // but we do need to cache these
        {
            // remove the node from the tree
            self.nodes[parent_index as usize].children = None;
            self.free_list.push(index);

            // and remove the chunk
            let chunk_index = self.nodes[index as usize].chunk;

            // but not so fast, because if we can overwrite it with a new chunk, do so
            // that way we can avoid a copy later on, which might be expensive
            if let Some(ToAddContainer { position, chunk, parent_node_index:parent_index}) =
                chunks_to_add_iter.next()
            {
                // add the node
                let new_node_index = match self.free_list.pop() {
                    Some(x) => {
                        // reuse a free node
                        self.nodes[x as usize] = TreeNode {
                            children: None,
                            chunk: chunk_index,
                        };

                        // old chunk that was previously in the array
                        // we initialize it to the new chunk, then swap them
                        let mut old_chunk = ChunkContainer {
                            index: x,
                            chunk,
                            position,
                        };

                        std::mem::swap(&mut old_chunk, &mut self.chunks[chunk_index as usize]);

                        // old chunk shouldn't be mutable anymore
                        let old_chunk = old_chunk;


                        x
                    }
                    // This can't be reached due to us *always* adding a chunk to the free list before popping it
                    None => unsafe { std::hint::unreachable_unchecked() },
                };

                // correctly set the children of the parent node.
                // because the last node we come by in with ordered iteration is on num_children - 1, we need to set it as such].
                // node 0 is the root, so the last child it has will be on num_children.
                // then subtracting num_children - 1 from that gives us node 1, which is the first child of the root.
                if new_node_index >= L::NUM_CHILDREN as u32 {
                    // because we loop in order, and our nodes are contiguous, the first node of the children got added on index i - (num children - 1)
                    // so we need to adjust for that
                    self.nodes[parent_index as usize].children =
                        NonZeroU32::new(new_node_index - (L::NUM_CHILDREN - 1) as u32);
                }
            } else {
                // otherwise we do need to do a regular swap remove
                let old_chunk = self.chunks.swap_remove(chunk_index as usize);

            }

            // and properly set the chunk pointer of the node of the chunk we just moved, if any
            // if we removed the last chunk, no need to update anything
            if chunk_index < self.chunks.len() as u32 {
                self.nodes[self.chunks[chunk_index as usize].index as usize].chunk = chunk_index;
            }
        }

        // add new chunks
        // we'll drain the vector here as well, as we won't need it anymore afterward
        for ToAddContainer { position, chunk, parent_node_index:parent_index } in chunks_to_add_iter {
            // add the node
            let new_node_index = match self.free_list.pop() {
                Some(x) => {
                    // reuse a free node
                    self.nodes[x as usize] = TreeNode {
                        children: None,
                        chunk: self.chunks.len() as u32,
                    };
                    self.chunks.push(ChunkContainer {
                        index: x,
                        chunk,
                        position,
                    });
                    x
                }
                None => {
                    // otherwise, use a new index
                    self.nodes.push(TreeNode {
                        children: None,
                        chunk: self.chunks.len() as u32,
                    });
                    self.chunks.push(ChunkContainer {
                        index: self.nodes.len() as u32 - 1,
                        chunk,
                        position,
                    });
                    (self.nodes.len() - 1) as u32
                }
            };

            // correctly set the children of the parent node.
            // because the last node we come by in with ordered iteration is on num_children - 1, we need to set it as such].
            // node 0 is the root, so the last child it has will be on num_children.
            // then subtracting num_children - 1 from that gives us node 1, which is the first child of the root.
            if new_node_index >= L::NUM_CHILDREN as u32 {
                // because we loop in order, and our nodes are contiguous, the first node of the children got added on index i - (num children - 1)
                // so we need to adjust for that
                self.nodes[parent_index as usize].children =
                    NonZeroU32::new(new_node_index - (L::NUM_CHILDREN - 1) as u32);
            }
        }

        // if there's only chunk left, we know it's the root, so we can get rid of all free nodes and unused nodes
        if self.chunks.len() == 1 {
            self.free_list.clear();
            self.nodes.resize(
                1,
                TreeNode {
                    children: None,
                    chunk: 0,
                },
            );
        }

        // and clear all internal arrays, so if this method is accidentally called twice, no weird behavior would happen
        self.chunks_to_add.clear();
        self.chunks_to_remove.clear();
        self.chunks_to_activate.clear();
        self.chunks_to_deactivate.clear();
    }*/



    /// clears the tree, removing all chunks and internal lists and cache
    #[inline]
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.nodes.clear();
        /*self.chunks_to_add.clear();
        self.chunks_to_remove.clear();
        self.chunks_to_activate.clear();
        self.chunks_to_deactivate.clear();*/

        self.processing_queue.clear();
    }

    /// Defragments the chunks array to enable fast iteration.
    /// This will have zero cost if array does not need defragmentation.
    /// The next update might take longer due to memory allocations.
    pub fn defragment_chunks(&mut self){

        todo!("rebuild the chunks by moving elements into holes in freelists");
    }

    /// Defragments the nodes array to enable fast iteration.
    /// This will have zero cost if array does not need defragmentation.
    /// The next update might take longer due to memory allocations.
    pub fn defragment_nodes(&mut self){
        todo!("rebuild the tree nodes by moving elements into holes in freelists");
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
        /*
        self.chunks_to_add.shrink_to_fit();
        self.chunks_to_remove.shrink_to_fit();
        self.chunks_to_activate.shrink_to_fit();
        self.chunks_to_deactivate.shrink_to_fit();
        self.chunks_to_delete.shrink_to_fit();*/
        self.processing_queue.shrink_to_fit();
    }



}

impl<const N:usize, C, L> Default for Tree<N, C, L>
where
    C: Sized,
    L: LodVec<N>,
{
    /// creates a new, empty tree, with no cache
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests {

    use super::*;

    struct TestChunk;

/*    #[test]
    fn update_tree() {
        // make a tree
        let mut tree = Tree::<TestChunk, QuadVec>::new();
        // as long as we need to update, do so
        for tgt in [QuadVec::build(1, 1, 2), QuadVec::build(2, 3, 2)] {
            dbg!(tgt);
            while tree.prepare_update(&[tgt], 0, &mut |_| TestChunk {}) {
                for c in tree.iter_chunks_to_activate_positions() {
                    println!("* {c:?}");
                }
                for c in tree.iter_chunks_to_deactivate_positions() {
                    println!("o {c:?}");
                }

                for c in tree.iter_chunks_to_remove_positions() {
                    println!("- {c:?}");
                }

                for c in tree.iter_chunks_to_add_positions() {
                    println!("+ {c:?}");
                }
                println!("updating...");
                // and actually update
                tree.do_update();
            }
        }
    }
    #[test]
    fn insert_into_tree() {
        // make a tree
        let mut tree = Tree::<TestChunk, QuadVec>::new();
        // as long as we need to update, do so
        for tgt in [
            QuadVec::build(1, 1, 1),
            QuadVec::build(0, 1, 1),
            QuadVec::build(2, 3, 2),
            QuadVec::build(2, 2, 2),
        ] {
            println!("====NEXT TARGET =====");
            dbg!(tgt);
            while tree.prepare_insert(&[tgt], 0, &mut |_| TestChunk {}) {
                for c in tree.iter_chunks_to_activate_positions() {
                    println!("* {c:?}");
                }
                for c in tree.iter_chunks_to_deactivate_positions() {
                    println!("o {c:?}");
                }

                for c in tree.iter_chunks_to_remove_positions() {
                    println!("- {c:?}");
                }

                for c in tree.iter_chunks_to_add_positions() {
                    println!("+ {c:?}");
                }
                println!("updating...");
                // and actually update
                tree.do_update();
            }
        }
    }
*/
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
    pub fn alignment() {
        assert_eq!(std::mem::size_of::<TreeNode>(), 64);
    }
}
