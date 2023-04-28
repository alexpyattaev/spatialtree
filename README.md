[![Crates.io](https://img.shields.io/crates/v/spatialtree)](https://crates.io/crates/spatialtree)
[![Documentation](https://docs.rs/spatialtree/badge.svg)](https://docs.rs/spatialtree)

# Spatial trees
Spatial trees, (aka QuadTrees, OctTrees, LodTrees) are a family of fast tree data structures that supports complex spatial queries at various level of detail. They are particularly well suited for sparse data storage and neighbor queries.

## Acknowledgements
Internals are partially based on:
  * <https://stackoverflow.com/questions/41946007/efficient-and-well-explained-implementation-of-a-quadtree-for-2d-collision-det>
  * <https://github.com/Dimev/lodtree>


## Goals
The aim of this crate is to provide a generic, easy to use tree data structure that can be used to make Quadtrees, Octrees for various realtime applications (e.g. games or GIS software).

Internally, the tree tries to keep all needed memory allocated in slab arenas to avoid the memory fragmentation and allocator pressure. All operations, where possible, use either stack allocations or allocate at most once.
 

## Features
 - Highly tunable for different scales (from 8 bit to 64 bit coordinates), 2D, 3D, N-D if desired.
 - Minimized memory (re)allocations and moves
 - Provides a selection iterators for finding chunks in certain bounds
 - Supports online defragmentation for data chunks to optimize sequential operations on all chunks
 - External chunk cache can be used to allow reusing chunks at a memory tradeoff

## Accepted design compromises

 - Data chunks that are nearby in space do not necessarily land in nearby locations in the tree's memory.
 - There is no way to defragment node storage memory short of rebuilding the tree from scratch (which means doubling memory usage)


### Examples:
 - [rayon](examples/rayon.rs): shows how to use the tree with rayon to generate new chunks in parallel, and cache chunks already made.
 - [glium](examples/glium.rs): shows how a basic drawing setup would work, with glium to do the drawing.

## Usage:
Import the crate
```rust
use spatialtree::*;
```

The tree is it's own struct, and accepts a chunk (anything that implements Sized) and the coordinate vector (Anything that implements the LodVec trait).
```rust
# use spatialtree::*;
struct Chunk {
  //important useful fileds go here
  }
// a new OctTree with no capacity
let mut tree = OctTree::<Chunk, OctVec>::new();
// a new OctTree with 32 slots for nodes and 64 slots for data chunks
let mut tree = QuadTree::<Chunk, QuadVec>::with_capacity(32, 64);
```

The given LodVec implementations (OctVec and QuadVec) take in 4 and 3 arguments respectively.
The first 3/2 are the position in the tree, which is dependant on the lod level.
and the last parameter is the lod level. No lods smaller than this will be generated for this target.

```rust
# use spatialtree::*;
// QuadVec takes x,y and depth
let qv1 = QuadVec::build(1u8, 2, 3);
let qv2 = QuadVec::new([1u8, 2], 3);
assert_eq!(qv1, qv2);
// OctVec takes x,y,z and depth
let ov1 = OctVec::build(1u8, 2, 3, 3);
let ov2 = OctVec::new([1u8, 2, 3], 3);
assert_eq!(ov1, ov2);
```


Inserts are most efficient when performed in large batches, as this minimizes tree traverse overhead.
```rust
# use spatialtree::*;
# struct Chunk {}
// create a tree
let mut tree = QuadTree::<Chunk, QuadVec>::with_capacity(32, 64);
// create a few targets
let targets = [
  QuadVec::build(1u8, 1, 3),
  QuadVec::build(2, 2, 3),
  QuadVec::build(3, 3, 3),
  ];
// ask tree to populate given positions, calling a function to construct new data as needed.
tree.insert_many(targets.iter().copied(), |_| Chunk {});
```

Alternatively, if you want to insert data one chunk at a time:
```rust
# use spatialtree::*;
// create a tree with usize as data
let mut tree = QuadTree::<usize, QuadVec>::with_capacity(32, 64);
// insert/replace a chunk at selected location, provided lambda builds the content
// we get chunk index back
let idx = tree.insert(QuadVec::new([1u8,2], 3), |p| {p.pos.iter().sum::<u8>() as usize} );
// we can access chunks by index (in this case to make sure insert worked ok)
assert_eq!(tree.get_chunk(idx).chunk, 3);
```


## Advanced usage
This structure can be used for purposes such as progressive LOD.

If you want to update chunks due to the camera being moved, you can do so with lod_update.
It takes in 3 parameters.

Targets: is the array of locations around which to generate the most detail.

Detail: The amount of detail for the targets.
The default implementation defines this as the amount of chunks at the target lod level surrounding the target chunk.

chunk_creator: the function to call when new chunk is needed
evict_callback: function to call when chunk is evicted from data structure

The purpose of evict_callback is to enable things such as caching, object reuse etc. If this is done it may be wise
to keep chunks fairly small such that moving them between tree and cache is not too expensive.

```rust
# use spatialtree::*;
# use std::collections::HashMap;
# use std::cell::RefCell;
# use std::borrow::BorrowMut;
// Tree with "active" data
let mut tree = QuadTree::<Vec<usize>, QuadVec>::with_capacity(32, 64);
// Cache for "inactive" data
let mut cache: RefCell<HashMap::<QuadVec,Vec<usize>>> = RefCell::new(HashMap::new());
# fn expensive_init(c:QuadVec, d:&mut Vec<usize>){ }

//function to populate new chunks. Will read from cache if possible
let mut chunk_creator = |c:QuadVec|->Vec<usize> {
  match cache.borrow_mut().remove(&c){
    Some(d)=>d,
    None => {
      let mut d = Vec::new();
      // run whatever mystery function may be needed to populate new chunk with reasonable data
      expensive_init(c, &mut d);
      d
      }
    }
};

//Function to deal with evicted chunks. Will move them to cache.
let mut  chunk_evict = |c:QuadVec, d:Vec<usize>|{
  println!("Chunk {d:?} at position {c:?} evicted");
  cache.borrow_mut().insert(c,d);
};

// construct vector pointing to location that we want to detail
let qv = QuadVec::from_float_coords([1.1, 2.4], 6);
// run the actual update rebuilding the tree
tree.lod_update(&[qv], 2, chunk_creator, chunk_evict);
```
Internally, lod_update will rebuild the tree to match needed node structure that reflects locations of all targets.
Thus, defragment_nodes is never needed after lod_update. You may want to defragment_chunks if you are going to iterate over them.

### Optimize memory layout

For best performance, memory compactness, you may wish to ensure that chunks are stored in a
contigous array. While tree will try to ensure this at all times, it will not move data it does not
have to, unless explicitly instructed to do so. Thus, after multiple deletions it may be necessary to
defragment the storage.

```rust
# use spatialtree::*;
# let mut tree = QuadTree::<usize, QuadVec>::with_capacity(32, 64);

// make sure there are no holes in chunks array for fast iteration
tree.defragment_chunks();
```
Note that defragment_chunks will not do anything if chunks array has no holes already, so it is safe to call it every time
you suspect you might need to.

Similarly, nodes storage can be rebuilt and defragmented, though this is a substantially more costly operation, and needs to allocate
memory. Thus, call this only when you have strong reasons (i.e. benchmarks) to do so.
```rust
# use spatialtree::*;
# let mut tree = QuadTree::<usize, QuadVec>::with_capacity(32, 64);

// make sure there are no holes in nodes array for fast iteration
tree.defragment_nodes();
```

Once structures are defragmented, any memory that was freed can be reclaimed with
```rust
# use spatialtree::*;
# let mut tree = QuadTree::<usize, QuadVec>::with_capacity(32, 64);

tree.shrink_to_fit();
```


## Roadmap
### 0.2.0:
 - There is no way to prune nodes (yet). They do not eat much RAM, but it may become a problem.
 - Organize benchmarks better

### 0.3.0:
 - Swap L and C, so the key (position) is before the chunk, which is consistent with other key-value datatypes in rust
 - Use generic const expressions to improve templating


## License
Licensed under either
 * GPL, Version 3.0
   ([LICENSE.txt](LICENSE.txt) )
 * Proprietary license for commercial use (contact author to arrange licensing)


## Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, shall be licensed as above, without any additional terms or conditions.
