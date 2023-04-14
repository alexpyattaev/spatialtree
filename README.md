[![Documentation](https://docs.rs/spatialtree/badge.svg)](https://docs.rs/spatialtree)

# Spatial trees
Spatial trees, (aka QuadTrees, OctTrees, LodTrees) are a family of fast tree data structures that supports complex spatial queries at various level of detail. They are particularly well suited for sparse data storage and neighbor queries.

## Acknowledgements
Internals are partially based on:
  * https://stackoverflow.com/questions/41946007/efficient-and-well-explained-implementation-of-a-quadtree-for-2d-collision-det
  * https://github.com/Dimev/lodtree


## Goals
The aim of this crate is to provide a generic, easy to use tree data structure that can be used to make Quadtrees, Octrees for various realtime applications (e.g. games or GIS software).

Internally, the tree tries to keep all needed memory allocated in slab arenas to avoid the memory fragmentation and allocator pressure. All operations, where possible, use either stack allocations or allocate at most once.
 
## Accepted design compromises

 - Data chunks that are nearby in space do not necessarily land in nearby locations in the tree's memory.
 - There is no way to defragment node storage memory short of rebuilding the tree from scratch (which means doubling memory usage)


## Features
 - Highly tunable for different scales (from 8 bit to 64 bit coordinates), 2D, 3D, N-D if desired.
 - Minimized memory (re)allocations and moves
 - Provides a selection iterators for finding chunks in certain bounds
 - Supports online defragmentation for data chunks to optimize sequential operations on all chunks
 - External chunk cache can be used to allow reusing chunks at a memory tradeoff


### Examples:
 - [rayon](examples/rayon.rs): shows how to use the tree with rayon to generate new chunks in parallel, and cache chunks already made.
 - [glium](examples/glium.rs): shows how a basic drawing setup would work, with glium to do the drawing.

## Usage:
Import the crate
```rust
use spatialtree::*;
```

The tree is it's own struct, and accepts a chunk (anything that implements Sized) and the lod vector (Anything that implements the LodVec trait).
```rust
let mut tree = OctTree::<Chunk, OctVec>::new();
```


## Roadmap
### 0.2.0:
 - swap L and C, so the key (position) is before the chunk, which is consistent with other key-value datatypes in rust
 - more benchmarks
### 0.3.0:
 - use generic const expressions to improve API

## License
Licensed under either
 * GPL, Version 3.0
   ([LICENSE.txt](LICENSE.txt) )
 * Proprietary license for commercial use (contact author to arrange licensing)


## Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, shall be licensed as above, without any additional terms or conditions.
