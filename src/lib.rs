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

#![doc = include_str!("../README.md")]




//!
//! When a chunk is removed from the tree, it will be put in the cache.
//! When a new chunk is then added to the tree, it's fetched from the cache when possible.
//! This should help avoid needing to regenerate all new chunks, as they are fetched from the internal cache.
//!
//! Caching is most effective with a larger cache size as well as the target position moving around in roughly the same area.
//! Of course, it comes at a memory tradeoff, as it will keep all chunks in the cache stored in memory
//!
//! # Chunk groups
//! There's several groups of chunks that can be accessed inside the tree.
//! - `chunks`: All chunks currently stored inside the tree
//! - `chunks_to_add`: Chunks that will be added after the next `tree.do_update();`
//! - `chunks_to_deactivate`: Chunks that have subdivided and thus need to be invisible.
//! - `chunks_to_activate`: Chunks that were previously subdivided, but are now going to be leaf nodes. This means they should be visible again
//! - `chunks_to_remove`: Chunks that will be removed from the tree after the next `tree.do_update()`. Note that these can be put in the chunk cache and appear in `chunks_to_add` at a later point
//! - `chunks_to_delete`: Chunks that are permanently removed from the tree, as they were removed from the tree itself, and will now also be removed from the chunk cache
//!
//! Cached chunks are also stored seperate from the tree, inside a HashMap. These can't be accessed.
//!
//! # Iterators
//! Iterators are provided for each chunk group, in the flavour of chunks, mutable chunks, chunk and positions and mutable chunk and positions.
//!
//! # Getters
//! Getters are also given for all chunk groups, in the flavor of get a chunk, get a mutable chunk, get a mutable pointer to a chunk and get the position of a chunk.

pub mod coords;
pub use crate::coords::*;

pub mod util_funcs;
pub use crate::util_funcs::*;

pub mod tree;
pub use crate::tree::*;


pub mod iter;
pub use crate::iter::*;
