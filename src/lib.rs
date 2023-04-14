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


pub mod coords;
pub use crate::coords::*;

pub mod util_funcs;
pub use crate::util_funcs::*;

pub mod tree;
pub use crate::tree::*;


pub mod iter;
pub use crate::iter::*;
