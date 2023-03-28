//! Contains coordinate structs, QuadVec for quadtrees, and OctVec for octrees, as well as their LodVec implementation

use std::cmp::Ordering;



pub trait LodVec<const N: usize>:
std::hash::Hash + Eq + Sized + Copy + Clone + Send + Sync +std::fmt::Debug
{
    const MAX_CHILDREN : usize  = 1<<N;

    /// gets one of the child node position of this node, defined by it's index.
    fn get_child(self, index: usize) -> Self;
    /// the number of child nodes a node can have in the tree.
    fn contains_child_node(self, child: Self) -> bool ;
    /// returns the lod vector as if it's at the root of the tree.
    fn root() -> Self;

    /// wether the node can subdivide, compared to another node and the required detail.
    ///
    /// Assumes self is the target position for a lod.
    ///
    /// The depth determines the max lod level allowed, detail determines the amount of chunks around the target.
    ///
    /// if the detail is 0, this may only return true if self is inside the node.
    ///
    /// The implementation used in the QuadVec implementation is as follows:
    /// ```rust
    /// # struct Chunk { x: u64, y: u64, depth: u8 }
    /// # impl Chunk {
    /// fn can_subdivide(self, node: Self, detail: u64) -> bool {
    ///    // return early if the level of this chunk is too high
    ///    if node.depth >= self.depth {
    ///        return false;
    ///    }
    ///
    ///    // difference in lod level between the target and the node
    ///    let level_difference = self.depth - node.depth;
    ///
    ///    // minimum corner of the bounding box
    ///    let min = (
    ///        (node.x << (level_difference + 1))
    ///            .saturating_sub(((detail + 1) << level_difference) - (1 << level_difference)),
    ///        (node.y << (level_difference + 1))
    ///            .saturating_sub(((detail + 1) << level_difference) - (1 << level_difference)),
    ///    );
    ///
    ///    // max as well
    ///    let max = (
    ///        (node.x << (level_difference + 1))
    ///            .saturating_add(((detail + 1) << level_difference) + (1 << level_difference)),
    ///        (node.y << (level_difference + 1))
    ///            .saturating_add(((detail + 1) << level_difference) + (1 << level_difference)),
    ///    );
    ///
    ///    // local position of the target, which is one lod level higher to allow more detail
    ///    let local = (self.x << 1, self.y << 1);
    ///
    ///    // check if the target is inside of the bounding box
    ///    local.0 >= min.0 && local.0 < max.0 && local.1 >= min.1 && local.1 < max.1
    /// }
    /// # }
    /// ```
    //fn can_subdivide(self, node: Self, detail: u32) -> bool;



    /// check if this chunk is inside of a bounding box
    /// where min is the lowest corner of the box, and max is the highest corner
    /// max_depth controls the depth at which the BB checking is done.
    /// ```
    fn is_inside_bounds(self, min: Self, max: Self, max_depth: u8) -> bool;

}

pub trait ReasonableIntegerLike: Default+num::Integer+ std::marker::Send+ std::marker::Sync+std::fmt::Debug+Copy+std::hash::Hash + std::ops::Shl<isize, Output = Self> + std::ops::Shr<isize, Output = Self> + std::ops::BitAnd<Self, Output = Self> {
    fn fromusize(value: usize) -> Self;
    fn tousize(self) -> usize;
}

#[macro_export]
macro_rules! reasonable_int_impl {
    (  $x:ty  ) => {
        impl ReasonableIntegerLike for $x {
            fn fromusize(value: usize) -> Self{
                value as $x
            }
            fn tousize(self) -> usize{
                self as usize
            }
        }
    }
}

reasonable_int_impl!(u8);
reasonable_int_impl!(u16);
reasonable_int_impl!(u32);
reasonable_int_impl!(u64);


#[cfg(test)]
mod tests{
use std::mem::size_of;
use crate::coords::{OctVec,QuadVec};
#[test]
fn sizes(){
    assert_eq!(3,size_of::<QuadVec>());
    assert_eq!(4,size_of::<OctVec>());

}


}

#[derive(Debug,Copy,Clone, PartialEq, Eq, std::hash::Hash)]
pub struct CoordVec< const N:usize, DT=u8>
where  DT:ReasonableIntegerLike
{
    pub pos: [DT; N],
    pub depth:u8,
}


impl <const N:usize, DT> CoordVec<N, DT>
where  DT: ReasonableIntegerLike
{

    /// creates a new vector from the raw x and y coords.
    /// # Args
    /// * `coord` The position in the tree. Allowed range scales with the depth (doubles as the depth increases by one)
    /// * `depth` the lod depth the coord is at. This is soft limited at roughly 60, and the tree might behave weird if it gets higher
    #[inline]
    pub fn new(pos:[DT;N], depth: u8) -> Self {
        debug_assert!(depth <= 60);
        debug_assert!(pos.iter().all(|e|{e.tousize() < (1 << depth) }));

        Self { pos, depth }
    }


    /// creates a new vector from floating point coords.
    /// mapped so that (0, 0, 0) is the front bottom left corner and (1, 1, 1) is the back top right.
    /// # Args
    /// * `x` x coord of the float vector, from 0 to 1
    /// * `y` y coord of the float vector, from 0 to 1
    /// * `z` z coord of the float vector, from 0 to 1
    /// * `depth` The lod depth of the coord
    #[inline]
    pub fn from_float_coords(pos:[f32;N], depth: u8) -> Self {
        // scaling factor due to the lod depth
        let scale_factor = (1 << depth) as f32;

        // and get the actual coord
        Self {
            pos: pos.map(|e| {DT::fromusize((e * scale_factor) as usize)}),
            depth,
        }
    }

    /// converts the coord into float coords.
    /// Returns a tuple of (x: f64, y: f64, z: f64) to represent the coordinates, at the front bottom left corner.
    #[inline]
    pub fn float_coords(self) -> [f32;N] {
        // scaling factor to scale the coords down with
        let scale_factor = 1.0 / (1 << self.depth) as f32;
        self.pos.map(|e| {e.tousize() as f32 * scale_factor})
    }

    /// gets the size the chunk of this lod vector takes up, with the root taking up the entire area.
    #[inline]
    pub fn float_size(self) -> f32 {
        1.0 / (1 << self.depth) as f32
    }
}

impl  <const N:usize, DT> LodVec<N> for CoordVec<N, DT>
where DT:ReasonableIntegerLike
{
    fn root()->Self{
        Self {pos:[DT::default();N], depth:0}
    }

    #[inline]
    fn get_child(self, index: usize) -> Self {
        debug_assert!(index < <CoordVec<N> as LodVec<N>>::MAX_CHILDREN);
        // the positions, doubled in scale

        let mut new = Self::root();
        let one = DT::fromusize(1 as usize);
        let index = DT::fromusize(index);
        for i in 0..N{

            new.pos[i] = (new.pos[i] <<1) + (index & (one<<i as isize));
        }
        new.depth = self.depth + 1;
        new
    }

    fn contains_child_node(self, child: Self) -> bool {
        // basically, move the child node up to this level and check if they're equal
        let level_difference = child.depth as isize - self.depth as isize ;
        self.pos.iter().zip(child.pos).all(|(s, c)|{*s == (c>>level_difference)})
    }

    fn is_inside_bounds(self, min: Self, max: Self, max_depth: u8) -> bool {
        // get the lowest lod level
        let level = self.depth.min(min.depth.min(max.depth)) as isize;

        // bring all coords to the lowest level
        let self_difference:isize = self.depth as isize - level;
        let min_difference:isize = min.depth as isize - level;
        let max_difference:isize = max.depth as isize - level;
        // println!("diff {:?},  {:?}, {:?}", self_difference, min_difference,max_difference);
        // get the coords to that level

        let self_lowered = self.pos.map(|e|{e>>self_difference});
        let min_lowered = min.pos.map(|e|{e>>min_difference});
        let max_lowered = max.pos.map(|e|{e>>max_difference});


        // then check if we are inside the AABB
        self.depth <= max_depth &&
        itertools::izip!(self_lowered,min_lowered, max_lowered)
        .all(|(slf, min, max)|{slf>=min && slf <=max })
    }


}

pub type OctVec<DT=u8> = CoordVec<3, DT>;
pub type QuadVec<DT=u8> = CoordVec<2, DT>;

impl <DT>OctVec<DT>
where DT:ReasonableIntegerLike
{
    pub fn build(x:DT,y:DT,z:DT,depth:u8)->Self{
        Self{pos:[x,y,z], depth}
    }
}


impl <DT>QuadVec<DT>
where DT:ReasonableIntegerLike
{
    pub fn build(x:DT,y:DT,depth:u8)->Self{
        Self{pos:[x,y], depth}
    }
}

impl <const N:usize, DT> PartialOrd for CoordVec<N, DT>
where DT:ReasonableIntegerLike
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.depth != other.depth {
            return None;
        }

        if self.pos ==other.pos{
            return Some(Ordering::Equal);
        }

        if self.pos.iter().zip(other.pos).all(|(s,o)|{*s<o}){
            return Some(Ordering::Less);
        }
        else if self.pos.iter().zip(other.pos).all(|(s,o)|{*s>o}){
            return Some(Ordering::Greater);
        }
        None
    }
}



/*

    #[inline]
    fn can_subdivide(self, node: Self, detail: u32) -> bool {
        let detail = detail as u64;
        // return early if the level of this chunk is too high
        if node.depth >= self.depth {
            return false;
        }

        // difference in lod level between the target and the node
        let level_difference = self.depth - node.depth;

        // minimum corner of the bounding box
        let min = (
            (node.x << (level_difference + 1))
                .saturating_sub(((detail + 1) << level_difference) - (1 << level_difference)),
            (node.y << (level_difference + 1))
                .saturating_sub(((detail + 1) << level_difference) - (1 << level_difference)),
        );

        // max as well
        let max = (
            (node.x << (level_difference + 1))
                .saturating_add(((detail + 1) << level_difference) + (1 << level_difference)),
            (node.y << (level_difference + 1))
                .saturating_add(((detail + 1) << level_difference) + (1 << level_difference)),
        );

        // local position of the target, which is one lod level higher to allow more detail
        let local = (self.x << 1, self.y << 1);

        // check if the target is inside of the bounding box
        local.0 >= min.0 && local.0 < max.0 && local.1 >= min.1 && local.1 < max.1
    }*/






impl  <const N:usize, DT> Default for CoordVec<N, DT>
where DT:ReasonableIntegerLike
{
    fn default() -> Self {
        Self::root()
    }
}
