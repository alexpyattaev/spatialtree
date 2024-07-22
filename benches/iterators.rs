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

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use rand::rngs::SmallRng;
use rand::SeedableRng;
use spatialtree::coords::*;
use spatialtree::*;

const N_LOOKUPS: usize = 40;

type DataType = u8;

fn generate_area_bounds(rng: &mut SmallRng, depth: u8) -> (OctVec<DataType>, OctVec<DataType>) {
    let cmax = ((1usize << depth as usize) - 1) as DataType;

    let min = rand_cv(
        rng,
        OctVec::new([0, 0, 0], depth),
        OctVec::new([cmax - 2, cmax - 2, cmax - 2], depth),
    );
    let max = rand_cv(
        rng,
        min + OctVec::new([1, 1, 1], depth),
        OctVec::new([cmax, cmax, cmax], depth),
    );

    (min, max)
}

struct ChuChunk {
    a_index: u8,
    b_index: u8,
    material_index: u16,
}

impl Default for ChuChunk {
    fn default() -> ChuChunk {
        ChuChunk {
            a_index: 1,
            b_index: 2,
            material_index: 3,
        }
    }
}

fn create_and_fill_octree<C: Default>(num_chunks: u32, depth: u8) -> OctTree<C, OctVec> {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut tree: OctTree<C, OctVec> =
        OctTree::with_capacity(num_chunks as usize, num_chunks as usize);

    let cmax = ((1usize << depth as usize) - 1) as u8;

    for _c in 0..num_chunks {
        let qv: CoordVec<3> = rand_cv(
            &mut rng,
            OctVec::new([0, 0, 0], depth),
            OctVec::new([cmax, cmax, cmax], depth),
        );
        tree.insert(qv, |_p| C::default());
    }
    tree
}

fn bench_lookups_in_octree(tree: &OctTree<ChuChunk, OctVec>, depth: u8) {
    let mut rng = SmallRng::seed_from_u64(42);
    for _ in 0..N_LOOKUPS {
        let (min, max) = generate_area_bounds(&mut rng, depth);
        for i in tree.iter_chunks_in_aabb(min, max) {
            black_box(i);
        }
    }
}

fn bench_mut_lookups_in_octree(tree: &mut OctTree<ChuChunk, OctVec>, depth: u8) {
    let mut rng = SmallRng::seed_from_u64(42);
    for _ in 0..N_LOOKUPS {
        let (min, max) = generate_area_bounds(&mut rng, depth);
        for i in tree.iter_chunks_in_aabb_mut(min, max) {
            i.1.material_index = i.1.material_index.wrapping_add(1);
            i.1.a_index = i.1.a_index.wrapping_add(1);
            i.1.b_index = i.1.b_index.wrapping_add(1);
        }
    }
}

pub fn tree_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutable iteration");

    for (&depth, samples_num) in [4u8, 6, 8].iter().zip([100, 40, 10]) {
        group.significance_level(0.1).sample_size(samples_num);

        let num_chunks: u32 = 2u32.pow(depth as u32).pow(3) / 10;
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            let mut tree = create_and_fill_octree::<ChuChunk>(num_chunks, depth);
            b.iter(|| {
                bench_mut_lookups_in_octree(&mut tree, depth);
            });
            black_box(tree);
        });
    }
    group.finish();

    let mut group = c.benchmark_group("immutable iteration");

    for (&depth, samples_num) in [4u8, 6, 8].iter().zip([100, 40, 10]) {
        group.significance_level(0.1).sample_size(samples_num);
        let num_chunks: u32 = 2u32.pow(depth as u32).pow(3) / 10;
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            let tree = create_and_fill_octree::<ChuChunk>(num_chunks, depth);
            b.iter(|| {
                bench_lookups_in_octree(&tree, depth);
            });
        });
    }
    group.finish();
}

pub fn tree_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree creation");

    for (&depth, samples_num) in [4u8, 6, 8].iter().zip([100, 40, 10]) {
        group.significance_level(0.1).sample_size(samples_num);
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            let volume = 2u32.pow(depth as u32).pow(3);
            let num_chunks: u32 = volume / 10;
            //println!("Creating {num_chunks} voxels out of {volume} possible");
            b.iter(|| {
                let t = create_and_fill_octree::<ChuChunk>(num_chunks, depth);
                black_box(t);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, tree_creation, tree_iteration);
criterion_main!(benches);
