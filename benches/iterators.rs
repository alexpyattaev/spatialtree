use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use rand::rngs::SmallRng;
use rand::SeedableRng;
use spatialtree::coords::*;
use spatialtree::*;

const N_LOOKUPS: usize = 40;
fn generate_area_bounds(rng: &mut SmallRng) -> (OctVec, OctVec) {
    const D: u8 = 4;
    let cmax = 1 << D;

    let min = rand_cv(
        rng,
        OctVec::new([0, 0, 0], D),
        OctVec::new([cmax - 2, cmax - 2, cmax - 2], D),
    );
    let max = rand_cv(
        rng,
        min + OctVec::new([1, 1, 1], D),
        OctVec::new([cmax, cmax, cmax], D),
    );

    return (min, max);
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

    let cmax = 1 << depth;

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

fn bench_lookups_in_octree(tree: &OctTree<ChuChunk, OctVec>) {
    let mut rng = SmallRng::seed_from_u64(42);
    for _ in 0..N_LOOKUPS {
        let (min, max) = generate_area_bounds(&mut rng);
        for i in tree.iter_chunks_in_aabb(min, max) {
            black_box(i);
        }
    }
}

fn bench_mut_lookups_in_octree(tree: &mut OctTree<ChuChunk, OctVec>) {
    let mut rng = SmallRng::seed_from_u64(42);
    for _ in 0..N_LOOKUPS {
        let (min, max) = generate_area_bounds(&mut rng);
        for i in tree.iter_chunks_in_aabb_mut(min, max) {
            i.1.material_index += 1;
            i.1.a_index += 1;
            i.1.b_index += 1;
        }
    }
}

pub fn tree_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutable iteration");
    let mut samples_num = 100;

    for depth in [4u8, 6, 8, 10].iter() {
        if *depth as i8 == 4 {
            samples_num = 100;
        }
        if *depth as i8 == 6 {
            samples_num = 40;
        }
        if *depth as i8 == 8 {
            samples_num = 10;
        }
        group.significance_level(0.1).sample_size(samples_num);

        let num_chunks: u32 = 2u32.pow(*depth as u32).pow(3) / 10;
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, depth| {
            let mut tree = create_and_fill_octree::<ChuChunk>(num_chunks, *depth);
            b.iter(|| {
                bench_mut_lookups_in_octree(&mut tree);
            });
            black_box(tree);
        });
    }
    group.finish();

    let mut group = c.benchmark_group("immutable iteration");
    let mut samples_num = 10;

    for depth in [4u8, 6, 8, 10].iter() {
        if *depth as i8 == 4 {
            samples_num = 100;
        }
        if *depth as i8 == 6 {
            samples_num = 40;
        }
        if *depth as i8 == 8 {
            samples_num = 10;
        }
        group.significance_level(0.1).sample_size(samples_num);
        let num_chunks: u32 = 2u32.pow(*depth as u32).pow(3) / 10;
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, depth| {
            let tree = create_and_fill_octree::<ChuChunk>(num_chunks, *depth);
            b.iter(|| {
                bench_lookups_in_octree(&tree);
            });
        });
    }
    group.finish();
}

pub fn tree_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree creation");

    let mut samples_num = 10;

    for depth in [4u8, 6, 8].iter() {
        if *depth as i8 == 4 {
            samples_num = 100;
        }
        if *depth as i8 == 6 {
            samples_num = 40;
        }
        if *depth as i8 == 8 {
            samples_num = 10;
        }
        group.significance_level(0.1).sample_size(samples_num);
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            let volume = 2u32.pow(depth as u32).pow(3);
            let num_chunks: u32 = volume / 10;
            println!("Creating {num_chunks} voxels out of {volume} possible");
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
