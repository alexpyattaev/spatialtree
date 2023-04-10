//#![feature(generic_const_exprs)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use spatialtree::coords::*;

fn run_eval<const N: usize, FL: LodVec<N>>(c: &mut Criterion, title: &str) {
    let mut group = c.benchmark_group(title);
    let mut rng = SmallRng::seed_from_u64(42);
    let samples_num = 10;

    //for depth in [1, 4, 16].iter() {
    for depth in [1].iter() {
        group.significance_level(0.1).sample_size(samples_num);
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            b.iter(|| {
                let x: FL = FL::root();
                let c = x.get_child(rng.gen_range(0..FL::MAX_CHILDREN));
                let b = x.contains_child_node(c);
                let d = c.contains_child_node(x);
                black_box(c);
                black_box(d);
                black_box(b);
            });
        });
    }
    group.finish();
}

pub fn using_u8(c: &mut Criterion) {
    run_eval::<2, QuadVec>(c, "coords_quadvec_u8");
    run_eval::<3, OctVec>(c, "coords_octvec_u8");
}

pub fn using_u16(c: &mut Criterion) {
    run_eval::<2, QuadVec<u16>>(c, "coords_quadvec_u16");
    run_eval::<3, OctVec<u16>>(c, "coords_octvec_u16");
}

criterion_group!(coordinates, using_u8, using_u16);
criterion_main!(coordinates);
