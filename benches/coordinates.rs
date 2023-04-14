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
use rand::{Rng, SeedableRng};

use spatialtree::coords::*;

fn run_eval<const N: usize, FL: LodVec<N>>(c: &mut Criterion, title: &str) {
    let mut group = c.benchmark_group(title);
    let mut rng = SmallRng::seed_from_u64(42);
    let samples_num = 10;

    //for depth in [1, 4, 16].iter() {
    for depth in [1].iter() {
        group.significance_level(0.1).sample_size(samples_num);
        //TODO: more sensible stuff here
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &_depth| {
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
