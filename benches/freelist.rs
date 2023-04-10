//#![feature(generic_const_exprs)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use freelist as libfreelist;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use spatialtree::freelist::FreeList;
use std::ops::Index;

use slab::Slab;

trait FlHarness {
    fn new() -> Self;
    fn add(&mut self, d: u32) -> usize;
    fn erase(&mut self, idx: usize);
    fn get_value(&self, index: usize) -> &u32;
}

impl FlHarness for Slab<u32> {
    fn new() -> Self {
        Slab::with_capacity(4)
    }

    fn add(&mut self, d: u32) -> usize {
        self.insert(d)
    }

    fn erase(&mut self, idx: usize) {
        self.remove(idx);
    }

    fn get_value(&self, index: usize) -> &u32 {
        &self[index]
    }
}

impl FlHarness for libfreelist::FreeList<u32> {
    fn new() -> Self {
        libfreelist::FreeList::new()
    }

    fn add(&mut self, d: u32) -> usize {
        self.add(d).get()
    }

    fn erase(&mut self, idx: usize) {
        self.remove(libfreelist::Idx::new(idx).unwrap())
    }
    fn get_value(&self, index: usize) -> &u32 {
        self.index(libfreelist::Idx::new(index).unwrap())
    }
}

impl<DT: spatialtree::Indexer> FlHarness for FreeList<u32, DT> {
    fn new() -> Self {
        FreeList::with_capacity(4)
    }

    fn add(&mut self, d: u32) -> usize {
        self.add(d) as usize
    }

    fn erase(&mut self, idx: usize) {
        self.remove(idx);
    }
    fn get_value(&self, index: usize) -> &u32 {
        self.index(index)
    }
}

fn fill_freelist<FL: FlHarness>(n: usize, rng: &mut SmallRng) -> FL {
    let mut lst = FL::new();
    for _ in 0..n {
        let v: u32 = rng.gen_range(0..1024);
        lst.add(v);
    }
    lst
}

fn freelist_eval<FL>(c: &mut Criterion, title: &str)
where
    FL: FlHarness,
{
    let mut group = c.benchmark_group(title);
    let mut rng = SmallRng::seed_from_u64(42);
    let samples_num = 10;

    for depth in [50 /* 1000, 100000*/].iter() {
        group.significance_level(0.1).sample_size(samples_num);
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            b.iter(|| {
                let mut lst: FL = fill_freelist(depth, &mut rng);
                let mut x: u32 = 0;
                for n in 1..depth {
                    x += lst.get_value(n);
                    lst.erase(n);
                }

                for n in 0..depth {
                    lst.add(n as u32);
                }
                for n in 1..depth {
                    x += lst.get_value(n);
                    lst.erase(n);
                }
                for n in 0..depth {
                    lst.add(n as u32);
                }
                black_box(lst);
            });
        });
    }
    group.finish();
}

pub fn freelist_library(c: &mut Criterion) {
    freelist_eval::<libfreelist::FreeList<u32>>(c, "freelist library");
}

pub fn freelist_inhouse(c: &mut Criterion) {
    freelist_eval::<FreeList<u32, isize>>(c, "freelist inhouse isize");
}

pub fn freelist_slab(c: &mut Criterion) {
    freelist_eval::<Slab<u32>>(c, "freelist slab");
}

criterion_group!(benches, freelist_library, freelist_inhouse, freelist_slab);
criterion_main!(benches);
