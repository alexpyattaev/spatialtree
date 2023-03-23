#![feature(generic_const_exprs)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use freelist::{FreeList, Idx};
use std::ops::Index;

struct VecFreeList<T>{
    data: Vec<T>,
    empty: Vec<u16>,
}


trait FlHarness{
    fn new()->Self;
    fn add(&mut self, d:u32)->usize;
    fn erase(&mut self, idx:usize);
    fn get_value(&self, index: Idx)->&u32;
}

impl FlHarness for FreeList<u32>
{
    fn new()->Self {
        FreeList::new()
    }

    fn add(&mut self,d:u32)->usize {
        self.add(d).get()
    }

    fn erase(&mut self,idx:usize) {
        self.remove(Idx::new(idx).unwrap())
    }
    fn get_value(&self, index: Idx) -> &u32 {
        self.index(index)
    }

}

impl FlHarness for VecFreeList<u32>
{
    fn new()->Self {
        Self{data:vec![], empty:vec![]}
    }

    fn add(&mut self,d:u32)->usize {
        match self.empty.pop(){
            Some(idx)=>{
                self.data[idx as usize] = d;
                idx as usize
            },
            None=>{
                self.data.push(d);
                self.data.len()
            }
        }
    }

    fn erase(&mut self,idx:usize) {
        self.empty.push(idx as u16)
    }
    fn get_value(&self, index: Idx) -> &u32 {
        &self.data[index.get()]
    }
}

fn fill_freelist<FL:FlHarness>(n:usize, rng:&mut SmallRng)->FL
{
    let mut lst = FL::new();
    for _ in 0..n{
        let v:u32 = rng.gen_range(0..1024);
        lst.add(v);
    }
    lst
}

fn freelist_eval<FL>(c: &mut Criterion, title: &str)
where FL:FlHarness
{
    let mut group = c.benchmark_group(title);
    let mut rng = SmallRng::seed_from_u64(42);
    let samples_num = 10;

    for depth in [48, 1024, 4096*64].iter() {

        group.significance_level(0.1).sample_size(samples_num);
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {

            b.iter(|| {
                let mut lst: FL = fill_freelist(depth, &mut rng);
                let mut x:u32 = 0;
                for n in 1..depth {
                    x += lst.get_value(Idx::new(n).unwrap());
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
    freelist_eval::<FreeList<u32>>(c,"freelist library");
}

pub fn freelist_vec(c: &mut Criterion) {
    freelist_eval::<VecFreeList<u32>>(c,"freelist vec");
}


criterion_group!(benches,freelist_library,freelist_vec);
criterion_main!(benches);
