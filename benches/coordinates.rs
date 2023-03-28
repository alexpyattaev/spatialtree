//#![feature(generic_const_exprs)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use lodtree::coords::*;


trait Harness{
    fn new()->Self;
    fn getchild(&self, idx:usize)->Self;
    fn containschild(&self, c:Self)->bool;
    fn max_children(&self)->usize;
}

impl <const N:usize, DT> Harness for CoordVec<N, DT> where  DT:ReasonableIntegerLike{
    fn new()->Self {
        CoordVec::<N, DT>::new([DT::fromusize(0);N], 8)
    }

    fn getchild(&self, idx:usize)->Self {
        self.get_child(idx)
    }

    fn containschild(&self, c:Self)->bool {
        self.contains_child_node(c)
    }

    fn max_children(&self)->usize {
        <Self as LodVec<N>>::MAX_CHILDREN
    }
}



fn run_eval<FL>(c: &mut Criterion, title: &str)
where FL:Harness+Copy
{
    let mut group = c.benchmark_group(title);
    let mut rng = SmallRng::seed_from_u64(42);
    let samples_num = 10;

    //for depth in [1, 4, 16].iter() {
    for depth in [1,].iter() {
        group.significance_level(0.1).sample_size(samples_num);
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {

            b.iter(|| {
                let x: FL = FL::new();
                let c = x.getchild(rng.gen_range(0..x.max_children()));
                let b = x.containschild(c);
                let d = c.containschild(x);
                black_box(c);
                black_box(d);
                black_box(b);
            });
        });
    }
    group.finish();
}


pub fn generic_trait(c: &mut Criterion) {
    run_eval::<QuadVec>(c,"coords generic_trait quad");
    run_eval::<OctVec>(c,"coords generic_trait oct");
}

pub fn normal_struct(c: &mut Criterion) {
    //run_eval::<QuadVec>(c,"coords normal_struct quad");
    //run_eval::<OctVec>(c,"coords normal_struct oct");
}


criterion_group!(coordinates,normal_struct,generic_trait);
criterion_main!(coordinates);
