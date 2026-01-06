use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_kernels(c: &mut Criterion) {
    // Benchmarks will be added as kernels are implemented
    let mut group = c.benchmark_group("kernels");
    group.finish();
}

criterion_group!(benches, benchmark_kernels);
criterion_main!(benches);
