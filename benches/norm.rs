use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_quaternion::{Q32, Q64};

pub fn bench_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("Q32 norm");
    group.significance_level(0.01);
    group.sample_size(1000);
    group.bench_function("Q32::norm", |b| {
        b.iter(|| black_box(Q32::new(1.0, 2.0, 3.0, 4.0)).norm())
    });
    group.bench_function("Q32::fast_norm: ", |b| {
        b.iter(|| black_box(Q32::new(1.0, 2.0, 3.0, 4.0)).fast_norm())
    });
    group.bench_function("Manual Q32 norm: ", |b| {
        b.iter(|| {
            let q = black_box(Q32::new(1.0, 2.0, 3.0, 4.0));
            q.w.hypot(q.x).hypot(q.y.hypot(q.z))
        })
    });
    group.bench_function("Manual Q32 fast norm: ", |b| {
        b.iter(|| {
            let q = black_box(Q32::new(1.0, 2.0, 3.0, 4.0));
            (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt()
        })
    });

    group.bench_function("Q64::norm", |b| {
        b.iter(|| black_box(Q64::new(1.0, 2.0, 3.0, 4.0)).norm())
    });
    group.bench_function("Q64::fast_norm: ", |b| {
        b.iter(|| black_box(Q64::new(1.0, 2.0, 3.0, 4.0)).fast_norm())
    });
    group.bench_function("Manual Q64 norm: ", |b| {
        b.iter(|| {
            let q = black_box(Q64::new(1.0, 2.0, 3.0, 4.0));
            q.w.hypot(q.x).hypot(q.y.hypot(q.z))
        })
    });
    group.bench_function("Manual Q64 fast norm: ", |b| {
        b.iter(|| {
            let q = black_box(Q64::new(1.0, 2.0, 3.0, 4.0));
            (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt()
        })
    });
}

criterion_group!(benches, bench_norm);
criterion_main!(benches);
