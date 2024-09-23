use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_quaternion::{Q32, Q64};

pub fn bench_norm(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("num_quaternion");
        group.bench_function("Q32::norm", |b| {
            b.iter(|| black_box(Q32::new(1.0, 2.0, 3.0, 4.0)).norm())
        });
        group.bench_function("Q32::fast_norm: ", |b| {
            b.iter(|| black_box(Q32::new(1.0, 2.0, 3.0, 4.0)).fast_norm())
        });
        group.bench_function("Q64::norm", |b| {
            b.iter(|| black_box(Q64::new(1.0, 2.0, 3.0, 4.0)).norm())
        });
        group.bench_function("Q64::fast_norm: ", |b| {
            b.iter(|| black_box(Q64::new(1.0, 2.0, 3.0, 4.0)).fast_norm())
        });
    }
    {
        let mut group = c.benchmark_group("Manual implementation");
        group.bench_function("f32 norm: ", |b| {
            b.iter(|| {
                let q = black_box(Q32::new(1.0, 2.0, 3.0, 4.0));
                q.w.hypot(q.x).hypot(q.y.hypot(q.z))
            })
        });
        group.bench_function("f32 fast norm: ", |b| {
            b.iter(|| {
                let q = black_box(Q32::new(1.0, 2.0, 3.0, 4.0));
                (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt()
            })
        });
        group.bench_function("f64 norm: ", |b| {
            b.iter(|| {
                let q = black_box(Q64::new(1.0, 2.0, 3.0, 4.0));
                q.w.hypot(q.x).hypot(q.y.hypot(q.z))
            })
        });
        group.bench_function("f64 fast norm: ", |b| {
            b.iter(|| {
                let q = black_box(Q64::new(1.0, 2.0, 3.0, 4.0));
                (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt()
            })
        });
    }
    {
        let mut group = c.benchmark_group("quaternion");
        group.bench_function("len<f32>", |b| {
            b.iter(|| {
                quaternion::len(black_box((1.0f32, [2.0f32, 3.0f32, 4.0f32])))
            })
        });
        group.bench_function("len<64>", |b| {
            b.iter(|| {
                quaternion::len(black_box((1.0f64, [2.0f64, 3.0f64, 4.0f64])))
            })
        });
    }

    {
        let mut group = c.benchmark_group("quaternion-core");
        group.bench_function("norm<f32>", |b| {
            b.iter(|| {
                quaternion_core::norm(black_box((
                    1.0f32,
                    [2.0f32, 3.0f32, 4.0f32],
                )))
            })
        });
        group.bench_function("norm<f64>", |b| {
            b.iter(|| {
                quaternion_core::norm(black_box((
                    1.0f64,
                    [2.0f64, 3.0f64, 4.0f64],
                )))
            })
        });
    }

    {
        let mut group = c.benchmark_group("nalgebra");
        group.bench_function("geometry::Quaternion<f32>::norm", |b| {
            b.iter(|| {
                black_box(nalgebra::geometry::Quaternion::new(
                    1.0f32, 2.0, 3.0, 4.0,
                ))
                .norm()
            })
        });
        group.bench_function("geometry::Quaternion<f64>::norm", |b| {
            b.iter(|| {
                black_box(nalgebra::geometry::Quaternion::new(
                    1.0f64, 2.0, 3.0, 4.0,
                ))
                .norm()
            })
        });
    }

    {
        let mut group = c.benchmark_group("micromath");
        group.bench_function("Quaternion::norm", |b| {
            b.iter(|| {
                black_box(micromath::Quaternion::new(1.0f32, 2.0, 3.0, 4.0))
                    .magnitude()
            })
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(1000);
    targets = bench_norm
}

criterion_main!(benches);
