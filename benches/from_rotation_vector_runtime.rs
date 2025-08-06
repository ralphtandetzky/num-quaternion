use criterion::{criterion_group, criterion_main, Criterion};
use num_quaternion::{UQ32, UQ64};
use std::hint::black_box;

pub fn bench_from_rotation_vector(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("num_quaternion");
        group.bench_function("UQ32::from_rotation_vector", |b| {
            b.iter(|| black_box(UQ32::from_rotation_vector(&[1.0, 2.0, 0.5])))
        });
        group.bench_function("UQ64::from_rotation_vector", |b| {
            b.iter(|| black_box(UQ64::from_rotation_vector(&[1.0, 2.0, 0.5])))
        });
    }
    {
        let mut group = c.benchmark_group("f32 polynomial");
        group.bench_function(
            "UQ32::from_rotation_vector_f32_polynomial",
            |b| {
                b.iter(|| {
                    black_box(UQ32::from_rotation_vector_f32_polynomial(
                        &[1.0, 2.0, 0.5],
                        5.25,
                    ))
                })
            },
        );
        group.bench_function(
            "UQ64::from_rotation_vector_f32_polynomial",
            |b| {
                b.iter(|| {
                    black_box(UQ64::from_rotation_vector_f32_polynomial(
                        &[1.0, 2.0, 0.5],
                        5.25,
                    ))
                })
            },
        );
    }
    {
        let mut group = c.benchmark_group("generic implementation");
        group.bench_function("UQ32::from_rotation_vector_generic", |b| {
            b.iter(|| {
                black_box(UQ32::from_rotation_vector_generic(
                    &[1.0, 2.0, 0.5],
                    5.25,
                ))
            })
        });
        group.bench_function("UQ64::from_rotation_vector_generic", |b| {
            b.iter(|| {
                black_box(UQ64::from_rotation_vector_generic(
                    &[1.0, 2.0, 0.5],
                    5.25,
                ))
            });
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.01).sample_size(2000);
    targets = bench_from_rotation_vector
}

criterion_main!(benches);
