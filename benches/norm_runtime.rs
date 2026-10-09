use algebraic::af32;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use num_quaternion::{Q32, Q64, Quaternion};
use num_traits::Float;
use rand::RngExt;
use rand::SeedableRng;
use std::hint::black_box;

/// Generates random quaternions once, before the benchmark starts.
fn generate_random_quaternions(count: usize) -> Vec<[f32; 4]> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0x7F0829AE4D31C6B5);
    (0..count)
        .map(|_| {
            [
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ]
        })
        .collect()
}

pub fn bench_norm(c: &mut Criterion) {
    const BATCH_SIZE: usize = 2_000;

    let components = generate_random_quaternions(BATCH_SIZE);
    let q32_inputs = components
        .iter()
        .map(|q| Q32::new(q[0], q[1], q[2], q[3]))
        .collect::<Vec<_>>();
    let q64_inputs = components
        .iter()
        .map(|q| Q64::new(q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64))
        .collect::<Vec<_>>();
    let af32_inputs = components
        .iter()
        .map(|q| {
            Quaternion::new(
                af32::from(q[0]),
                af32::from(q[1]),
                af32::from(q[2]),
                af32::from(q[3]),
            )
        })
        .collect::<Vec<_>>();
    let tuple_f32_inputs = components
        .iter()
        .map(|q| (q[0], [q[1], q[2], q[3]]))
        .collect::<Vec<_>>();
    let tuple_f64_inputs = components
        .iter()
        .map(|q| (q[0] as f64, [q[1] as f64, q[2] as f64, q[3] as f64]))
        .collect::<Vec<_>>();
    let nalgebra_f32_inputs = components
        .iter()
        .map(|q| nalgebra::geometry::Quaternion::new(q[0], q[1], q[2], q[3]))
        .collect::<Vec<_>>();
    let nalgebra_f64_inputs = components
        .iter()
        .map(|q| {
            nalgebra::geometry::Quaternion::new(
                q[0] as f64,
                q[1] as f64,
                q[2] as f64,
                q[3] as f64,
            )
        })
        .collect::<Vec<_>>();
    let micromath_inputs = components
        .iter()
        .map(|q| micromath::Quaternion::new(q[0], q[1], q[2], q[3]))
        .collect::<Vec<_>>();

    {
        let mut group = c.benchmark_group("num_quaternion");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));
        group.bench_function("Q32::norm", |b| {
            b.iter(|| {
                for q in &q32_inputs {
                    black_box(black_box(q).norm());
                }
            })
        });
        group.bench_function("Q32::fast_norm", |b| {
            b.iter(|| {
                for q in &q32_inputs {
                    black_box(black_box(q).fast_norm());
                }
            })
        });
        group.bench_function("Quaternion<af32>::norm", |b| {
            b.iter(|| {
                for q in &af32_inputs {
                    black_box(black_box(q).norm());
                }
            })
        });
        group.bench_function("Quaternion<af32>::fast_norm", |b| {
            b.iter(|| {
                for q in &af32_inputs {
                    black_box(black_box(q).fast_norm());
                }
            })
        });
        group.bench_function("Q64::norm", |b| {
            b.iter(|| {
                for q in &q64_inputs {
                    black_box(black_box(q).norm());
                }
            })
        });
        group.bench_function("Q64::fast_norm", |b| {
            b.iter(|| {
                for q in &q64_inputs {
                    black_box(black_box(q).fast_norm());
                }
            })
        });
    }
    {
        let mut group = c.benchmark_group("Manual implementation");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));
        group.bench_function("f32 norm", |b| {
            b.iter(|| {
                for q in &q32_inputs {
                    let q = black_box(q);
                    black_box(q.w.hypot(q.x).hypot(q.y.hypot(q.z)));
                }
            })
        });
        group.bench_function("f32 fast norm", |b| {
            b.iter(|| {
                for q in &q32_inputs {
                    let q = black_box(q);
                    black_box(
                        (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt(),
                    );
                }
            })
        });
        group.bench_function("af32 norm", |b| {
            b.iter(|| {
                for q in &af32_inputs {
                    let q = black_box(q);
                    black_box(q.w.hypot(q.x).hypot(q.y.hypot(q.z)));
                }
            })
        });
        group.bench_function("af32 fast norm", |b| {
            b.iter(|| {
                for q in &af32_inputs {
                    let q = black_box(q);
                    black_box(
                        (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt(),
                    );
                }
            })
        });
        group.bench_function("f64 norm", |b| {
            b.iter(|| {
                for q in &q64_inputs {
                    let q = black_box(q);
                    black_box(q.w.hypot(q.x).hypot(q.y.hypot(q.z)));
                }
            })
        });
        group.bench_function("f64 fast norm", |b| {
            b.iter(|| {
                for q in &q64_inputs {
                    let q = black_box(q);
                    black_box(
                        (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt(),
                    );
                }
            })
        });
    }
    {
        let mut group = c.benchmark_group("quaternion");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));
        group.bench_function("len<f32>", |b| {
            b.iter(|| {
                for q in &tuple_f32_inputs {
                    black_box(quaternion::len(black_box(*q)));
                }
            })
        });
        group.bench_function("len<64>", |b| {
            b.iter(|| {
                for q in &tuple_f64_inputs {
                    black_box(quaternion::len(black_box(*q)));
                }
            })
        });
    }

    {
        let mut group = c.benchmark_group("quaternion-core");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));
        group.bench_function("norm<f32>", |b| {
            b.iter(|| {
                for q in &tuple_f32_inputs {
                    black_box(quaternion_core::norm(black_box(*q)));
                }
            })
        });
        group.bench_function("norm<f64>", |b| {
            b.iter(|| {
                for q in &tuple_f64_inputs {
                    black_box(quaternion_core::norm(black_box(*q)));
                }
            })
        });
    }

    {
        let mut group = c.benchmark_group("nalgebra");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));
        group.bench_function("geometry::Quaternion<f32>::norm", |b| {
            b.iter(|| {
                for q in &nalgebra_f32_inputs {
                    black_box(black_box(q).norm());
                }
            })
        });
        group.bench_function("geometry::Quaternion<f64>::norm", |b| {
            b.iter(|| {
                for q in &nalgebra_f64_inputs {
                    black_box(black_box(q).norm());
                }
            })
        });
    }

    {
        let mut group = c.benchmark_group("micromath");
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));
        group.bench_function("Quaternion::norm", |b| {
            b.iter(|| {
                for q in &micromath_inputs {
                    black_box(black_box(q).magnitude());
                }
            })
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.01).sample_size(2000);
    targets = bench_norm
}

criterion_main!(benches);
