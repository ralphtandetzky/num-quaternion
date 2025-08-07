use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use num_quaternion::{UQ32, UQ64};
use rand::Rng;
use rand::SeedableRng;
use std::f32::consts::PI;
use std::hint::black_box;

/// Generates a vector of random 3D vectors where the norm of each is < PI.
fn generate_random_vectors(count: usize) -> Vec<[f32; 3]> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0x7F0829AE4D31C6B5);
    (0..count)
        .map(|_| {
            // Use rejection sampling to ensure norm < PI
            loop {
                let v: [f32; 3] = [
                    rng.random_range(-PI..PI),
                    rng.random_range(-PI..PI),
                    rng.random_range(-PI..PI),
                ];
                // Check using norm squared to avoid a sqrt call
                if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) < PI * PI {
                    return v;
                }
            }
        })
        .collect()
}

pub fn bench_from_rotation_vector(c: &mut Criterion) {
    {
        // We'll process a batch of vectors in each iteration to measure throughput accurately.
        const BATCH_SIZE: usize = 2_000;

        // Pre-generate the random input data.
        let inputs = generate_random_vectors(BATCH_SIZE);

        // Create a buffer for the results to ensure the compiler can't optimize away the work.
        let mut outputs = vec![UQ32::default(); BATCH_SIZE];

        let mut group = c.benchmark_group("batch_conversion");

        // Set the measurement to be based on the number of elements processed.
        group.throughput(Throughput::Elements(BATCH_SIZE as u64));

        // Define the benchmark. `b.iter` runs the provided closure multiple times.
        group.bench_function("UQ32::from_rotation_vector", |b| {
            b.iter(|| {
                // The code inside this closure is what's measured.
                for i in 0..BATCH_SIZE {
                    // Use `black_box` on the input to prevent the compiler from making assumptions.
                    let input_vec = black_box(&inputs[i]);
                    outputs[i] = UQ32::from_rotation_vector(input_vec);
                }
                // Use `black_box` on the output to ensure the write to the vector is not optimized away.
                black_box(&mut outputs);
            })
        });
        group.bench_function(
            "UQ32::from_rotation_vector_f32_polynomial",
            |b| {
                b.iter(|| {
                    // The code inside this closure is what's measured.
                    for i in 0..BATCH_SIZE {
                        // Use `black_box` on the input to prevent the compiler from making assumptions.
                        let input_vec = black_box(&inputs[i]);
                        let sqr_norm = input_vec[0] * input_vec[0]
                            + input_vec[1] * input_vec[1]
                            + input_vec[2] * input_vec[2];
                        outputs[i] = UQ32::from_rotation_vector_f32_polynomial(
                            input_vec, sqr_norm,
                        );
                    }
                    // Use `black_box` on the output to ensure the write to the vector is not optimized away.
                    black_box(&mut outputs);
                })
            },
        );
        group.bench_function("UQ32::from_rotation_vector_generic", |b| {
            b.iter(|| {
                // The code inside this closure is what's measured.
                for i in 0..BATCH_SIZE {
                    // Use `black_box` on the input to prevent the compiler from making assumptions.
                    let input_vec = black_box(&inputs[i]);
                    let sqr_norm = input_vec[0] * input_vec[0]
                        + input_vec[1] * input_vec[1]
                        + input_vec[2] * input_vec[2];
                    outputs[i] =
                        UQ32::from_rotation_vector_generic(input_vec, sqr_norm);
                }
                // Use `black_box` on the output to ensure the write to the vector is not optimized away.
                black_box(&mut outputs);
            })
        });

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.01).sample_size(2000);
    targets = bench_from_rotation_vector
}

criterion_main!(benches);
