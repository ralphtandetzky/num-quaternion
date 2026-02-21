use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use num_quaternion::UQ32;
use rand::Rng;
use rand::SeedableRng;
use std::hint::black_box;

/// Generates a vector of random unit quaternions.
fn generate_random_unit_quaternions(count: usize) -> Vec<UQ32> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0x7F0829AE4D31C6B5);
    (0..count).map(|_| rng.random()).collect()
}

pub fn bench_to_rotation_vector(c: &mut Criterion) {
    // We'll process a batch of unit quaternions in each iteration to measure
    // throughput accurately.
    const BATCH_SIZE: usize = 2_000;

    // Pre-generate the random input data.
    let inputs = generate_random_unit_quaternions(BATCH_SIZE);

    // Create a buffer for the results to ensure the compiler can't optimize
    // away the work.
    let mut outputs = vec![[0.0f32; 3]; BATCH_SIZE];

    let mut group = c.benchmark_group("batch_conversion");

    // Set the measurement to be based on the number of elements processed.
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_function("UQ32::to_rotation_vector", |b| {
        b.iter(|| {
            // The code inside this closure is what's measured.
            for i in 0..BATCH_SIZE {
                let input_quat = black_box(&inputs[i]);
                outputs[i] = input_quat.to_rotation_vector();
            }
            black_box(&mut outputs);
        })
    });

    group.bench_function(
        "UQ32::to_rotation_vector_impl_f32eps (internal implementation)",
        |b| {
            b.iter(|| {
                // The code inside this closure is what's measured.
                for i in 0..BATCH_SIZE {
                    let input_quat = black_box(&inputs[i]);
                    outputs[i] = input_quat.to_rotation_vector_impl_f32eps();
                }
                black_box(&mut outputs);
            })
        },
    );

    group.bench_function(
        "UQ32::to_rotation_vector_impl_generic (internal implementation)",
        |b| {
            b.iter(|| {
                // The code inside this closure is what's measured.
                for i in 0..BATCH_SIZE {
                    let input_quat = black_box(&inputs[i]);
                    outputs[i] = input_quat.to_rotation_vector_impl_generic();
                }
                black_box(&mut outputs);
            })
        },
    );

    group.bench_function("quaternion_core::to_rotation_vector", |b| {
        // Convert UQ32 to quaternion_core's quaternion type
        let core_inputs: Vec<_> = inputs
            .iter()
            .map(|q| {
                let q = q.as_quaternion();
                (q.w, [q.x, q.y, q.z])
            })
            .collect();

        b.iter(|| {
            for i in 0..BATCH_SIZE {
                let input_quat = black_box(&core_inputs[i]);
                outputs[i] = quaternion_core::to_rotation_vector(*input_quat);
            }
            black_box(&mut outputs);
        })
    });

    group.bench_function(
        "nalgebra::geometry::UnitQuaternion::scaled_axis",
        |b| {
            // Convert UQ32 to nalgebra's unit quaternion type
            let nalgebra_inputs: Vec<_> = inputs
                .iter()
                .map(|q| {
                    let q = q.as_quaternion();
                    nalgebra::UnitQuaternion::from_quaternion(
                        nalgebra::Quaternion::new(q.w, q.x, q.y, q.z),
                    )
                })
                .collect();

            let mut nalgebra_outputs =
                vec![nalgebra::Vector3::zeros(); BATCH_SIZE];

            b.iter(|| {
                for i in 0..BATCH_SIZE {
                    let input_quat = black_box(&nalgebra_inputs[i]);
                    nalgebra_outputs[i] = input_quat.scaled_axis();
                }
                black_box(&mut nalgebra_outputs);
            })
        },
    );

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.01).sample_size(2000);
    targets = bench_to_rotation_vector
}

criterion_main!(benches);
