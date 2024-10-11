use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_quaternion::{Q32, Q64, UQ32, UQ64};

pub fn bench_from_euler_angles(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("num_quaternion");
        group.bench_function("UQ32::from_euler_angles", |b| {
            b.iter(|| {
                let (roll, pitch, yaw) = black_box((1.0, 2.0, 3.0));
                UQ32::from_euler_angles(roll, pitch, yaw)
            })
        });
        group.bench_function("UQ64::from_euler_angles", |b| {
            b.iter(|| {
                let (roll, pitch, yaw) = black_box((1.0, 2.0, 3.0));
                UQ64::from_euler_angles(roll, pitch, yaw)
            })
        });
    }
    {
        let mut group = c.benchmark_group("Manual implementation");
        group.bench_function("f32 from_euler_angles", |b| {
            b.iter(|| {
                let (yaw, pitch, roll) = black_box((1.0f32, 2.0f32, 3.0f32));
                let half_yaw = yaw * 0.5;
                let half_pitch = pitch * 0.5;
                let half_roll = roll * 0.5;
                let sin_yaw = half_yaw.sin();
                let cos_yaw = half_yaw.cos();
                let sin_pitch = half_pitch.sin();
                let cos_pitch = half_pitch.cos();
                let sin_roll = half_roll.sin();
                let cos_roll = half_roll.cos();
                Q32::new(
                    cos_yaw * cos_pitch * cos_roll
                        + sin_yaw * sin_pitch * sin_roll,
                    sin_yaw * cos_pitch * cos_roll
                        - cos_yaw * sin_pitch * sin_roll,
                    cos_yaw * sin_pitch * cos_roll
                        + sin_yaw * cos_pitch * sin_roll,
                    cos_yaw * cos_pitch * sin_roll
                        - sin_yaw * sin_pitch * cos_roll,
                )
            })
        });
        group.bench_function("f64 from_euler_angles", |b| {
            b.iter(|| {
                let (yaw, pitch, roll) = black_box((1.0f64, 2.0f64, 3.0f64));
                let half_yaw = yaw * 0.5;
                let half_pitch = pitch * 0.5;
                let half_roll = roll * 0.5;
                let sin_yaw = half_yaw.sin();
                let cos_yaw = half_yaw.cos();
                let sin_pitch = half_pitch.sin();
                let cos_pitch = half_pitch.cos();
                let sin_roll = half_roll.sin();
                let cos_roll = half_roll.cos();
                Q64::new(
                    cos_yaw * cos_pitch * cos_roll
                        + sin_yaw * sin_pitch * sin_roll,
                    sin_yaw * cos_pitch * cos_roll
                        - cos_yaw * sin_pitch * sin_roll,
                    cos_yaw * sin_pitch * cos_roll
                        + sin_yaw * cos_pitch * sin_roll,
                    cos_yaw * cos_pitch * sin_roll
                        - sin_yaw * sin_pitch * cos_roll,
                )
            })
        });
    }
    {
        let mut group = c.benchmark_group("quaternion");
        group.bench_function("euler_angles<f32>", |b| {
            b.iter(|| {
                let (roll, pitch, yaw) = black_box((1.0f32, 2.0f32, 3.0f32));
                quaternion::euler_angles(roll, pitch, yaw)
            })
        });
        group.bench_function("euler_angles<f64>", |b| {
            b.iter(|| {
                let (roll, pitch, yaw) = black_box((1.0f64, 2.0f64, 3.0f64));
                quaternion::euler_angles(roll, pitch, yaw)
            })
        });
    }
    {
        let mut group = c.benchmark_group("quaternion-core");
        group.bench_function("from_euler_angles<f32>", |b| {
            b.iter(|| {
                let angles = black_box([1.0f32, 2.0f32, 3.0f32]);
                quaternion_core::from_euler_angles(
                    quaternion_core::RotationType::Intrinsic,
                    quaternion_core::RotationSequence::XYZ,
                    angles,
                )
            })
        });
        group.bench_function("from_euler_angles<f64>", |b| {
            b.iter(|| {
                let angles = black_box([1.0f64, 2.0f64, 3.0f64]);
                quaternion_core::from_euler_angles(
                    quaternion_core::RotationType::Intrinsic,
                    quaternion_core::RotationSequence::XYZ,
                    angles,
                )
            })
        });
    }
}

criterion_group! {
name = benches;
config = Criterion::default().significance_level(0.01).sample_size(2000);
targets = bench_from_euler_angles
}

criterion_main!(benches);
