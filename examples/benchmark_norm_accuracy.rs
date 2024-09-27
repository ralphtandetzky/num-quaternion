use rand::Rng;
use rand::SeedableRng;

const NUM_SAMPLES: usize = 1000000;

fn norm_num_quaternion_f32(w: f32, x: f32, y: f32, z: f32) -> f32 {
    num_quaternion::Q32::new(w, x, y, z).norm()
}

fn norm_num_quaternion_fast_f32(w: f32, x: f32, y: f32, z: f32) -> f32 {
    num_quaternion::Q32::new(w, x, y, z).fast_norm()
}

fn norm_manual_f32(w: f32, x: f32, y: f32, z: f32) -> f32 {
    w.hypot(y).hypot(x.hypot(z))
}

fn norm_manual_fast_f32(w: f32, x: f32, y: f32, z: f32) -> f32 {
    (w * w + x * x + y * y + z * z).sqrt()
}

fn norm_quaternion_f32(w: f32, x: f32, y: f32, z: f32) -> f32 {
    quaternion::len((w, [x, y, z]))
}

fn norm_quaternion_core_f32(w: f32, x: f32, y: f32, z: f32) -> f32 {
    quaternion_core::norm((w, [x, y, z]))
}

fn norm_nalgebra_f32(w: f32, x: f32, y: f32, z: f32) -> f32 {
    nalgebra::geometry::Quaternion::new(w, x, y, z).norm()
}

fn norm_micromath_f32(w: f32, x: f32, y: f32, z: f32) -> f32 {
    micromath::Quaternion::new(w, x, y, z).magnitude()
}

fn main() {
    type NormFunc = fn(f32, f32, f32, f32) -> f32;
    let norm_funcs: [(NormFunc, &'static str); 8] = [
        (norm_num_quaternion_f32, "norm_num_quaternion_f32"),
        (norm_num_quaternion_fast_f32, "norm_num_quaternion_fast_f32"),
        (norm_manual_f32, "norm_manual_f32"),
        (norm_manual_fast_f32, "norm_manual_fast_f32"),
        (norm_quaternion_f32, "norm_quaternion_f32"),
        (norm_quaternion_core_f32, "norm_quaternion_core_f32"),
        (norm_nalgebra_f32, "norm_nalgebra_f32"),
        (norm_micromath_f32, "norm_micromath_f32"),
    ];

    let scales = [
        (1.0, "1.0"),
        (f32::MIN_POSITIVE.sqrt(), "sqrt(MIN_POS)"),
        (f32::MIN_POSITIVE, "MIN_POS"),
        (f32::MAX / 2.0, "MAX / 2"),
    ];
    println!("Scale           | Implementation                 | RMS rel. error in epsilons");
    println!("================|================================|===========================");

    for (scale, scale_name) in scales.into_iter() {
        for (norm_impl, impl_name) in norm_funcs {
            let mut rng =
                rand::rngs::SmallRng::seed_from_u64(0x7F0829AE4D31C6B5);
            let mut sum_sqr_error = 0.0;
            for _ in 0..NUM_SAMPLES {
                let w = rng.gen_range(-scale..scale);
                let x = rng.gen_range(-scale..scale);
                let y = rng.gen_range(-scale..scale);
                let z = rng.gen_range(-scale..scale);
                let norm_f32 = norm_impl(w, x, y, z);
                let norm_f64 = ((w as f64).powi(2)
                    + (x as f64).powi(2)
                    + (y as f64).powi(2)
                    + (z as f64).powi(2))
                .sqrt();
                sum_sqr_error += (norm_f32 as f64 / norm_f64 - 1.0).powi(2);
            }
            let mean_sqr_error = sum_sqr_error / NUM_SAMPLES as f64;
            let rms_error = mean_sqr_error.sqrt();
            println!(
                "{:15} | {:30} | {:.4}",
                scale_name,
                impl_name,
                rms_error / f32::EPSILON as f64
            );
        }
    }
}
