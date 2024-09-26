const NUM_SAMPLES: usize = 1000000;

fn norm_num_quaternion_f32(w: f32, x: f32, y: f32, z: f32) -> f32 {
    num_quaternion::Q32::new(w, x, y, z).norm()
}

fn norm_manual_f32(w: f32, x: f32, y: f32, z: f32) -> f32 {
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
    let norm_funcs: [(NormFunc, &'static str); 6] = [
        (norm_num_quaternion_f32, "norm_num_quaternion_f32"),
        (norm_manual_f32, "norm_manual_f32"),
        (norm_quaternion_f32, "norm_quaternion_f32"),
        (norm_quaternion_core_f32, "norm_quaternion_core_f32"),
        (norm_nalgebra_f32, "norm_nalgebra_f32"),
        (norm_micromath_f32, "norm_micromath_f32"),
    ];

    for (norm_func, name) in norm_funcs {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x7F0829AE4D31C6B5);
        let mut sum_sqr_error = 0.0;
        for _ in 0..NUM_SAMPLES {
            let w = rng.gen_range(-1.0..1.0);
            let x = rng.gen_range(-1.0..1.0);
            let y = rng.gen_range(-1.0..1.0);
            let z = rng.gen_range(-1.0..1.0);
            let norm_f32 = norm_func(w, x, y, z);
            let norm_f64 = ((w as f64).powi(2)
                + (x as f64).powi(2)
                + (y as f64).powi(2)
                + (z as f64).powi(2))
            .sqrt();
            sum_sqr_error += (norm_f32 as f64 - norm_f64).powi(2);
        }
        let mean_sqr_error = sum_sqr_error / NUM_SAMPLES as f64;
        let rms_error = mean_sqr_error.sqrt();
        println!(
            "{:25} RMS error: {:.5} epsilons",
            name,
            rms_error / f32::EPSILON as f64
        );
    }
}
