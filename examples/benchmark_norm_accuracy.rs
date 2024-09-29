use rand::Rng;
use rand::SeedableRng;
use std::io::{self, Write};

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
        1.0,
        f32::MIN_POSITIVE.sqrt(),
        f32::MIN_POSITIVE,
        f32::MAX / 2.0,
    ];

    println!(
        "Benchmarking the relative accuracy of quaternion norm implementations"
    );
    println!("for different scales of the input quaternion.\n");
    println!(
        "{:28} | {:^13} | {:^13} | {:^13} | {:^13}",
        "Implementation \\ Scale", "1.0", "sqrt(MIN_POS)", "MIN_POS", "MAX / 2"
    );
    print!("{0:=<28}=|={0:=<13}=|={0:=<13}=|={0:=<13}=|={0:=<13}", "");

    for (norm_impl, impl_name) in norm_funcs {
        print!("\n{:28}", impl_name);
        for scale in scales.into_iter() {
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
            let rms_error_in_eps = rms_error / f32::EPSILON as f64;
            let formatted_rms_error =
                if rms_error_in_eps <= 1.0 / f32::EPSILON as f64 {
                    format!("{:.4}", rms_error_in_eps)
                } else {
                    format!("{:.4e}", rms_error_in_eps)
                };
            let color_code = if rms_error_in_eps < 0.3 {
                "92" // green
            } else if rms_error_in_eps < 1.0 {
                "93" // yellow
            } else {
                "91" // red
            };
            print!(" | \x1b[{}m{:>13}\x1b[0m", color_code, formatted_rms_error);
            io::stdout().flush().unwrap();
        }
    }
    println!("\n\nThe columns of the table determine the scale of the input quaternion.");
    println!("The rows of the table determine the implementation of the quaternion norm.");
    println!("The values in the table are the relative RMS error of the quaternion norm.");
    println!("\nThe column `1.0` is for quaternions with all components in the range [-1.0, 1.0].");
    println!("The column `sqrt(MIN_POS)` is for quaternions with all components in the range ");
    println!("[sqrt(MIN_POS), sqrt(MIN_POS)], where `MIN_POS` is the minimal positive normal ");
    println!("floating point value for IEEE 754 floating point values of 32-bit width.");
    println!("Similarly for `MIN_POS` and `MAX / 2`, where `MAX` is the maximal finite value ");
    println!("for IEEE 754 floating point values of 32-bit width.");
}
