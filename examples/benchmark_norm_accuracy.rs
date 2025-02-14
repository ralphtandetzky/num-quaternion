use rand::Rng;
use rand::SeedableRng;
use std::io::{self, Write};

const NUM_SAMPLES: usize = 10000000;

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
        (norm_num_quaternion_f32, "num_quaternion::Q32::norm"),
        (
            norm_num_quaternion_fast_f32,
            "num_quaternion::Q32::fast_norm",
        ),
        (norm_manual_f32, "hypot implementation"),
        (norm_manual_fast_f32, "sqrt(a² + b² + c² + d²)"),
        (norm_quaternion_f32, "quaternion::len"),
        (norm_quaternion_core_f32, "quaternion_core::norm"),
        (norm_nalgebra_f32, "nalgebra::...::Quaternion::norm"),
        (norm_micromath_f32, "micromath::Quaternion::magnitude"),
    ];
    let func_space =
        norm_funcs.iter().map(|(_, name)| name.len()).max().unwrap();
    let col_width = 13;

    println!(
        "Benchmarking the relative accuracy of quaternion norm implementations for different scales of the"
    );
    println!("input quaternion.\n");
    println!(
        "{1:func_space$} | {2:^0$} | {3:^0$} | {4:^0$} | {5:^0$}",
        col_width,
        "Implementation \\ Scale",
        "1.0",
        "sqrt(MIN_POS)",
        "MIN_POS",
        "MAX / 2"
    );
    print!(
        "{1:=<func_space$}=|={1:=<0$}=|={1:=<0$}=|={1:=<0$}=|={1:=<0$}",
        col_width, ""
    );

    let scales = [
        1.0,
        f32::MIN_POSITIVE.sqrt(),
        f32::MIN_POSITIVE,
        f32::MAX / 2.0,
    ];

    for (norm_impl, impl_name) in norm_funcs {
        print!("\n{:func_space$}", impl_name);
        for scale in scales.into_iter() {
            let mut rng =
                rand::rngs::SmallRng::seed_from_u64(0x7F0829AE4D31C6B5);
            let mut sum_sqr_error = 0.0;
            for _ in 0..NUM_SAMPLES {
                let w = rng.random_range(-scale..scale);
                let x = rng.random_range(-scale..scale);
                let y = rng.random_range(-scale..scale);
                let z = rng.random_range(-scale..scale);
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
            print!(
                " | \x1b[{color_code}m{formatted_rms_error:>col_width$}\x1b[0m"
            );
            io::stdout().flush().unwrap();
        }
    }
    println!("\n\nThe columns of the table determine the scale of the input quaternion.");
    println!("The rows of the table determine the implementation of the quaternion norm.");
    println!("The values in the table are the relative RMS error of the quaternion norm.");
    println!("\nThe column `1.0` is for quaternions with all components uniformly sampled from the range [-1.0, 1.0].");
    println!("The column `sqrt(MIN_POS)` is for quaternions with all components in the range [sqrt(MIN_POS), sqrt(MIN_POS)],");
    println!("where `MIN_POS` is the minimal positive normal 32-bit IEEE-754 floating point value. Similarly for `MIN_POS`");
    println!("and `MAX / 2`, where `MAX` is the maximal finite `f32` value.");
}
