use num_quaternion::{Q32, UQ32};
use rand::Rng;
use rand::SeedableRng;
use std::io::{self, Write};

const NUM_SAMPLES: usize = 1_000_000;

/// Measures the accuracy of the round-trip conversion:
/// q -> to_rotation_vector -> from_rotation_vector -> q'
/// Returns the maximum absolute error of the quaternion components.
fn measure_round_trip_error(
    q: UQ32,
    to_vec: fn(UQ32) -> [f32; 3],
    from_vec: fn(&[f32; 3]) -> Q32,
) -> (f64, f64) {
    let q = q.adjust_norm();
    let rotation_vector = to_vec(q);
    let q_reconstructed = from_vec(&rotation_vector);

    let q_diff = q - q_reconstructed;
    let abs_error = q_diff.norm();

    let q_reconstructed = if abs_error < 1.0 {
        q_reconstructed
    } else {
        -q_reconstructed
    };

    let q_diff = q - q_reconstructed;
    let abs_error = q_diff.norm();

    let angle = rotation_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    let abs_error_w = q_diff.w.abs();
    let abs_error_xyz =
        (q_diff.x.powi(2) + q_diff.y.powi(2) + q_diff.z.powi(2)).sqrt();
    let rel_error_xyz = if angle > 0.0 {
        abs_error_xyz / angle
    } else {
        0.0
    };
    let mixed_error = abs_error_w.max(rel_error_xyz);

    (abs_error.into(), mixed_error.into())
}

fn main() {
    println!("Benchmarking the absolute accuracy of rotation vector round-trip conversions");
    println!("(to_rotation_vector followed by from_rotation_vector) for different rotation angles.\n");

    let col_width = 15;

    // Test different crates
    let crates: [(&str, fn(UQ32) -> [f32; 3], fn(&[f32; 3]) -> Q32); _] = [
        (
            "num_quaternion",
            |q| q.to_rotation_vector(),
            |v| UQ32::from_rotation_vector(v).into(),
        ),
        (
            "num_quaternion (gen)",
            |q| q.to_rotation_vector_impl_generic(),
            |v| {
                UQ32::from_rotation_vector_generic(
                    v,
                    v[0] * v[0] + v[1] * v[1] + v[2] * v[2],
                )
                .into()
            },
        ),
        (
            "num_quaternion (f64)",
            |q| {
                let q = q.as_quaternion();
                let q_f64 = num_quaternion::Q64::new(
                    q.w as f64, q.x as f64, q.y as f64, q.z as f64,
                )
                .normalize()
                .unwrap();
                let rv_f64 = q_f64.to_rotation_vector();
                [rv_f64[0] as f32, rv_f64[1] as f32, rv_f64[2] as f32]
            },
            |v| {
                let v_f64 = [v[0] as f64, v[1] as f64, v[2] as f64];
                let q_f64 = *num_quaternion::UQ64::from_rotation_vector(&v_f64)
                    .as_quaternion();
                Q32::new(
                    q_f64.w as f32,
                    q_f64.x as f32,
                    q_f64.y as f32,
                    q_f64.z as f32,
                )
            },
        ),
        (
            "nalgebra",
            |q| {
                let q = q.as_quaternion();
                let na_q = nalgebra::UnitQuaternion::from_quaternion(
                    nalgebra::Quaternion::new(q.w, q.x, q.y, q.z),
                );
                let scaled_axis = na_q.scaled_axis();
                [scaled_axis.x, scaled_axis.y, scaled_axis.z]
            },
            |v| {
                let scaled_axis = nalgebra::Vector3::new(v[0], v[1], v[2]);
                let na_q =
                    nalgebra::UnitQuaternion::from_scaled_axis(scaled_axis);
                let q = na_q.quaternion();
                Q32::new(q.w, q.coords.x, q.coords.y, q.coords.z)
            },
        ),
        (
            "quaternion_core",
            |q| {
                let q = q.as_quaternion();
                quaternion_core::to_rotation_vector((q.w, [q.x, q.y, q.z]))
            },
            |v| {
                let (w, xyz) = quaternion_core::from_rotation_vector(*v);
                Q32::new(w, xyz[0], xyz[1], xyz[2])
            },
        ),
    ];

    println!(
        "{:20} | {:^width$} | {:^width$} | {:^width$} | {:^width$}",
        "Method",
        "Max error",
        "RMS error",
        "Max rel error",
        "RMS rel error",
        width = col_width
    );
    println!(
        "{:=<20}=|={:=<width$}=|={:=<width$}=|={:=<width$}=|={:=<width$}=",
        "",
        "",
        "",
        "",
        "",
        width = col_width
    );

    for (label, to_vec, from_vec) in crates {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x7F0829AE4D31C6B5);

        let mut max_abs_error = 0.0f64;
        let mut sum_sqr_abs_error = 0.0f64;
        let mut max_rel_error = 0.0f64;
        let mut sum_sqr_rel_error = 0.0f64;

        for _ in 0..NUM_SAMPLES {
            // Generate random unit quaternion
            let q = rng.random::<UQ32>();
            let q = if q.as_quaternion().w >= 0.0 { q } else { -q };

            let (abs_error, rel_error) =
                measure_round_trip_error(q, to_vec, from_vec);

            max_abs_error = max_abs_error.max(abs_error);
            sum_sqr_abs_error += (abs_error).powi(2);
            max_rel_error = max_rel_error.max(rel_error);
            sum_sqr_rel_error += (rel_error).powi(2);
        }

        let rms_abs_error = (sum_sqr_abs_error / NUM_SAMPLES as f64).sqrt();
        let rms_rel_error = (sum_sqr_rel_error / NUM_SAMPLES as f64).sqrt();

        // Convert errors to multiples of f32::EPSILON
        let max_abs_error_in_eps = max_abs_error as f64 / f32::EPSILON as f64;
        let rms_abs_error_in_eps = rms_abs_error / f32::EPSILON as f64;
        let max_rel_error_in_eps = max_rel_error as f64 / f32::EPSILON as f64;
        let rms_rel_error_in_eps = rms_rel_error / f32::EPSILON as f64;

        // Format the errors
        let format_error = |error_in_eps: f64| -> String {
            // Color code based on RMS error
            let color_code = if error_in_eps < 1.0 {
                "92" // green
            } else if error_in_eps < 2.0 {
                "93" // yellow
            } else {
                "91" // red
            };
            format!(
                "\x1b[{color_code}m{:>width$.4}\x1b[0m",
                error_in_eps,
                width = col_width
            )
        };

        println!(
            "{:20} | {} | {} | {} | {}",
            label,
            format_error(max_abs_error_in_eps),
            format_error(rms_abs_error_in_eps),
            format_error(max_rel_error_in_eps),
            format_error(rms_rel_error_in_eps),
        );
        io::stdout().flush().unwrap();
    }

    println!("\nThe table shows the accuracy of the round-trip conversion:");
    println!("  q -> to_rotation_vector() -> from_rotation_vector() -> q'");
    println!("\nAll errors are expressed as multiples of f32::EPSILON.");
    println!(
        "The rotation angles are uniformly sampled from the specified ranges."
    );
    println!("The rotation axes are uniformly sampled from the unit sphere.");
}
