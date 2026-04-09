use num_quaternion::{Q32, Q64, UQ32, UQ64};
use rand::RngExt;
use rand::SeedableRng;
use std::io::{self, Write};

const NUM_SAMPLES: usize = 1_000_000;
const SEED: u64 = 0x7F0829AE4D31C6B5;

/// Upcasts a `UQ32` to `UQ64` by converting each component to `f64`.
fn upcast_to_f64(q: UQ32) -> UQ64 {
    let q = q.as_quaternion();
    Q64::new(q.w as f64, q.x as f64, q.y as f64, q.z as f64)
        .normalize()
        .unwrap()
}

/// Computes the backward error `||(Jf)^{-1} * (f_exact(q) - f_approx(q))||`,
/// where `f` is the exact unit quaternion to rotation vector mapping,
/// `f_approx` is the approximation under test, and `Jf` is the Jacobian of
/// `f`.
///
/// The backward error measures how large an input perturbation would be needed
/// to explain the observed output error. If this value is on the order of
/// `f32::EPSILON`, the implementation is essentially as accurate as the input
/// allows.
///
/// The formula for the inverse Jacobian is:
/// ```text
/// (Jf)^{-1} = sin(θ)/θ · I + (1 − sin(θ)/θ) · n̂⊗n̂ − (1−cos(θ))/θ · [n̂]×
/// ```
/// where θ is the rotation angle and n̂ is the rotation axis.
///
/// This is the right-inverse Jacobian of the rotation vector map at `q`,
/// computed with respect to right-quaternion perturbations
/// `q' = q · exp(δ/2)`.
///
/// `UQ64::to_rotation_vector()` is assumed to be exact.
fn backward_error(q: UQ32, r_approx: [f32; 3]) -> f64 {
    // Compute "exact" rotation vector via f64.
    let r_exact = upcast_to_f64(q).to_rotation_vector();

    // Error vector e = f_exact(q) - f_approx(q).
    let e = [
        r_exact[0] - r_approx[0] as f64,
        r_exact[1] - r_approx[1] as f64,
        r_exact[2] - r_approx[2] as f64,
    ];

    // Rotation angle θ from the exact rotation vector.
    let theta_sq =
        r_exact[0] * r_exact[0] + r_exact[1] * r_exact[1] + r_exact[2] * r_exact[2];
    let theta = theta_sq.sqrt();

    if theta < 1e-300 {
        // Near the identity J^{-1} ≈ I, so the backward error equals the
        // direct error.
        return (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
    }

    // Rotation axis n̂ = r_exact / θ.
    let inv_theta = 1.0 / theta;
    let n = [
        r_exact[0] * inv_theta,
        r_exact[1] * inv_theta,
        r_exact[2] * inv_theta,
    ];

    // Coefficients for the inverse Jacobian, with Taylor series for small θ
    // to avoid cancellation.
    let (sinc, one_minus_cos_div_theta) = if theta < 1e-7 {
        // sin(θ)/θ ≈ 1 - θ²/6 + θ⁴/120
        // (1-cos(θ))/θ ≈ θ/2 - θ³/24
        let theta_sq = theta * theta;
        (
            1.0 - theta_sq / 6.0 * (1.0 - theta_sq / 20.0),
            theta * (0.5 - theta_sq / 24.0),
        )
    } else {
        (theta.sin() / theta, (1.0 - theta.cos()) / theta)
    };
    let one_minus_sinc = 1.0 - sinc;

    // n̂ · e  and  n̂ × e
    let n_dot_e = n[0] * e[0] + n[1] * e[1] + n[2] * e[2];
    let n_cross_e = [
        n[1] * e[2] - n[2] * e[1],
        n[2] * e[0] - n[0] * e[2],
        n[0] * e[1] - n[1] * e[0],
    ];

    // (Jf)^{-1} * e = sinc·e + (1−sinc)·(n̂·e)·n̂ − (1−cos θ)/θ · (n̂ × e)
    let b = [
        sinc * e[0] + one_minus_sinc * n_dot_e * n[0]
            - one_minus_cos_div_theta * n_cross_e[0],
        sinc * e[1] + one_minus_sinc * n_dot_e * n[1]
            - one_minus_cos_div_theta * n_cross_e[1],
        sinc * e[2] + one_minus_sinc * n_dot_e * n[2]
            - one_minus_cos_div_theta * n_cross_e[2],
    ];

    (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt()
}

/// Generates a small-angle unit quaternion using a random rotation axis and an
/// angle sampled uniformly from `[0, max_angle)`.
fn gen_small_angle_q(rng: &mut rand::rngs::SmallRng, max_angle: f64) -> UQ32 {
    // Generate a uniformly distributed rotation axis on S² via rejection
    // sampling inside the unit ball.
    let (nx, ny, nz) = loop {
        let x = rng.random_range(-1.0f64..1.0);
        let y = rng.random_range(-1.0f64..1.0);
        let z = rng.random_range(-1.0f64..1.0);
        let r_sq = x * x + y * y + z * z;
        if r_sq > 1e-10 && r_sq <= 1.0 {
            let r = r_sq.sqrt();
            break (x / r, y / r, z / r);
        }
    };

    // Small rotation angle θ ∈ [0, max_angle).
    let angle: f64 = rng.random::<f64>() * max_angle;
    let (sin_half, cos_half) = (angle / 2.0).sin_cos();

    Q32::new(
        cos_half as f32,
        (sin_half * nx) as f32,
        (sin_half * ny) as f32,
        (sin_half * nz) as f32,
    )
    .normalize()
    .unwrap()
}

fn main() {
    let col_width = 15;

    // Table of implementations to compare.
    // Each entry is a (label, to_rotation_vector_fn) pair.
    let implementations: [(&str, fn(UQ32) -> [f32; 3]); 5] = [
        ("num_quaternion", |q| q.to_rotation_vector()),
        ("num_quaternion (gen)", |q| {
            q.to_rotation_vector_impl_generic()
        }),
        (
            "num_quaternion (f64)",
            // Up-cast to f64, compute exactly, down-cast back to f32.
            |q| {
                let rv64 = upcast_to_f64(q).to_rotation_vector();
                [rv64[0] as f32, rv64[1] as f32, rv64[2] as f32]
            },
        ),
        (
            "nalgebra",
            |q| {
                let q = q.as_quaternion();
                let na_q = nalgebra::UnitQuaternion::from_quaternion(
                    nalgebra::Quaternion::new(q.w, q.x, q.y, q.z),
                );
                let ax = na_q.scaled_axis();
                [ax.x, ax.y, ax.z]
            },
        ),
        (
            "quaternion_core",
            |q| {
                let q = q.as_quaternion();
                quaternion_core::to_rotation_vector((q.w, [q.x, q.y, q.z]))
            },
        ),
    ];

    // -----------------------------------------------------------------------
    // Scenario 1: uniformly random quaternions (positive hemisphere)
    // -----------------------------------------------------------------------
    println!(
        "Benchmarking the backward accuracy of `to_rotation_vector` for \
         uniformly random unit quaternions.\n"
    );
    println!(
        "The backward error is ||(Jf)^{{-1}} · (f_exact(q) − f̃(q))|| where \
         f is the exact rotation-vector map and Jf is its Jacobian. It \
         measures how large an input perturbation would explain the observed \
         output error. `UQ64::to_rotation_vector` is used as the exact \
         reference."
    );
    println!();
    print_table(&implementations, col_width, |rng| {
        let q = rng.random::<UQ32>();
        if q.as_quaternion().w >= 0.0 { q } else { -q }
    });

    // -----------------------------------------------------------------------
    // Scenario 2: small rotation angles
    // -----------------------------------------------------------------------
    let max_small_angle = 1e-3_f64; // ~0.057°
    println!(
        "\nBenchmarking the backward accuracy of `to_rotation_vector` for \
         small rotation angles (θ ∈ [0, {max_small_angle:.0e}] rad).\n"
    );
    print_table(&implementations, col_width, |rng| {
        gen_small_angle_q(rng, max_small_angle)
    });

    println!("\nAll errors are expressed as multiples of f32::EPSILON ({:.3e}).", f32::EPSILON);
    println!(
        "Values ≤ 1 (green) indicate the implementation is as accurate as \
         the f32 input allows."
    );
}

fn print_table<F>(
    implementations: &[(&str, fn(UQ32) -> [f32; 3])],
    col_width: usize,
    mut gen_q: F,
) where
    F: FnMut(&mut rand::rngs::SmallRng) -> UQ32,
{
    println!(
        "{:25} | {:^width$} | {:^width$}",
        "Method",
        "Max error",
        "RMS error",
        width = col_width
    );
    println!(
        "{:=<25}=|={:=<width$}=|={:=<width$}=",
        "",
        "",
        "",
        width = col_width
    );

    for (label, to_vec) in implementations {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);

        let mut max_error = 0.0_f64;
        let mut sum_sqr_error = 0.0_f64;

        for _ in 0..NUM_SAMPLES {
            let q = gen_q(&mut rng);
            let r_approx = to_vec(q);
            let err = backward_error(q, r_approx);
            if err > max_error {
                max_error = err;
            }
            sum_sqr_error += err * err;
        }

        let rms_error = (sum_sqr_error / NUM_SAMPLES as f64).sqrt();

        // Express errors as multiples of f32::EPSILON.
        let max_in_eps = max_error / f32::EPSILON as f64;
        let rms_in_eps = rms_error / f32::EPSILON as f64;

        let format_error = |error_in_eps: f64| -> String {
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
            "{:25} | {} | {}",
            label,
            format_error(max_in_eps),
            format_error(rms_in_eps),
        );
        io::stdout().flush().unwrap();
    }
}
