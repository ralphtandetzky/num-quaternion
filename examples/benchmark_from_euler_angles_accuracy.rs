use rand::Rng;
use rand::SeedableRng;
use std::io::{self, Write};

const NUM_SAMPLES: usize = 10000000;

fn from_euler_angles_num_quaternion_f32(
    roll: f32,
    pitch: f32,
    yaw: f32,
) -> [f32; 4] {
    let num_quaternion::Q32 { w, x, y, z } =
        num_quaternion::UQ32::from_euler_angles(roll, pitch, yaw).into();
    [w, x, y, z]
}

fn from_euler_angles_quaternion_f32(
    roll: f32,
    pitch: f32,
    yaw: f32,
) -> [f32; 4] {
    let (w, [x, y, z]) = quaternion::euler_angles(roll, pitch, yaw);
    [w, x, y, z]
}

fn from_euler_angles_quaternion_core_f32(
    roll: f32,
    pitch: f32,
    yaw: f32,
) -> [f32; 4] {
    let (w, [x, y, z]) = quaternion_core::from_euler_angles(
        quaternion_core::RotationType::Extrinsic,
        quaternion_core::RotationSequence::XYZ,
        [roll, pitch, yaw],
    );
    [w, x, y, z]
}

fn from_euler_angles_nalgebra_f32(roll: f32, pitch: f32, yaw: f32) -> [f32; 4] {
    let q =
        nalgebra::geometry::UnitQuaternion::from_euler_angles(roll, pitch, yaw);
    let v = q.as_vector();
    [q.scalar(), v[0], v[1], v[2]]
}

fn from_euler_angles_micromath_f32(
    roll: f32,
    pitch: f32,
    yaw: f32,
) -> [f32; 4] {
    use micromath::{vector::Vector3d, Quaternion};
    let q = Quaternion::axis_angle(
        Vector3d {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        yaw,
    ) * Quaternion::axis_angle(
        Vector3d {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        pitch,
    ) * Quaternion::axis_angle(
        Vector3d {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        roll,
    );
    [q.w(), q.x(), q.y(), q.z()]
}

fn from_euler_angles_ground_truth(roll: f32, pitch: f32, yaw: f32) -> [f64; 4] {
    let num_quaternion::Q64 { w, x, y, z } =
        num_quaternion::UQ64::from_euler_angles(
            roll as f64,
            pitch as f64,
            yaw as f64,
        )
        .into();
    [w, x, y, z]
}

fn main() {
    type FromEulerAnglesFunc = fn(f32, f32, f32) -> [f32; 4];
    let from_euler_angles_funcs: [(FromEulerAnglesFunc, &'static str); 5] = [
        (from_euler_angles_num_quaternion_f32, "num_quaternion"),
        (from_euler_angles_quaternion_f32, "quaternion"),
        (from_euler_angles_quaternion_core_f32, "quaternion_core"),
        (from_euler_angles_nalgebra_f32, "nalgebra"),
        (from_euler_angles_micromath_f32, "micromath"),
    ];
    let func_space = from_euler_angles_funcs
        .iter()
        .map(|(_, name)| name.len())
        .max()
        .unwrap();
    let col_width = 13;

    println!(
        "Benchmarking the relative accuracy of `from_euler_angles` implementations.\n"
    );
    println!(
        "{1:func_space$} | {2:^0$}",
        col_width, "Implementation", "RMS error [ε]"
    );
    print!("{1:=<func_space$}=|={1:=<0$}", col_width, "");

    for (func, name) in from_euler_angles_funcs.iter() {
        print!("\n{name:func_space$} | ");
        io::stdout().flush().unwrap();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x7F0829AE4D31C6B5);
        let mut sum_sqr_error = 0.0;
        for _ in 0..NUM_SAMPLES {
            let pi = std::f32::consts::PI;
            let roll = rng.gen_range(-pi..pi);
            let pitch = rng.gen_range(-pi / 2.0..pi / 2.0);
            let yaw = rng.gen_range(-pi..pi);
            let q32 = func(roll, pitch, yaw);
            let q64_1 = from_euler_angles_ground_truth(roll, pitch, yaw);
            sum_sqr_error += q32
                .iter()
                .zip(q64_1.iter())
                .map(|(&a, &b)| (a as f64 - b).powi(2))
                .sum::<f64>();
        }
        let rms_error = (sum_sqr_error / NUM_SAMPLES as f64).sqrt();
        let rms_error_in_eps = rms_error / f32::EPSILON as f64;
        let formatted_rms_error = if rms_error_in_eps <= 1.0 {
            format!("{:.4}", rms_error_in_eps)
        } else {
            format!("{:.4}", rms_error_in_eps)
        };
        let color_code = if rms_error_in_eps <= 1.0 {
            "92" // green
        } else {
            "91" // red
        };
        print!("\x1b[{color_code}m{formatted_rms_error:>col_width$}\x1b[0m");
    }

    println!(
        "\n\nThe root mean square error is given in `f32` machine epsilons."
    );
    println!("The very large error in the `quaternion` crate is due to an incorrect ");
    println!("implementation of the `from_euler_angles` function.");
    println!("The `micromath` crate does not provide a `from_euler_angles` function,");
    println!("so we implemented it using the `axis_angle` function.");
    println!("The remaining traits implement the `from_euler_angles` function using ");
    println!("exactly the same algorithm, though the APIs differ somewhat.");
    println!("The roll pitch and yaw angles where sampled from uniform distributions ");
    println!("over the ranges [-π, π], [-π/2, π/2], and [-π, π] respectively.");
}
