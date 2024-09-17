//! This example program computes intermediate values between two poses. It
//! clearly demonstrates that spherical linear interpolation (slerp) is
//! different from linear interpolation of individual euler angles. Slerp
//! takes the shortest path between two orientations, while linear interpolation
//! of euler angles can exhibit gimbal lock and other artifacts. If you're
//! interested in some details, go watch [Quaternions and 3d rotation, explained
//! interactively from 3Blue1Brown](https://www.youtube.com/watch?v=zjMuIxRvygQ)
//! and explore some of the links given in the video description.

use num_quaternion::UQ32;

struct CameraPose {
    roll: f32,
    pitch: f32,
    yaw: f32,
    x: f32,
    y: f32,
    z: f32,
}

impl CameraPose {
    fn interpolate(&self, other: &CameraPose, t: f32) -> CameraPose {
        let q1 = UQ32::from_euler_angles(self.roll, self.pitch, self.yaw);
        let q2 = UQ32::from_euler_angles(other.roll, other.pitch, other.yaw);
        let q = q1.slerp(&q2, t);
        let euler = q.to_euler_angles();
        CameraPose {
            roll: euler.roll,
            pitch: euler.pitch,
            yaw: euler.yaw,
            x: self.x + t * (other.x - self.x),
            y: self.y + t * (other.y - self.y),
            z: self.z + t * (other.z - self.z),
        }
    }
}

impl std::fmt::Display for CameraPose {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "roll = {:4.3}, pitch = {:4.3}, yaw = {:4.3}, x = {:.1}, y = {:.1}, z = {:.1}",
            self.roll, self.pitch, self.yaw, self.x, self.y, self.z
        )
    }
}

fn main() {
    // Create two arbitrary poses
    let pose1 = CameraPose {
        roll: 0.0,
        pitch: 1.0,
        yaw: 0.0,
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    let pose2 = CameraPose {
        roll: 0.0,
        pitch: 0.0,
        yaw: 1.0,
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    println!("Initial pose: {}", pose1);
    println!("Final pose:   {}", pose2);

    // Compute intermediate poses
    println!("Intermediate poses:");
    for i in 0..=10 {
        let t = i as f32 / 10.0;
        let pose = pose1.interpolate(&pose2, t);
        println!("     t = {:.1}, {}", t, pose);
    }
}
