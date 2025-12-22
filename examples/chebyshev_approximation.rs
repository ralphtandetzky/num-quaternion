//! This executable computes polynomial approximations of special functions
//! which are used in the implementation of the crate. The polynomial
//! approximations are compared against the naive implementations and generally
//! show higher accuracy and faster runtime.

use std::{
    f64::consts::PI,
    ops::{Add, Mul},
};

/// A type representing a polynomial as a vector of its coefficients.
/// The coefficients are ordered from the constant term upwards, e.g.,
/// `vec![c0, c1, c2]` represents the polynomial `c0 + c1*x + c2*x^2`.
/// Leading coefficients are always non-zero.
#[derive(Clone, Debug, PartialEq)]
struct Poly(Vec<f64>);

/* Polynomial Utility Functions */

impl Poly {
    fn new(coeffs: Vec<f64>) -> Self {
        let mut p = Poly(coeffs);
        p.trim_zeros();
        p
    }

    fn degree(&self) -> usize {
        self.0.len().max(1) - 1
    }

    fn eval(&self, x: f64) -> f64 {
        self.0.iter().rev().fold(0.0, |acc, &coeff| acc * x + coeff)
    }

    fn eval_f32(&self, x: f32) -> f32 {
        self.0
            .iter()
            .rev()
            .fold(0.0, |acc, &coeff| acc * x + coeff as f32)
    }

    /// Trims trailing zero coefficients from a polynomial representation.
    /// This is useful for cleaning up results from polynomial arithmetic.
    fn trim_zeros(&mut self) {
        while let Some(&0.0) = self.0.last() {
            self.0.pop();
        }
    }

    /// Composes a polynomial `P(y)` with a linear function `y = mx + c`.
    /// The result is a new polynomial in `x`, `P(mx + c)`.
    /// This implementation uses a Horner-like scheme for polynomial composition.
    fn compose_linear(&self, m: f64, c: f64) -> Poly {
        if self.0.len() <= 1 {
            return self.clone();
        }

        let linear_poly = Poly::new(vec![c, m]); // Represents c + mx

        // Start with the highest coefficient as a constant polynomial
        let mut result =
            Poly::new(vec![*self.0.last().expect("Polynomial is zero")]);

        // Iterate downwards from the second-to-last coefficient
        for &coeff in self.0.iter().rev().skip(1) {
            result = &result * &linear_poly;
            result.0[0] += coeff;
        }
        result
    }
}

impl Add for &Poly {
    type Output = Poly;

    fn add(self, other: &Poly) -> Poly {
        let mut result = self.0.clone();
        let max_len = result.len().max(other.0.len());
        result.resize(max_len, 0.0);
        for (r, o) in result.iter_mut().zip(other.0.iter()) {
            *r += o;
        }
        let mut result = Poly::new(result);
        result.trim_zeros();
        result
    }
}

impl Mul<f64> for &Poly {
    type Output = Poly;

    fn mul(self, scalar: f64) -> Poly {
        if scalar == 0.0 {
            return Poly::new(vec![]);
        }
        let result = self.0.iter().map(|&c| c * scalar).collect();
        Poly::new(result)
    }
}

impl Mul for &Poly {
    type Output = Poly;

    fn mul(self, other: &Poly) -> Poly {
        if self.0 == Vec::<f64>::new() || other.0 == Vec::<f64>::new() {
            return Poly::new(vec![]);
        }
        let deg1 = self.degree();
        let deg2 = other.degree();
        let mut result = vec![0.0; deg1 + deg2 + 1];
        for i in 0..=deg1 {
            for j in 0..=deg2 {
                result[i + j] += self.0[i] * other.0[j];
            }
        }
        Poly::new(result)
    }
}

/// Generates the `k`-th Chebyshev polynomial of the first kind, `T_k(x)`.
fn chebyshev_poly(k: usize) -> Poly {
    match k {
        0 => Poly::new(vec![1.0]),
        1 => Poly::new(vec![0.0, 1.0]),
        _ => {
            let mut tkm1 = Poly::new(vec![1.0]); // T_{k-1}
            let mut tk = Poly::new(vec![0.0, 1.0]); // T_k

            for _ in 1..k {
                // T_{k+1} = 2x * T_k - T_{k-1}
                let two_x = Poly::new(vec![0.0, 2.0]);
                let tkp1 = &(&two_x * &tk) + &(&tkm1 * -1.0);

                tkm1 = tk;
                tk = tkp1;
            }
            tk
        }
    }
}

/// Computes the L2 Chebyshev approximation for a function.
///
/// # Arguments
/// * `f` - The function to approximate.
/// * `a`, `b` - The start and end points of the interval `[a, b]`.
/// * `max_degree` - The maximum degree of the approximating polynomial.
/// * `epsilon` - The desired upper bound maximum error on the interval `[a, b]`.
///
/// # Returns
/// A `Result` containing the coefficients of the approximating polynomial,
/// or an error string if the inputs are invalid.
fn chebyshev_l2_approximation(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    max_degree: usize,
    epsilon: f64,
) -> Result<Poly, String> {
    if a >= b {
        return Err("Invalid interval: a must be less than b.".to_string());
    }
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.".to_string());
    }
    if max_degree == 0 {
        return Err("max_degree must be at least 1.".to_string());
    }

    // 1. Compute Chebyshev Coefficients using Clenshaw-Curtis Quadrature
    // The number of nodes N should be > 2*max_degree for accuracy.
    let n_nodes = 2 * max_degree + 2;
    let mut chebyshev_coeffs = vec![0.0; max_degree + 1];

    for (k, coeff) in chebyshev_coeffs.iter_mut().enumerate() {
        let mut sum = 0.0;
        for j in 0..=n_nodes {
            // Map standard Chebyshev nodes from [-1, 1] to [a, b]
            let y_j = (j as f64 * PI / n_nodes as f64).cos();
            let x_j = 0.5 * (a + b) + 0.5 * (b - a) * y_j;

            let f_val = f(x_j);
            let term =
                f_val * (k as f64 * j as f64 * PI / n_nodes as f64).cos();

            // First and last terms of the sum are halved
            if j == 0 || j == n_nodes {
                sum += 0.5 * term;
            } else {
                sum += term;
            }
        }
        *coeff = 2.0 / n_nodes as f64 * sum;
    }
    // The c_0 coefficient has a different normalization in the error formula,
    // but it's simpler to just handle it here.
    chebyshev_coeffs[0] /= 2.0;

    // 2. Find the minimal degree `n` that satisfies the error bound.
    let mut tail_sum = 0.0;
    let mut n = max_degree;
    for k in (1..=max_degree).rev() {
        let new_tail_sum = tail_sum + chebyshev_coeffs[k].abs();
        if new_tail_sum > epsilon {
            n = k;
            break;
        }
        tail_sum = new_tail_sum;
    }

    // 3. Construct the approximating polynomial from the truncated series.
    // P_n(x) = sum_{k=0 to n} c_k * T_k(y), where y is the mapped x.
    let mut approx_poly = Poly::new(vec![chebyshev_coeffs[0]]);
    let m = 2.0 / (b - a);
    let c = -(a + b) / (b - a);

    for k in 1..=n {
        let tk_poly = chebyshev_poly(k);
        let tk_composed = tk_poly.compose_linear(m, c);
        approx_poly = &approx_poly + &(&tk_composed * chebyshev_coeffs[k]);
    }

    Ok(approx_poly)
}

fn sinc_sqrt<F>(x: F) -> F
where
    F: num_traits::Float,
{
    if x == num_traits::zero() {
        num_traits::one()
    } else {
        let s = x.sqrt();
        s.sin() / s
    }
}

fn cos_sqrt<F>(x: F) -> F
where
    F: num_traits::Float,
{
    x.sqrt().cos()
}

fn correction_factor_to_argument_pos<F>(x: F) -> F
where
    F: num_traits::Float,
{
    if x == F::one() {
        return F::one() + F::one();
    }
    let y = (F::one() - x * x).sqrt();
    let phi = x.acos();
    let quot = phi / y;
    quot + quot
}

fn main() {
    let a = 0.0;
    let b = (std::f64::consts::PI / 2.0).powi(2);
    let max_degree = 50;
    let epsilon = 2.0 * f32::EPSILON as f64;

    run_chebyshev_approximation(
        "sinc(sqrt(x))",
        sinc_sqrt,
        sinc_sqrt,
        a,
        b,
        max_degree,
        epsilon,
    );

    run_chebyshev_approximation(
        "cos(sqrt(x))",
        cos_sqrt,
        cos_sqrt,
        a,
        b,
        max_degree,
        epsilon,
    );

    run_chebyshev_approximation(
        "correction_factor_to_argument_pos(x)",
        correction_factor_to_argument_pos,
        correction_factor_to_argument_pos,
        0.0,
        1.0,
        50,
        epsilon / 2.0,
    );
}

fn run_chebyshev_approximation(
    func_name: &str,
    ground_truth: impl Fn(f64) -> f64,
    naive_f32_impl: impl Fn(f32) -> f32,
    a: f64,
    b: f64,
    max_degree: usize,
    epsilon: f64,
) {
    println!(
        "Approximating the function {} on the interval [{}, {}].",
        func_name, a, b
    );

    match chebyshev_l2_approximation(&ground_truth, a, b, max_degree, epsilon) {
        Ok(approx_coeffs) => {
            println!("Approximation successful!");
            println!("Maximum degree: {}", max_degree);
            println!("Minimized degree: {}", approx_coeffs.degree());
            println!(
                "Resulting coefficients: {:?}",
                approx_coeffs
                    .0
                    .iter()
                    .map(|c| *c as f32)
                    .collect::<Vec<_>>()
            );

            // Verify the error at a few points
            let mut max_error_f64 = 0.0f64;
            let mut max_error_f32 = 0.0f64;
            let mut max_naive_error = 0.0f64;
            for i in ((a * 100.0) as i32)..=((b * 100.0) as i32) {
                // Compute error when using highest precision (f64)
                let x = i as f64 / 100.0;
                let original_val = ground_truth(x);
                let approx_val = approx_coeffs.eval(x);
                let error =
                    (original_val - approx_val).abs() / f32::EPSILON as f64;
                max_error_f64 = max_error_f64.max(error);

                // Compute error when using f32 precision
                let x_f32 = x as f32;
                let ground_truth_val = ground_truth(x_f32 as f64);
                let approx_val_f32 = approx_coeffs.eval_f32(x_f32);
                let error_f32 = (ground_truth_val - approx_val_f32 as f64)
                    .abs()
                    / f32::EPSILON as f64;
                max_error_f32 = max_error_f32.max(error_f32);

                // Naive f32 implementation error
                let naive_val = naive_f32_impl(x_f32) as f64;
                let ground_truth_val = ground_truth(x_f32 as f64);
                let naive_error =
                    (ground_truth_val - naive_val).abs() / f32::EPSILON as f64;
                max_naive_error = max_naive_error.max(naive_error);
            }
            println!(
                "Maximum error at sample points at f64 precision: {:.3} eps",
                max_error_f64
            );
            println!(
                "Maximum error at sample points at f32 precision: {:.3} eps",
                max_error_f32
            );
            println!(
                "Maximum error for naive_f32_impl at sample points: {:.3} eps",
                max_naive_error
            );
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}
