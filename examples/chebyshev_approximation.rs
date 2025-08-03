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

/// Computes the L2 Chebyshev approximation for a function given as a polynomial.
///
/// # Arguments
/// * `f_coeffs` - The coefficients of the input polynomial `f(x)`.
/// * `a`, `b` - The start and end points of the interval `[a, b]`.
/// * `epsilon` - The desired upper bound maximum error on the interval `[a, b]`.
///
/// # Returns
/// A `Result` containing the coefficients of the approximating polynomial,
/// or an error string if the inputs are invalid.
fn chebyshev_l2_approximation(
    f_coeffs: &Poly,
    a: f64,
    b: f64,
    epsilon: f64,
) -> Result<Poly, String> {
    if a >= b {
        return Err("Invalid interval: a must be less than b.".to_string());
    }
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.".to_string());
    }

    let degree = f_coeffs.degree();
    if degree == 0 {
        return Ok(f_coeffs.clone()); // Function is a constant
    }

    // 1. Compute Chebyshev Coefficients using Clenshaw-Curtis Quadrature
    // The number of nodes N should be > 2*degree for accuracy.
    let n_nodes = 2 * degree + 2;
    let mut chebyshev_coeffs = vec![0.0; degree + 1];

    for (k, coeff) in chebyshev_coeffs.iter_mut().enumerate() {
        let mut sum = 0.0;
        for j in 0..=n_nodes {
            // Map standard Chebyshev nodes from [-1, 1] to [a, b]
            let y_j = (j as f64 * PI / n_nodes as f64).cos();
            let x_j = 0.5 * (a + b) + 0.5 * (b - a) * y_j;

            let f_val = f_coeffs.eval(x_j);
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
    let mut n = degree;
    for k in (1..=degree).rev() {
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

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, x| acc * x as f64)
}

fn main() {
    // Example: Approximate f(x) = sinc(sqrt(x)) on [0, (pi/2)^2]
    // We use the Taylor series of sinc(x) up to degree 50 as our input polynomial.
    // sinc(x) = sin(x)/x = 1 - x^2/6 + x^4/120 - x^6/5040 + x^8/362880 + ...
    // sinc(sqrt(x)) = 1 - x/6 + x^2/120 - x^3/5040 + x^4/362880 + ...
    let sinc_sqrt_poly = Poly::new(
        (0..=50)
            .map(|n| if n % 2 == 0 { 1.0 } else { -1.0 } / factorial(2 * n + 1))
            .collect(),
    );

    let a = 0.0;
    let b = (std::f64::consts::PI / 2.0).powi(2);
    let epsilon = 5.0 * f32::EPSILON as f64;

    println!(
        "Approximating a polynomial of degree {}",
        sinc_sqrt_poly.degree()
    );

    match chebyshev_l2_approximation(&sinc_sqrt_poly, a, b, epsilon) {
        Ok(approx_coeffs) => {
            println!("Approximation successful!");
            println!("Original degree: {}", sinc_sqrt_poly.degree());
            println!("Minimized degree: {}", approx_coeffs.degree());
            println!("Resulting coefficients: {:?}", approx_coeffs);

            // Verify the error at a few points
            println!("\nError verification at sample points:");
            let mut max_error = 0.0f64;
            for i in ((a * 100.0) as i32)..=((b * 100.0) as i32) {
                let x = i as f64 / 100.0;
                let original_val = sinc_sqrt_poly.eval(x);
                let approx_val = approx_coeffs.eval(x);
                let error =
                    (original_val - approx_val).abs() / f32::EPSILON as f64;
                max_error = max_error.max(error);
                println!(
                    "x={:>6.2}  f(x)={:>12.7}  p(x)={:>12.7}  err={:>10.2}eps",
                    x, original_val, approx_val, error
                );
            }
            println!("Maximum error: {:.7} eps", max_error);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}
