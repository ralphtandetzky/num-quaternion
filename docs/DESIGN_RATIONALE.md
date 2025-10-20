# Design Rationale

## Design Goals

The `num-quaternion` crate is created with the following design goals in mind:

1. **Run-time efficiency**: The processor spends a minimal amount of time for a computation to finish. The implementation of `num-quaternion` minimizes both the latency and the reciprocal throughput of computations.
2. **Numerical accuracy**: The functions in `num-quaternion` mostly have a maximum relative error of only a few ulps (units in the last place) or floating point epsilons. Accuracy is limited by the accuracy of the input data and the underlying transcendental function implementation.
3. **Correctness for all inputs**: The crate guarantees intuitive and correct behavior in edge cases. The implementation generally checks for edge cases with a minimal number of conditions and branches, always optimizing the general case. With reasonable branch prediction, the checking overhead should generally be negligible.
4. **Intuitive interface**: The crate provides an interface that is easy to understand and use. Quaternions can be used just like built-in floating point numbers or the complex numbers from the [`num-complex` crate](https://github.com/rust-num/num-complex).

## Error Handling & IEEE-754 Floating Point Values

**TLDR:** Use `is_finite()` to check for the success of floating point operations in `num-quaternion` functions. Don't use signaling `NaN`s.

Quaternions are normally composed of `f32` or `f64` values. *Errors are reported by infinite and `NaN` return values*. `NaN` values are infectious, meaning that if `NaN` values are fed into a function, then `NaN` values are returned accordingly. This ensures correct error propagation. Computations can be checked for success by using the `is_finite()` method on the final result, which ensures that neither infinite nor `NaN` values are present. The details of error conditions are documented for each function individually.

Client code should not rely on the *floating point status flags* for error handling with `num-quaternion`. The floating point environment is essentially global state with all the nasty problems that entails. Providing additional guarantees for the floating point flags would require additional computational costs, which are completely unnecessary because reporting errors through infinities and `NaN`s is already sufficient. Apart from that, Rust does not meaningfully support floating point exceptions.

The `num-quaternion` crate assumes that there are *no signaling `NaN`* inputs to functions. Let's discuss the background for why we make this assumption. The IEEE-754 standard on floating point numbers allows for signaling `NaN`s which cause signals to be raised when used in arithmetic operations. IEEE also guarantees that the result of any computation never creates any signaling `NaN` value, even if the inputs are signaling. There is some discussion on whether Rust should completely comply with this requirement, because it may prevent some optimizations: Replacing the expression `x * 1.0` by `x` would not be valid if `x` is a signaling `NaN` value, since the output needs to be a quiet `NaN` value. Therefore, the assumption that there are no signaling `NaN` values allows certain optimizations. On the flip side, the assumption that there are no signaling `NaN` in any input is very strong, because signaling `NaN`s can only be produced by explicitly calling `from_bits` or some equivalent unsafe operation. Feel free to read more details about Rust and IEEE-754 floating point values [here](https://github.com/rust-lang/rfcs/pull/3514/files).
