# How to Run Benchmarks

This document describes how to run the benchmarks in this repository and how to
ensure they produce valid and reproducible results.

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
cargo bench --all-features

# Run a specific benchmark file
cargo bench --bench from_rotation_vector_runtime --features std,rand
cargo bench --bench to_rotation_vector_runtime --features std,rand
cargo bench --bench norm_runtime

# Run with additional optimizations
RUSTFLAGS="-C target-cpu=native" cargo bench --all-features
```

### Understanding the Output

Please refer to [Criterion](https://github.com/criterion-rs/criterion.rs)
crate, which provides:
- Statistical analysis of timing measurements
- Detection of performance regressions
- HTML reports in `target/criterion/`

## Ensuring Valid Benchmark Results

Benchmark results can be affected by system state, background processes, and
power management. Follow these steps to ensure stable and reproducible
measurements.

### 1. Set CPU Governor to Performance Mode

On Linux systems, the CPU governor controls frequency scaling. For benchmarking,
use performance mode:

```bash
sudo cpupower frequency-set --governor performance
```

**Note**: On laptops, this will increase power consumption and heat generation.

### 2. Minimize Background Activity

```bash
# Check what's consuming CPU
htop
```

Close unnecessary applications
- Web browsers with many tabs
- IDEs not in use
- Update managers
- File indexing services
- etc.

### 3. Run Multiple Times

Even with all precautions, results can vary. Run benchmarks multiple times.

**Note:** There are more things that you can do to get better or more consistent
performance according to different website. The above methods have been most
effective to the author of the crate.

## Comparing Benchmark Results

Criterion automatically compares against baseline measurements:

```bash
# Establish baseline
cargo bench

# Make your changes...

# Compare against baseline
cargo bench
```

Look for the "change" column in the output. Criterion will indicate if the
change is statistically significant.
