# num-quaternion

<p align="center">
  <a href="https://github.com/ralphtandetzky/num-quaternion/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/ralphtandetzky/num-quaternion/cargo_build_and_test.yml?branch=master" />
  </a>
  <a href="https://docs.rs/num-quaternion/latest/num-quaternion/">
    <img src="https://img.shields.io/docsrs/num-quaternion" />
  </a>
  <a href="https://crates.io/crates/num-quaternion">
    <img src="https://img.shields.io/crates/d/num-quaternion" />
  </a>
  <a href="https://choosealicense.com/licenses/mit/">
    <img src="https://img.shields.io/crates/l/num-quaternion" />
  </a>
  <a href="https://crates.io/crates/num-quaternion">
    <img src="https://img.shields.io/crates/v/num-quaternion" />
  </a>
  <a href="https://github.com/ralphtandetzky/num-quaternion/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/ralphtandetzky/num-quaternion" />
  </a>
</p>

Quaternions for Rust.


## Usage

Run
```bash
cargo add num-quaternion
```
to add this crate as a dependency to your project.

## Documentation

Please find the reference documentation [here on docs.rs](https://docs.rs/num-quaternion/latest/num-quaternion/).


## Features

This crate can be used without the standard library (`#![no_std]`) by disabling
the default `std` feature. Use this in `Cargo.toml`:

```toml
[dependencies]
num-quaternion = { version = "0.1.0", default-features = false }
```

Features based on `Float` types are only available when `std` or `libm` is
enabled. Where possible, `FloatCore` is used instead.  Formatting complex
numbers only supports format width when `std` is enabled.


## Releases

Release notes are available in [RELEASES.md](RELEASES.md).


## Contributing

Contributions are highly welcome. Unless you explicitly state otherwise, 
any contribution intentionally submitted for inclusion in the work by you, 
as defined in the Apache-2.0 license, shall be dual licensed as above, 
without any additional terms or conditions.


## Bug Reports and Feature Requests

If you spot a bug, please report it 
[here](https://github.com/ralphtandetzky/num-quaternion/issues). 
If you would like a feature to be worked on with higher priority, 
please don't hesitate to let me know 
[here](https://github.com/ralphtandetzky/num-quaternion/issues).


## License

Licensed under either of

 * [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
 * [MIT license](http://opensource.org/licenses/MIT)

at your option.
