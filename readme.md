CUFFT Server
============

This is a simple library that provides some rust interfaces to CUDAs CUFFT
library.

Usage
-----

### fft_batch_gpu

`fft_batch_gpu<const NFFT: usize>(data: &mut [[Complex<f32>; NFFT]]) -> Result<()>`

This runs CUFFT on a batch of arrays in `data` each of size `NFFT`. The FFTs
are performed in place, and the return is `Ok(())` on success.

### Array2D

Provides an interface using a struct that stores 2d array of data and can run
the CUFFT batch FFT on all data in the struct.

```rust
let mut a2d = Array2D::new(nfft);
a2d.push_array(&data1).unwrap();
a2d.push_array(&data2).unwrap();
a2d.push_array(&data3).unwrap();
a2d.pop_array().unwrap();
a2d.fft().unwrap();
let output1: &[Complex<f32>] = a2d.get(1).unwrap(); // gets reference to transform of data2
let output2: Vec<&[Complex<f32>]> = a2d.get_all();  // gets references to all arrays in a Vec
```

Build
-----

Requires CMake 3.28+, CUDA with support for compute 7.5, and gcc 12. This is
basically just tuned exactly to my system, since that's all I care about right
now.

Build is straightforward using the cmake crate. Basically just like normal:

```
$ cargo build
```

Test
----

There is a single test.

```
cargo test
```
