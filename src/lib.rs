#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use anyhow::{bail, Result};
use num_complex::Complex;

pub unsafe fn fft_batch_gpu<const NFFT: usize>(data: &mut [[Complex<f32>; NFFT]]) -> Result<()> {
    let batch = data.len();
    if batch < 1 {
        bail!("data must contain at least 1 array to perform FFT");
    }
    c2c_fft_batch(data as *mut [[Complex::<f32>; NFFT]] as *mut f32, NFFT as i32, batch as i32);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const nfft: usize = 8;
    const gold: [Complex<f32>; 8] = [
        Complex::new(28.0, -28.0),
        Complex::new(5.65685425, 13.65685425),
        Complex::new(0.0, 8.0),
        Complex::new(-2.34314575, 5.65685425),
        Complex::new(-4.0, 4.0),
        Complex::new(-5.65685425, 2.34314575),
        Complex::new(-8.0, 0.0),
        Complex::new(-13.65685425, -5.65685425),
    ];

    #[test]
    fn test_const_generic_version() {
        let batch = 2;
        let mut data: Vec<_> = (0..batch).map(|_| [Complex::<f32>::new(0.0, 0.0); nfft]).collect();
        for array in &mut data {
            for n in 0..nfft {
                array[n] = Complex::new(n as f32, -(n as f32));
            }
        }
        unsafe { fft_batch_gpu::<nfft>(&mut data) }.unwrap();
        for array in data {
            for k in 0..nfft {
                assert!((array[k] - gold[k]).norm() < 1e-4);
            }
        }
    }
}
