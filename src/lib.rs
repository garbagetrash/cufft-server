#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use anyhow::{bail, Result};
use num_complex::Complex;

pub struct Array2D {
    data: Vec<Complex<f32>>,
    nfft: usize,
}

impl Array2D {
    pub fn new(nfft: usize) -> Self {
        Self { data: vec![], nfft }
    }

    pub fn fft(&mut self) -> Result<()> {
        let batch = self.number_of_arrays();
        unsafe {
            c2c_fft_batch(
                &mut self.data[..] as *mut [Complex<f32>] as *mut f32,
                self.nfft as i32,
                batch as i32,
            )
        };
        Ok(())
    }

    pub fn number_of_arrays(&self) -> usize {
        self.data.len() / self.nfft
    }

    pub fn push_array(&mut self, array: &[Complex<f32>]) -> Result<()> {
        if array.len() != self.nfft {
            bail!("array length must be of size nfft");
        }
        self.data.append(&mut array.to_vec());
        Ok(())
    }

    pub fn pop_array(&mut self) -> Result<()> {
        if self.data.is_empty() {
            bail!("can't pop array from empty Array2D");
        }
        self.data.truncate(self.data.len() - self.nfft);
        Ok(())
    }

    pub fn get(&self, index: usize) -> Result<&[Complex<f32>]> {
        if index + 1 >= self.number_of_arrays() {
            bail!("index out of bounds");
        }
        Ok(&self.data[self.nfft * index..self.nfft * (index + 1)])
    }

    pub fn get_all(&self) -> Vec<&[Complex<f32>]> {
        self.data.chunks_exact(self.nfft).collect()
    }
}

pub fn fft_batch_gpu<const NFFT: usize>(data: &mut [[Complex<f32>; NFFT]]) -> Result<()> {
    let batch = data.len();
    if batch < 1 {
        bail!("data must contain at least 1 array to perform FFT");
    }
    unsafe {
        c2c_fft_batch(
            data as *mut [[Complex<f32>; NFFT]] as *mut f32,
            NFFT as i32,
            batch as i32,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const nfft: usize = 8;
    const gold: [Complex<f32>; nfft] = [
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
        let mut data: Vec<_> = (0..batch)
            .map(|_| [Complex::<f32>::new(0.0, 0.0); nfft])
            .collect();
        for array in &mut data {
            for n in 0..nfft {
                array[n] = Complex::new(n as f32, -(n as f32));
            }
        }
        fft_batch_gpu::<nfft>(&mut data).unwrap();
        for array in data {
            for k in 0..nfft {
                assert!((array[k] - gold[k]).norm() < 1e-4);
            }
        }
    }

    #[test]
    fn test_array2d() {
        let batch = 5;
        let data: Vec<_> = (0..nfft)
            .map(|n| Complex::new(n as f32, -(n as f32)))
            .collect();
        let mut a2d = Array2D::new(nfft);
        for _ in 0..batch {
            a2d.push_array(&data).unwrap();
        }
        a2d.fft().unwrap();
        for array in a2d.get_all() {
            for k in 0..nfft {
                assert!((array[k] - gold[k]).norm() < 1e-4);
            }
        }
    }
}
