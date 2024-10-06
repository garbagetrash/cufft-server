#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use num_complex::Complex;

pub unsafe fn fft_batch_gpu<const NFFT: usize, const BATCH: usize>(data: &mut [[Complex<f32>; NFFT]; BATCH]) {
    c2c_fft_batch(unsafe { std::mem::transmute::<&mut [[Complex::<f32>; NFFT]; BATCH], *mut f32>(data) }, NFFT as i32, BATCH as i32);
}

const nfft: usize = 32;
const batch: usize = 32;

fn main() {
    let mut data = [[Complex::<f32>::new(0.0, 0.0); nfft]; batch];
    for b in 0..batch {
        for n in 0..nfft {
            data[b][n] = Complex::new(n as f32, -(n as f32));
        }
    }
    unsafe { fft_batch_gpu::<nfft, batch>(&mut data); }
    for i in 0..batch {
        for k in 0..nfft {
            println!("{}", data[i][k].re);
        }
    }
}
