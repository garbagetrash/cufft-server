#ifndef __CUFFT_SERVER_H__
#define __CUFFT_SERVER_H__

#ifdef __cplusplus
extern "C" {
#endif

// Prototypes
void c2c_fft_batch(float *h_data, int nfft, int batch);

#ifdef __cplusplus
}
#endif


#endif // __CUFFT_SERVER_H__
