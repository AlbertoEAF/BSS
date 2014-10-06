#include <fftw3.h>
#include <cmath>

#include <iostream>
#include "libBuffers/Buffer.h"

#include "types.h"
#include "libs/config_parser.h"
#include "wav.h"
#include "gnuplot_ipp/gnuplot_ipp.h"
#include "filters.h"
#include "extra.h"

#include <string.h> // memcpy

#include <limits.h>

#include "libs/timer.h"


using std::cout;
using std::endl;

#include <complex>

real complex_norm(real re, real im)
{
  return sqrt(re*re + im*im);
}

/// @warn Might not behave well for n=odd!

/* READ!!: http://www.fftw.org/doc/The-Halfcomplex_002dformat-DFT.html */
void HC2magnitude(int N, real *hc, real *magnitude)
{
  magnitude[0] = std::abs(hc[0]);
  for (idx i=1; i < N/2; ++i)
    magnitude[i] = complex_norm(hc[i], hc[N-i]); // Not true for odd N!!!
}


/**
   Z = Z1*Z2
   @param[in] re1 - Re{Z1}
   @param[in] im1 - Im{Z1}
   @param[in] re2 - Re{Z2}
   @param[in] im2 - Im{Z2}
   @param[out] re - Re{Z}
   @param[out] im - Im{Z}

*/
inline void complex_multiply(real re1, real im1, real re2, real im2, real *re, real *im)
{
  *re = re1*re2 - im1*im2;
  *im = re1*im2 + im1*re2;
}

/**
   HalfComplex representation multiply
   @param[in] z1 - Input  HC array
   @param[in] z2 - Input  HC array
   @param[out] z - Output HC array
   @param[in] size - Size of the HC array

   @warn: ONLY FOR EVEN TRANSFORMATIONS!!!
*/
void hc_multiply (real *z1, real *z2, real *z, idx size)
{
  z[0] = z1[0]*z2[0];

  idx max_i = size/2;
  for (idx i=1; i < max_i; ++i)
    complex_multiply(z1[i], z1[size-i],
		     z2[i], z2[size-i],
		     &z[i], &z[size-i]);
}



int main(int argc, char **argv)
{
  /* Name convention throughout this file:
  	
     i - input
     o - output
     m - magnitude

     and capital letters for the frequency domain
  */	
  Gnuplot  pwav, pi, po, pM, ph, p;

	
  fftw_plan forward_plan;

  Guarantee(argc >= 2, "Missing program options:\n \tconvolver <input_wav>");


  SndfileHandle input_wav(argv[1]);


  uint sample_rate_Hz = input_wav.samplerate();

  Guarantee(wav::ok(input_wav), "File doesn't exist.");
  Guarantee(wav::mono(input_wav), "Input must be mono.");

  size_t FFT_N = input_wav.frames();
  FFT_N += (FFT_N%2);

  // g = wav, h = impulse response, g*h = convolution (output)
  Buffer<real> g(FFT_N, 0, fftw_malloc, fftw_free), G(g), M(g), 
    f(g), t(g);

  input_wav.read(g(), input_wav.frames());


  real FFT_df = sample_rate_Hz / (real) FFT_N;
  real T_sampling = 1/(real)sample_rate_Hz;

  int FFT_flags = FFTW_ESTIMATE; // Use wisdom + FFTW_EXHAUSTIVE later!

  forward_plan   = fftw_plan_r2r_1d(FFT_N, g() , G() , FFTW_R2HC, FFT_flags); 

  // Fill plot x-axis buffers
  for (idx i=0; i < FFT_N; ++i)
    {
      t[i] = i * T_sampling;
      f[i] = i * FFT_df;      
    }


  fftw_execute(forward_plan);
  p.set_labels("t (s)", "Amplitude");
  p.plot(t(), g(), g.size(), "g(t)");
  pM.set_labels("f (Hz)", "Magnitude");
  //  pM.cmd("set logscale y");
  HC2magnitude(FFT_N, G(), M());
  pM.plot(f(), M(), FFT_N/2-1, "|H(f)|");

  fftw_destroy_plan(forward_plan); 
  wait();
  return 0;
}
