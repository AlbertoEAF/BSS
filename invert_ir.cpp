#include <fftw3.h>
#include <cmath>

#include <iostream>
#include "Buffer.h"

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
  magnitude[0] = hc[0];
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
  Gnuplot  pwav, pi, po, pM, pMIH, ph, pih, p;

  fftw_plan forward_plan, backwards_plan, h_forward_plan;
  real FFT_df;
  idx FFT_N;
  int FFT_flags;

  Guarantee(argc >= 3, "Missing program options:\n \tinvert_ir <impulse_response_wav> <output_inverse_impulse_response_wav> [wait]");

  SndfileHandle input_h(argv[1]);
  Guarantee(wav::ok(input_h), "Impulse response not found.");
  Guarantee(wav::mono(input_h), "The impulse response must be mono.");

  uint sample_rate_Hz = input_h.samplerate();
  real T_sampling = 1/(real)sample_rate_Hz;

  //  idx h_size = input_h.frames() * 2 - 1;
  idx h_size = input_h.frames();
  h_size += h_size % 2; // Implemented only for even FFT_N sizes (HC representation indexing simplicity)
  Buffer<real> h(h_size, 0, fftw_malloc, fftw_free), ih(h), H(h), IH(h), M(h), MIH(h);
  input_h.read(h(), input_h.frames());


  /// FFT
	
  FFT_N = h_size;	
  FFT_df = sample_rate_Hz / (real) FFT_N;

  FFT_flags = FFTW_ESTIMATE; // Use wisdom + FFTW_EXHAUSTIVE later!
  h_forward_plan = fftw_plan_r2r_1d(FFT_N, h() , H() , FFTW_R2HC, FFT_flags); 
  backwards_plan = fftw_plan_r2r_1d(FFT_N, IH(), ih(), FFTW_HC2R, FFT_flags); 

  // Plot x-axis buffers
  Buffer<real> f(FFT_N), t(FFT_N);
  for (idx i=0; i < FFT_N; ++i)
    {
      t[i] = i * T_sampling;
      f[i] = i * FFT_df;
    }

  fftw_execute(h_forward_plan);
  ph.set_labels("t (s)", "Amplitude");
  ph.plot_xy(t(), h(), FFT_N, "h(t)");
  pM.set_labels("f (Hz)", "Magnitude");
  pM.cmd("set logscale y");
  HC2magnitude(FFT_N, H(), M());
  //  pM.plot_xy(&f[1], &M[1], FFT_N/2, "|H(f)| AC");
  pM.plot_xy(f(), M(), FFT_N/2, "|H(f)|");

  // Calculate H^{-1}
  IH[0] = 1/H[0];
  for (size_t i=1; i < FFT_N/2; ++i)
    {
      // z = (a+bi)/(c+di) , (a+bi) = 1
      real c = H[i];
      real d = H[FFT_N-i];

      IH[i      ] = (c+d)/(c*c+d*d);
      IH[FFT_N-i] = (c-d)/(c*c+d*d);
    }
  IH[FFT_N/2] = 1/H[FFT_N/2];

  HC2magnitude(FFT_N, IH(), MIH());
  pMIH.set_labels("f (Hz)", "Magnitude");
  pMIH.cmd("set logscale y");
  pMIH.plot_xy(f(), MIH(), h_size/2, "|H^{-1}(f)|");

  fftw_execute(backwards_plan);

  ih /= (real) FFT_N;

  pih.set_labels("t (s)", "Amplitude");
  pih.plot_xy(t(), ih(), h_size, "h^{-1}");

  wav::write_mono (argv[2], ih(), h_size, sample_rate_Hz);

  if (argc > 3)
    wait();
  fftw_destroy_plan(h_forward_plan);
  fftw_destroy_plan(backwards_plan);
  puts("\nSuccess!");
  return 0;
}
