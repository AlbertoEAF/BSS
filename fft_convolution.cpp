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

int valid_FFT_convolution(idx h_nonzero_size, idx FFT_N)
{
  idx g_nonzero_size = FFT_N - h_nonzero_size + 1;
  printf(
	 "FFT convolution with \n"
	 "    FFT_N  = %ld\n"
	 "    g_size = %ld\n"
	 "    h_size = %ld\n",
	 FFT_N, g_nonzero_size, h_nonzero_size);
  if (g_nonzero_size < 1)
    {
      puts("Invalid configuration!");
      return 0;
    }
  return 1;
}


inline void fillFFTblock(real *data, idx data_size, real *block, idx block_size)
{
  idx i;
  for (i=0; i < data_size; ++i)
    block[i] = data[i];
  for (i=data_size; i < block_size; ++i)
    block[i] = 0.0;
  // memset((void*)wav_out, 0, sizeof(real) * (N_wav+h_size-1));
  // memcpy()
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



int main()
{
  /* Name convention throughout this file:
  	
     i - input
     o - output
     m - magnitude

     and capital letters for the frequency domain
  */	
  Gnuplot  pwav, pi, po, pM, ph, p;

  real *wav, *wav_out;
  real *g, *G, *h, *H, *gh, *GH, *M; // gh is the convolution
  real *x, *f; // X-axis
  	
  idx W_size , h_size ; // number of non-zero elements in the filter
  idx g_overlap, h_overlap;
  idx t;

  fftw_plan forward_plan, backwards_plan, h_forward_plan;
  real FFT_df;
  idx FFT_N, FFT_spacing, FFT_pos;
  int FFT_flags;

  Options o("settings.cfg", Quit, 1);

  /// Read + plot input .wav

  SndfileHandle input_wav(o("wav_filepath"));

  uint sample_rate_Hz = input_wav.samplerate();

  idx N_wav = input_wav.frames(); 
  wav = new real[N_wav];
  input_wav.read(wav, N_wav);
  printf("\nProcessing input file with %lu frames @ %u Hz.\n\n", 
	 N_wav, sample_rate_Hz);	

  printf("Max int: %d\n"
	 "Max idx: %ld\n", INT_MAX, LONG_MAX);
  printf("Indexing usage: %.2f%%\n\n", 0.01*(float)input_wav.frames()/(float)LONG_MAX);
	

  //pwav.plot_y(wav, N_wav, "Input");

  /// FFT
	
  // Wee need first the size of the impulse response to configure the convolution
  puts("Measuring impulse response...");
  real A               = o.d("h_A");
  real tau             = o.d("h_tau"); 
  real max_filter_time = o.d("h_max_time");
  real *h_data = decay_filter (A, tau, 0., max_filter_time, sample_rate_Hz, &h_size);

  W_size = h_size + 1; // +1 so FFT_N becomes a power of 2
  FFT_N = W_size+h_size-1;
  g_overlap = FFT_N-W_size;
  h_overlap = FFT_N-h_size;

  if ( ! valid_FFT_convolution(h_size, FFT_N) )
    return -1;
	
  FFT_df = sample_rate_Hz / (real) FFT_N;
  FFT_flags = FFTW_ESTIMATE; // Use wisdom + FFTW_EXHAUSTIVE later!

  x  = fftw_alloc_real(FFT_N);
  f  = fftw_alloc_real(FFT_N);
  g  = fftw_alloc_real(FFT_N);
  G  = fftw_alloc_real(FFT_N);
  h  = fftw_alloc_real(FFT_N);
  H  = fftw_alloc_real(FFT_N);
  M  = fftw_alloc_real(FFT_N);
  gh = fftw_alloc_real(FFT_N);
  GH = fftw_alloc_real(FFT_N);
  wav_out = new real[N_wav+h_size-1];
	

  cout << "Estimating FFT plan..." << endl;
  cout << "The fast way!\n";
  FFT_flags = FFTW_ESTIMATE;
  forward_plan   = fftw_plan_r2r_1d(FFT_N, g , G , FFTW_R2HC, FFT_flags); 
  h_forward_plan = fftw_plan_r2r_1d(FFT_N, h , H , FFTW_R2HC, FFT_flags); 
  backwards_plan = fftw_plan_r2r_1d(FFT_N, GH, gh, FFTW_HC2R, FFT_flags); 
  cout << "DONE" << endl;

  // f
  for (idx i=0; i < FFT_N; ++i)
    f[i] = i * FFT_df;

  // h & H
  for (t = 0; t < h_size; ++t)
    h[t] = h_data[t];
  for (t = h_size; t < FFT_N; ++t)
    h[t] = 0.0;

  fftw_execute(h_forward_plan);
  ph.plot_y(h, FFT_N, "h(t)");
  pM.set_labels("f (Hz)", "Magnitude");
  HC2magnitude(FFT_N, H, M);
  pM.plot_xy(&f[1], &M[1], FFT_N-1, "|H(f)| AC");


  Timer convolution_timer;
  convolution_timer.start();
  memset((void*)wav_out, 0, sizeof(real) * (N_wav+h_size-1));
  if (FFT_N%2)
    {
      puts("Odd FFTs are not implemented!!!");
      return EXIT_FAILURE;
    }
  real normalization_factor = 1/(real)FFT_N;
  for (FFT_pos = 0; FFT_pos+FFT_N <= N_wav+h_size-1; FFT_pos += W_size)
    {
      fillFFTblock(&wav[FFT_pos], W_size, g, FFT_N);
		
      fftw_execute(forward_plan);


      hc_multiply(G, H, GH, FFT_N);
		
      fftw_execute(backwards_plan);

      // add
      for (t = 0; t < W_size; ++t)
	wav_out[FFT_pos+t] += gh[t] * normalization_factor;
      // copy
      for (t = W_size; t < FFT_N; ++t)
	wav_out[FFT_pos+t] = gh[t] * normalization_factor;
    }
  convolution_timer.stop();

  printf("Convolution execution time: %lf (s)\n", convolution_timer.getElapsedTimeInSec());
  /* The maximum won't be for sure after the sound ends except after pathological cases
     and we won't search in the negative space either. */
  normalize_to(wav_out, N_wav+h_size-1, max(wav_out, N_wav+h_size-1));

  p.plot_y(wav_out, N_wav+h_size-1, "g*h");

  write_mono_wav ("gh_fft.wav", wav_out, N_wav+h_size-1, sample_rate_Hz);

  wait();
  fftw_destroy_plan(forward_plan); 
  fftw_destroy_plan(h_forward_plan);
  fftw_destroy_plan(backwards_plan);
  fftw_free(x);
  fftw_free(f);
  fftw_free(g);
  fftw_free(G);
  fftw_free(h);
  fftw_free(H);
  fftw_free(gh);
  fftw_free(GH);
  delete[] h_data;
  delete[] wav;
  delete[] wav_out;
  puts("\nSuccess!");
  return 0;
}
