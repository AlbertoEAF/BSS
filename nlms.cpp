// Implementation of NLMS adaptive filter. Saves the final filter coefficients.

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
using std::cin;
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
  Assert(argc >= 5, "Missing program options:\n \tNLMS <d> <x> <w> <w_size> [wait]");

  SndfileHandle file_d(argv[1]), file_x(argv[2]);
  Guarantee(wav::ok(file_d) && wav::ok(file_x), "Impulse response not found.");
  Guarantee(wav::mono(file_d) && wav::mono(file_x), "Files must be mono.");

  uint sample_rate_Hz = file_d.samplerate();
  real T_sampling = 1/(real)sample_rate_Hz;

  Guarantee(file_d.samplerate() == file_x.samplerate(), "Sampling rates must match.");

  // w-vector are the weights of the adaptive filter and we are initializing them at 1
  // File_x is larger than file_d due to the convolution. Set both to the same size.
  Buffer<real> d(file_d.frames()), x(file_x.frames()), w(atol(argv[4])), y(x.size()+w.size()-1);
  file_d.read(d(), file_d.frames());
  file_x.read(x(), file_x.frames());

  
  // Plot x-axis buffers
  Buffer<real> t(y.size());
  for (idx i=0; i < t.size(); ++i)
    {
      t[i] = i * T_sampling;
      //f[i] = i * FFT_df;
    }


  idx x_size = x.size();
  idx d_size = d.size();
  idx w_size = w.size();
  idx y_size = y.size();

  Buffer<real> e(y), mu(y);
  Buffer<real> e_rms_history(1000); 
  real emu;

  Gnuplot pw, py, pe, pe_rms;
  pw.set_labels("Coefficients", "Weight value");
  py.set_labels("t (s)", "Amplitude");
  pe.set_labels("Step", "Error");
  
  // Skip initial partial convolution for now 
  //           (implement here)
  //
  /*
  for (idx n=0; n < w_size-1; ++n)
    {
      // This loop performs two operations:
      // y[n] = w^T[n] . x[n]
      // mu[n] = x^T[n] . x[n]
      for (idx i=0; i < n+1; ++i)
	{
	  y[n] += w[i]*x[n-i]; 	 

	  //	  mu[n] += x[n-i]*x[n-i];
	}

      e[n] = d[n] - y[n];

      mu[n] = array_ops::energy(x(), n+1);

      emu = e[n] * mu[n];
      for (idx i=0; i < n+1; ++i)
	w[i] += emu * x[n-i];

      printf("emu=%f E(w)=%f \n", emu, array_ops::energy(w(),w.size()));

      pw.replot(w(),w.size(), "w");
      py.replot(t(),y(),y.size(), "y");
      pe.replot(e(),e.size(), "e");
    }
  */


  pe_rms.set_labels("Step", "Error rms");
  
  int repetitions = 0;


  // Reuse the energies of the initial signal for a static case (STATIC)
  for (idx n=w_size-1; n < x_size; ++n)
    mu[n] = 1/array_ops::energy(&x[n-w_size+1], w_size);
 reconverge:

  y.clear();
  e.clear();
  //  mu.clear();
  


  for (idx n=w_size-1; n < x.size(); ++n)
    {
      // This loop performs two operations:
      // y[n] = w^T[n] . x[n]
      // mu[n] = x^T[n] . x[n]
      for (idx i=0; i < w_size; ++i)
	{
	  y[n] += w[i]*x[n-i]; 	 

	  // mu[n] += x[n-i]*x[n-i];
	}

      e[n] = d[n] - y[n];

      // Dynamic mu (moved outside the loop for re-convergence)
      // mu[n] = 1/array_ops::energy(&x[n-w_size+1], w_size);      

      emu = e[n] * mu[n];
      for (idx i=0; i < w_size; ++i)
	w[i] += emu * x[n-i];

      if (n%10000 == 0)
	{
	  if (n%50000 == 0)
	    pw.replot(w(),w.size(), "w");
	  /*
	  if (n%50000==0)
	    {
	      py.replot(&t[w_size-1],y(),n-w_size+1, "y");
	      pe.replot(&e[w_size-1],n-w_size+1, "e");
	      usleep(100000);
	    }
	  */
	  cout << n << "/" << y_size << endl;
	}
      
    }

  pw.replot(w(),w.size(), "w");
  py.replot(t(),y(),y.size(), "y");
  //  pe.replot(e(),e.size(), "e");

  e_rms_history[repetitions] = array_ops::rms(e(),e.size());
  repetitions += 1;
  
  pe_rms.replot(e_rms_history(), repetitions, "e_rms");

  cout << "Repeat convergence procedure?\n";
  bool yes;
  cin >> yes;
  if (yes)
    goto reconverge;
  
  // Skip final partial convolution for now 
  //           (implement here)
  //



  wav::write_mono (argv[3], w(), w.size(), sample_rate_Hz);

  /*
  ph.set_labels("t (s)", "Amplitude");
  ph.plot_xy(t(), h(), FFT_N, "h(t)");
  pM.set_labels("f (Hz)", "Magnitude");
  pM.cmd("set logscale y");
  HC2magnitude(FFT_N, H(), M());
  //  pM.plot_xy(&f[1], &M[1], FFT_N/2, "|H(f)| AC");
  pM.plot_xy(f(), M(), FFT_N/2, "|H(f)|");
  */


  /*
  HC2magnitude(FFT_N, IH(), MIH());
  pMIH.set_labels("f (Hz)", "Magnitude");
  pMIH.cmd("set logscale y");
  pMIH.plot_xy(f(), MIH(), h_size/2, "|H^{-1}(f)|");

  fftw_execute(backwards_plan);

  pih.set_labels("t (s)", "Amplitude");
  pih.plot_xy(t(), ih(), h_size, "h^{-1}");
  */

  if (argc > 5)
    wait();
  puts("\nSuccess!");
  return 0;
}
