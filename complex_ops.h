#ifndef COMPLEX_OPS_H__
#define COMPLEX_OPS_H__


#include "abs.h" // Includes "types.h"
#include "Buffer.h"

/// Specialization for even sizes ( read http://www.fftw.org/doc/The-Halfcomplex_002dformat-DFT.html )
void evenHC2magnitude(int samples, real *hc, real *magnitude)
{
  magnitude[0] = hc[0];
  idx I = samples/2;
  for (idx i=1; i < I; ++i)
    magnitude[i] = abs(hc[i], hc[samples-i]); // Indexing not true for odd samples!!!
}

void evenHC2power(int samples, real *hc, real *power)
{
  power[0] = hc[0];
  idx I = samples/2;
  for (idx i=1; i < I; ++i)
    power[i] = abs2(hc[i], hc[samples-i]); // Indexing not true for odd samples!!!
}

void evenHC2magnitude(Buffer<real> &hc, Buffer<real> &magnitude)
{
  magnitude[0] = hc[0];
  idx I = hc.size()/2, samples = hc.size();
  for (idx i=1; i < I; ++i)
    magnitude[i] = abs(hc[i], hc[samples-i]); // Indexing not true for odd samples!!!
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

// z = z1/z2
inline void complex_divide(real re1, real im1, real re2, real im2, real *re, real *im)
{
  real denominator = re2*re2 + im2*im2;

  *re = (re1*re2 + im1*im2) / denominator;
  *im = (im1*re2 - re1*im2) / denominator;
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

  const idx max_i = size/2;
  for (idx i=1; i < max_i; ++i)
    complex_multiply(z1[i], z1[size-i],
		     z2[i], z2[size-i],
		     &z[i], &z[size-i]);
}


#endif // COMPLEX_OPS_H__
