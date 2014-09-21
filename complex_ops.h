#ifndef COMPLEX_OPS_H__
#define COMPLEX_OPS_H__


#include <iostream>

#include "types.h"

#include "abs.h"
#include "Buffer.h"



/// Specialization for even sizes ( read http://www.fftw.org/doc/The-Halfcomplex_002dformat-DFT.html )
void evenHC2magnitude(int samples, real *hc, real *magnitude);

void evenHC2power(int samples, real *hc, real *power);

void evenHC2magnitude(Buffer<real> &hc, Buffer<real> &magnitude);

void evenHC2power(Buffer<real> &hc, Buffer<real> &power);

/**
   Z = Z1*Z2
   @param[in] re1 - Re{Z1}
   @param[in] im1 - Im{Z1}
   @param[in] re2 - Re{Z2}
   @param[in] im2 - Im{Z2}
   @param[out] re - Re{Z}
   @param[out] im - Im{Z}
*/
void complex_multiply(real re1, real im1, real re2, real im2, real *re, real *im);

// z = z1/z2
void complex_divide(real re1, real im1, real re2, real im2, real *re, real *im);

/**
   HalfComplex representation multiply
   @param[in] z1 - Input  HC array
   @param[in] z2 - Input  HC array
   @param[out] z - Output HC array
   @param[in] size - Size of the HC array

   @warn: ONLY FOR EVEN TRANSFORMATIONS!!!
*/
void hc_multiply (real *z1, real *z2, real *z, idx size);

#endif // COMPLEX_OPS_H__
