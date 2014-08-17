#ifndef matrix_access_h__
#define matrix_access_h__

#include "cuda_extend.h"

/* 
   We'll use macros to access elements from 
   the matrices always in the same way. 

   Implicit dependency of N is not very safe.
   Explicit dependency forces user knowledge.
*/

/*
  M* methods access matrices directly.

  I* methods return only the index.
*/


/* Standard 1D matrix allocation */



// by some strange reason, using the function instead of the macro yields better performance in CUDA   (In CPU should be the other way around)
/*
 // DON'T USE FUNCTIONS!! IT'S DANGEROUS::: IT REPORTS ABSOLUTELY NO ERRORS EVEN WHEN YOU USE IT TO RECEIVE VALUES: M(m,i,j,N) = 3 which doesnt work because only the value is returned
template <class T>
_cub_ T Mcols(T *m, const uint i, const uint j, const uint N) { return m[i+N*j]; }
_cub_ uint Icols(const uint i, const uint j, const uint N) { return i+N*j; }
*/

#define Mfast(matrix,i,j,N) matrix[i+(N)*(j)]
#define Ifast(       i,j,N)       (i+(N)*(j))


#define Mslow(matrix,i,j,N) matrix[j+(N)*(i)]
#define Islow(       i,j,N)       (j+(N)*(i))


/* Matrix separated into even and odd elements */

// MsplitSlow
#define Msplit1(matrix,i,j,N) matrix[(i)*(N) + ((j)/2) + ((N)/2)*((j)%2)]
#define Isplit1(       i,j,N)       ((i)*(N) + ((j)/2) + ((N)/2)*((j)%2))

// MsplitFast
#define Msplit2(matrix,i,j,N) matrix[(j)*(N) + ((i)/2) + ((N)/2)*((i)%2)]
#define Isplit2(       i,j,N)       ((j)*(N) + ((i)/2) + ((N)/2)*((i)%2))




// To use this accessors, in the source file choose one of them before calling main


#endif
