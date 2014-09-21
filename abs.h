#ifndef ABS_H__
#define ABS_H__

#include <cmath>
//#include "types.h"
/*
real abs (real a, real b) { return std::sqrt(a*a + b*b); }
real abs2(real a, real b) { return a*a + b*b; }
*/

template <class T> T abs (T a, T b) { return std::sqrt(a*a + b*b); }
template <class T> T abs2(T a, T b) { return a*a + b*b; }




#endif // ABS_H__
