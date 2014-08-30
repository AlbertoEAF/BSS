#ifndef WINDOWS_H__
#define WINDOWS_H__

#include <cmath>
#include "types.h"

// This functions must be called with n<=N





real Rectangular(idx n, idx N)
{
  return 1;
}


// Hamming family

real Hann(idx n, idx N) 
{ 
  return 0.5 * (1.0 - std::cos(_2Pi*n/(N-1.0))); 
}
real Hamming0(idx n, idx N) 
{ 
  return 0.53836 + 0.46164*std::cos(_2Pi*n/(N-1.0)); 
}
real Hamming2digits(idx n, idx N) 
{ 
  return 0.54 - 0.46*std::cos(_2Pi*n/(N-1.0));
}
real Hamming(idx n, idx N) 
{ 
  static const real alpha = 25.0/46.0; 
  static const real beta  = 21.0/46.0;

  return alpha - beta*std::cos(_2Pi*n/(N-1.0));
}
real weakHamming(idx n, idx N) 
{ 
  static const real alpha = 25.0/46.0; 
  static const real beta  = 21.0/46.0 - 0.1;

  return alpha - beta*std::cos(_2Pi*n/(N-1.0));
}
real weakerHamming(idx n, idx N) 
{ 
  static const real alpha = 25.0/46.0; 
  static const real beta  = 21.0/46.0 - 0.2;

  return alpha - beta*std::cos(_2Pi*n/(N-1.0));
}
real HammingEquiripple(idx n, idx N) 
{ 
  return 0.53836 - 0.46164*std::cos(_2Pi*n/(N-1.0));
}
real myHamming(idx n, idx N) 
{ 
  return 0.46164 - 0.46164*std::cos(_2Pi*n/(N-1.0));
}

/// Blackman family

real Blackman(idx n, idx N) 
{ 
  static const real a0 = 7938.0/18608.0; 
  static const real a1 = 9240.0/18608.0;
  static const real a2 = 1430.0/18608.0;

  return a0 - a1*std::cos(_2Pi*n/(N-1.0)) + a2*std::cos(4*M_PI*n/(N-1.0));
}
real Nutall(idx n, idx N) 
{ 
  static const real a0 = 0.355768;
  static const real a1 = 0.487396;
  static const real a2 = 0.144232;
  static const real a3 = 0.012604;

  return a0 - a1*std::cos(_2Pi*n/(N-1.0)) + a2*std::cos(4*M_PI*n/(N-1.0)) - a3*std::cos(6*M_PI*n/(N-1.0));
}
real BlackmanNutall(idx n, idx N) 
{ 
  static const real a0 = 0.3635819;
  static const real a1 = 0.4891775;
  static const real a2 = 0.1365995;
  static const real a3 = 0.0106411;

  return a0 - a1*std::cos(_2Pi*n/(N-1.0)) + a2*std::cos(4*M_PI*n/(N-1.0)) - a3*std::cos(6*M_PI*n/(N-1.0));
}


// Window utility functions

void build_window(Buffer<real> &W, real (*Wfunction)(idx n, idx N))
{
  idx N = W.size();
  for (idx n=0; n < N; ++n)
    W[n] = Wfunction(n,N);
}

void select_window(const std::string &window, Buffer<real> &W)
{
  printf(YELLOW "W = %s\n" NOCOLOR, window.c_str());

  if (window == "Hamming0")
    build_window(W,Hamming0);      

  else if (window == "myHamming")
    build_window(W,myHamming);      

  else if (window == "Hamming")
    build_window(W,Hamming);      

  else if (window == "weakHamming")
    build_window(W,weakHamming);      

  else if (window == "weakerHamming")
    build_window(W,weakerHamming);      

  else if (window == "Hamming2digits")
    build_window(W,Hamming2digits);      

  else if (window == "HammingEquiripple")
    build_window(W,HammingEquiripple);      

  else if (window == "Hann")
    build_window(W,Hann);

  else if (window == "Blackman")
    build_window(W,Blackman);

  else if (window == "Nutall")
    build_window(W,Nutall);

  else if (window == "BlackmanNutall")
    build_window(W,BlackmanNutall);

  else if (window == "Rect" || window == "Rectangular")
    build_window(W,Rectangular);
    
  else
    {
      printf(RED "Window %s is not available.\n" NOCOLOR, window.c_str());
      std::exit(1);
    }
}

#endif // WINDOWS_H__
