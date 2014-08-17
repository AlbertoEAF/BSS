#ifndef COMPLEX_H__
#define COMPLEX_H__

#include "cuda_extend.h" // allows compatibility with cuda and normal compilation for when it's unavailable
#include <iostream>
#include <cmath>

// Para fazer template das funcoes friend
template <class T> class Complex;
template <class T> std::ostream & operator << (std::ostream &, const Complex<T> &);

template <class T> _cudab_ Complex<T> operator+ (const Complex<T> &, const Complex<T> &);
template <class T> _cudab_ Complex<T> operator- (const Complex<T> &, const Complex<T> &);
template <class T> _cudab_ Complex<T> operator* (const Complex<T> &, const Complex<T> &);
template <class T> _cudab_ Complex<T> operator/ (const Complex<T> &, const Complex<T> &);

// NOTE: There is no explicit implementation for int*Complex because it may lead to wrong results if one is not careful!


// Specialized functions with best performance for each platform

template <class T> _cudab_ Complex<T> operator+ (const Complex<T> &a, const T b) 
{ 
#ifdef __CUDA_ARCH__ 
  return a+Complex<T>(b,0);         // More FLOPS, less registers --> Better for GPU
#else 
  return Complex<T>(a._Re+b,a._Im); // Less FLOPS, more registers --> Better for CPU
#endif 
}
template <class T> _cudab_ Complex<T> operator+ (const T a, const Complex<T> &b) 
{ 
#ifdef __CUDA_ARCH__
  return Complex<T>(a,0)+b;
#else
  return Complex<T>(b._Re+a,b._Im); 
#endif
}
template <class T> _cudab_ Complex<T> operator- (const Complex<T> &a, const T b) 
{ 
#ifdef __CUDA_ARCH__
  return a-Complex<T>(b,0);
#else
 return Complex<T>(a._Re-b,a._Im);
#endif
}
template <class T> _cudab_ Complex<T> operator- (const T a, const Complex<T> &b) 
{ 
#ifdef __CUDA_ARCH__
  return Complex<T>(a,0)-b; 
#else
  return Complex<T>(a-b._Re,-b._Im);
#endif
}
template <class T> _cudab_ Complex<T> operator* (const Complex<T> &a, const T b) 
{ 
#ifdef __CUDA_ARCH__
  return a*Complex<T>(b,0);
#else
  return Complex<T>(a._Re*b,a._Im*b);
#endif
}
template <class T> _cudab_ Complex<T> operator* (const T a, const Complex<T> &b) 
{ 
#ifdef __CUDA_ARCH__
  return Complex<T>(a,0)*b;
#else
  return Complex<T>(b._Re*a,b._Im*a);
#endif
}
template <class T> _cudab_ Complex<T> operator/ (const Complex<T> &a, const T b) 
{ 
  return Complex<T>(a._Re/b,a._Im/b); // full division is very expensive (14 FLOPS) --> Best to reduce FLOPS in both platforms
}


template <class T> _cudab_ Complex<T> operator/ (const T a, const Complex<T> &b) { return Complex<T>(a,0)/b; } // hard-coding provides 11FLOPS vs. 14 FLOPS, not very relevant



/**********************************************************************************************/


// Real and Imaginary operations at the end of the file









/*
      -- ONLY AVAILABLE IN CUDA PROGRAMS --

  MOTIVATION: 
  CUDA doesn't allow constructors for __constant__

  This struct is useful for constant memory uses 
  since you no longer need to split Re and Im 
  in constant memory as 2 entities of type T,
  since you can then use the Complex class since 
  it now has a constructor from this __Complex type.

  Also, since the internal data structure on both 
  __Complex and Complex is the same, you can use
  memcpy's from one to another.
*/
CUDA
(
 template <class T>
 struct __Complex {
   
   _cudab_ void init(T re, T im) { Re = re; Im = im; }
   
   T Re;
   T Im;
 };
)


// For now Real and Imaginary only serve to improve the performance of constructors on the device if __constant__ memory is loaded and only Real or Complex is needed but may be later used to improve performance on products and sums, etc.

// Also, __Real and __Imaginary is useful for constant memory by the same reasons

template <class T>
struct __Real { 

  _cudab_ void init(T value) { x = value; }

  T x; 
};

template <class T>
struct __Imaginary { 

  _cudab_ void init(T value) { x = value; }

  T x; 
};


template <class T>
struct Real { 

  _cudab_ Real (T value) : x(value) {}
  _cudab_ Real (__Real<T> r) : x(r.x) {}
  T x; 
};

template <class T>
struct Imaginary { 

  _cudab_ Imaginary (T value) : x(value) {}
  _cudab_ Imaginary (__Imaginary<T> r) : x(r.x) {}
  T x; 
};



template <class T>
class ALIGN(sizeof(T)*2) Complex {  

  friend std::ostream & operator << <>(std::ostream &, const Complex<T> &);

  friend Complex<T> _cudab_ operator+ <>(const Complex<T> &, const Complex<T> &);
  friend Complex<T> _cudab_ operator- <>(const Complex<T> &, const Complex<T> &);
  friend Complex<T> _cudab_ operator* <>(const Complex<T> &, const Complex<T> &);  
  friend Complex<T> _cudab_ operator/ <>(const Complex<T> &, const Complex<T> &);

  friend Complex<T> _cudab_ operator+ <>(const Complex<T> &, const T );
  friend Complex<T> _cudab_ operator+ <>(const T , const Complex<T> &);
  friend Complex<T> _cudab_ operator- <>(const Complex<T> &, const T );
  friend Complex<T> _cudab_ operator- <>(const T , const Complex<T> &);
  friend Complex<T> _cudab_ operator* <>(const Complex<T> &, const T );
  friend Complex<T> _cudab_ operator* <>(const T , const Complex<T> &);
  friend Complex<T> _cudab_ operator/ <>(const Complex<T> &, const T );
  friend Complex<T> _cudab_ operator/ <>(const T , const Complex<T> &);
  
public:
  _cudab_ Complex ()           : _Re(0),  _Im(0)    {}
  _cudab_ Complex (T Re, T Im) : _Re(Re), _Im(Im)   {}
  _cudab_ Complex (int Re)     : _Re(Re), _Im(0)    {}
  _cudab_ Complex (float Re)   : _Re(Re), _Im(0)    {}

  _cudab_ Complex (Real<T> R)      : _Re(R.x), _Im(0)   {}
  _cudab_ Complex (Imaginary<T> I) : _Re(0)  , _Im(I.x) {}


  /* 
       -- ONLY AVAILABLE WITH CUDA --

     This is useful for constant memory. 
     Declare an object of __Complex type as __constant__ 
     and in the device function create a Complex object from this one.
  */
  CUDA(_cudab_ Complex (__Complex<T> __c) { _Re = __c.Re; _Im = __c.Im; })

  // For Texture Memory
CUDA(  
     _cudab_ Complex (float2  f) { _Re = f.x; _Im = f.y; } 

     _cudad_ Complex (int4 f) 
     {
#ifdef __CUDACC__
#if __CUDA_ARCH__ < 130
#warning Double precision textures unavailable in CUDA architectures < 1.2. Complex(int4) is forbidden and will yield 'undefined __hiloint2double'
#endif
#endif

       _Re = __hiloint2double(f.y,f.x); 
       _Im = __hiloint2double(f.w,f.z);   
     }

    )

  #if __CUDA_ARCH__ >= 130
  _cudab_ Complex (double t)   : _Re(t),  _Im(0)    {}
 #endif


  _cudab_ inline T Re () const {return _Re;};
  _cudab_ inline T Im () const {return _Im;}
  _cudab_ inline void Re(T Re) { _Re = Re; }
  _cudab_ inline void Im(T Im) { _Im = Im; }

  _cudab_ inline Complex<T> conjugate () { return Complex<T>(_Re, -_Im); }

  _cudab_ inline T abs () const { return sqrt(_Re*_Re + _Im*_Im); }
  _cudab_ inline T phase () const { return tan(_Im/_Re); }
  
  _cudab_ inline T abs_squared() { return _Re*_Re + _Im*_Im; }

//private: // gives too much work -- we will go back to it later
  T _Re;
  T _Im;
};


template <class T> _cudab_ 
Complex<T> operator+ (const Complex<T> &a, const Complex<T> &b)
{
  return Complex<T>(a._Re+b._Re, a._Im+b._Im);
}

template <class T> _cudab_ 
Complex<T> operator- (const Complex<T> &a, const Complex<T> &b)
{
  return Complex<T>(a._Re-b._Re, a._Im-b._Im);
}

template <class T> _cudab_ 
Complex<T> operator* (const Complex<T> &a, const Complex<T> &b)
{
  return Complex<T>(a._Re*b._Re - a._Im*b._Im, a._Re*b._Im + a._Im*b._Re); 
}

template <class T> _cudab_ 
Complex<T> operator/ (const Complex<T> &a, const Complex<T> &b)
{
  return Complex<T>((a._Re*b._Re + a._Im*b._Im)/(b._Re*b._Re + b._Im*b._Im), (a._Im*b._Re - a._Re*b._Im)/(b._Re*b._Re + b._Im*b._Im));
}


template <class T>
std::ostream & operator << (std::ostream &out, const Complex<T> &c)
{
  std::cout << c._Re;
  if (c._Im >=0)
    std::cout << "+";
  std::cout << c._Im << "i";

  return out;
}




// Complex & Real
template <class T> _cudab_ Complex<T> operator+ (const Complex<T> &a, const Real<T> &b) { return Complex<T>(a._Re+b.x,a._Im); }
template <class T> _cudab_ Complex<T> operator+ (const Real<T> &a, const Complex<T> &b) { return Complex<T>(b._Re+a.x,b._Im); }
template <class T> _cudab_ Complex<T> operator- (const Complex<T> &a, const Real<T> &b) { return Complex<T>(a._Re-b.x,a._Im); }
template <class T> _cudab_ Complex<T> operator- (const Real<T> &a, const Complex<T> &b) { return Complex<T>(a.x-b._Re,-b._Im); }

template <class T> _cudab_ Complex<T> operator* (const Complex<T> &a, const Real<T> &b) { return Complex<T>(a._Re*b.x,a._Im*b.x); }
template <class T> _cudab_ Complex<T> operator* (const Real<T> &a, const Complex<T> &b) { return Complex<T>(b._Re*a.x,b._Im*b.x); }
/*
template <class T> _cudab_ Complex<T> operator/ (const Complex<T> &a, const T &b) { return a/Complex<T>(b,0); }
template <class T> _cudab_ Complex<T> operator/ (const T &a, const Complex<T> &b) { return Complex<T>(a,0)/b; }
*/

// Complex & Imaginary
template <class T> _cudab_ Complex<T> operator+ (const Complex<T> &a, const Imaginary<T> &b) { return Complex<T>(a._Re,a._Im+b.x); }
template <class T> _cudab_ Complex<T> operator+ (const Imaginary<T> &a, const Complex<T> &b) { return Complex<T>(b._Re,b._Im+a.x); }
template <class T> _cudab_ Complex<T> operator- (const Complex<T> &a, const Imaginary<T> &b) { return Complex<T>(a._Re,a._Im-b.x); }
template <class T> _cudab_ Complex<T> operator- (const Imaginary<T> &a, const Complex<T> &b) { return Complex<T>(-b._Re,a.x-b._Im); }

template <class T> _cudab_ Complex<T> operator* (const Complex<T> &a, const Imaginary<T> &b) { return Complex<T>(-a._Im*b.x,a._Re*b.x); }
template <class T> _cudab_ Complex<T> operator* (const Imaginary<T> &a, const Complex<T> &b) { return Complex<T>(-b._Im*a.x,b._Re*a.x); }
/*
template <class T> _cudab_ Complex<T> operator/ (const Complex<T> &a, const T &b) { return a/Complex<T>(b,0); }
template <class T> _cudab_ Complex<T> operator/ (const T &a, const Complex<T> &b) { return Complex<T>(a,0)/b; }
*/

// Complex & __Imaginary
template <class T> _cudab_ Complex<T> operator+ (const Complex<T> &a, const __Imaginary<T> &b) { return Complex<T>(a._Re,a._Im+b.x); }
template <class T> _cudab_ Complex<T> operator+ (const __Imaginary<T> &a, const Complex<T> &b) { return Complex<T>(b._Re,b._Im+a.x); }
template <class T> _cudab_ Complex<T> operator- (const Complex<T> &a, const __Imaginary<T> &b) { return Complex<T>(a._Re,a._Im-b.x); }
template <class T> _cudab_ Complex<T> operator- (const __Imaginary<T> &a, const Complex<T> &b) { return Complex<T>(-b._Re,a.x-b._Im); }

template <class T> _cudab_ Complex<T> operator* (const Complex<T> &a, const __Imaginary<T> &b) { return Complex<T>(-a._Im*b.x,a._Re*b.x); }
template <class T> _cudab_ Complex<T> operator* (const __Imaginary<T> &a, const Complex<T> &b) { return Complex<T>(-b._Im*a.x,b._Re*a.x); }





#endif
 
