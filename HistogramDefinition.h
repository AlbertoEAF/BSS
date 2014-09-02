// http://www.phyast.pitt.edu/~zov1/gnuplot/html/histogram.html // gnuplot cool drawings

#ifndef HISTOGRAM_H__
#define HISTOGRAM_H__


#include <iostream> // Just due to template instantiation
#include <stdlib.h>
#include <stdio.h>

#include "Buffer.h"

#include "color_codes.h" // For Histogram::print()

#include <fstream>
#include <cmath> // std::isnan // same in standard C library

// Fot the optional splot() command. Gnuplot_ipp must be added to the main project before Histogram is included.
class Gnuplot;


// Histogram class ////////////////////////////////////////////

#ifndef HISTOGRAM_BOUNDS_TYPE__
#define HISTOGRAM_BOUNDS_TYPE__
namespace HistogramBounds {
  enum Type {Boundless, DiscardBeyondBound, Bounded};
}
#endif // HISTOGRAM_BOUNDS_TYPE__

template <class T> class Histogram;
template <class T> std::ostream &operator << (std::ostream &, Histogram<T> &);
template <class T> std::istream &operator >> (std::istream &, Histogram<T> &);

template <class T>
class Histogram 
{
  friend std::ostream &operator << <>(std::ostream &, Histogram<T> &);
  friend std::istream &operator >> <>(std::istream &, Histogram<T> &);

 public:
  // (dx,dy) might undergo small changes so that the Histogram bins fit the area perfectly
  Histogram(double bin_dx, double x_min, double x_max, HistogramBounds::Type bounds_type);
  //  Histogram(size_t bins_x, size_t bins_y, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type);
  Histogram(const Histogram<T> &copy);
  ~Histogram();

  T & bin(size_t ibin); // Access to bin directly by bin coordinates (faster than Histogram::(x,y))
  T & guarantee_bin (size_t ibinx); // Makes sure a bin is found (runtime assert)
T & operator() (double x);          // Access to bin by (x,y) coordinates

  // (dx,dy) might undergo small changes so that the Histogram bins fit the area perfectly
  //  inline void reshape (double bin_dx, double bin_dy, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type); 
  void reshape (size_t bins, double min, double max, HistogramBounds::Type bounds_type);
  inline void clear () { _m->clear(); }

  bool  get_bin_index (double x, size_t &ibin);
  double get_bin_center(size_t bin);

  Buffer<T> * raw() { return _m; }

  size_t bins() { return _bins; }
  double dx() { return _dx; }
  double min() { return _min; }
  double max() { return _max; }

  Histogram<T> & operator = (Histogram<T> &other);
  void operator += (Histogram<T> &other);
  void operator -= (Histogram<T> &other);
  void operator *= (Histogram<T> &other);

  void print_format() { printf(YELLOW "Histogram(%lu) : x=[%g] y=[%g] (dx)=(%g)\n" NOCOLOR,_bins,_min,_max,_dx); }

  void plot(Gnuplot &p, const char * title);

  void smooth_add(T value, double x, double smooth_dx);

  Buffer<T> gen_gaussian_kernel(T stddev);

  void kernel_convolution(Buffer<T> &kernel, Buffer<T> &m);

 private:
  Buffer<T> *_m;
  double _dx, _min, _max;
  size_t _bins;
  HistogramBounds::Type _bound_type;
  T _dummy; // To return a bin reference with value=0 when there's out of bounds access in DiscardBeyondBound allocation mode
};

#endif // HISTOGRAM_H__
