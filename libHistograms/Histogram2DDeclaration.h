// http://www.phyast.pitt.edu/~zov1/gnuplot/html/histogram.html // gnuplot cool drawings

#ifndef HISTOGRAM_2D_DECLARATION_H__
#define HISTOGRAM_2D_DECLARATION_H__

#ifndef HISTOGRAM_2D_H__
#include <stdio.h>
#include <iostream>
#endif

#include "Histogram.h"

#include "Matrix.h"

#include "color_codes.h" // For Histogram2D::print()

#include <fstream>
#include <cmath> // std::isnan // same in standard C library

// Fot the optional splot() command. Gnuplot_ipp must be added to the main project before Histogram is included.
class Gnuplot;

/// Auxiliary class for bin indexes: Point2D 
template <class T> class Point2D;

template <class T> std::ostream &operator << (std::ostream &output, Point2D<T> &p);
/* End of auxiliary class: Point2D */


// Histogram2D class ////////////////////////////////////////////

#ifndef HISTOGRAM_BOUNDS_TYPE__
#define HISTOGRAM_BOUNDS_TYPE__
namespace HistogramBounds {
  enum Type {Boundless, DiscardBeyondBound, Bounded};
}
#endif // HISTOGRAM_BOUNDS_TYPE__

template <class T> class Histogram2D;
template <class T> std::ostream &operator << (std::ostream &, Histogram2D<T> &);
template <class T> std::istream &operator >> (std::istream &, Histogram2D<T> &);

template <class T>
class Histogram2D 
{
  friend std::ostream &operator << <>(std::ostream &, Histogram2D<T> &);
  friend std::istream &operator >> <>(std::istream &, Histogram2D<T> &);

 public:
  // (dx,dy) might undergo small changes so that the Histogram bins fit the area perfectly
  Histogram2D(double bin_dx, double bin_dy, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type);
  //  Histogram2D(size_t bins_x, size_t bins_y, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type);
  Histogram2D(const Histogram2D<T> &copy);
  ~Histogram2D() { delete _m; }

  inline T & bin(size_t binx, size_t biny); // Access to bin directly by bin coordinates (faster than Histogram::(x,y))
  inline T & guarantee_bin (size_t binx, size_t biny); // Makes sure a bin is found (runtime assert)
  inline T & operator() (double x, double y);          // Access to bin by (x,y) coordinates

  // (dx,dy) might undergo small changes so that the Histogram bins fit the area perfectly
  //  inline void reshape (double bin_dx, double bin_dy, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type); 
  inline void reshape (size_t bins_x, size_t bins_y, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type);
  inline void clear () { _m->clear(); }

  bool get_bin_index  (double x, double y, size_t &bin_x_index, size_t &bin_y_index);
  void get_bin_center_(size_t binx, size_t biny, T *cx, T *cy);
  Point2D<double> get_bin_center(size_t binx, size_t biny); // Get the bin (i,j)'s center (x,y) coordinates
  Point2D<double> get_bin_center(Point2D<size_t> &bin_coords);

  Matrix<T> * raw() { return _m; }

  int write_to_gnuplot_pm3d_data       (const char *filepath);
  int write_to_gnuplot_pm3d_binary_data(const char *filepath);

  size_t xbins() { return _xbins; }
  size_t ybins() { return _ybins; }  
  size_t bins() { return _xbins * _ybins; }
  double dx() { return _dx; }
  double dy() { return _dy; }
  double xmin() { return _xmin; }
  double xmax() { return _xmax; }
  double ymin() { return _ymin; }
  double ymax() { return _ymax; }

  Histogram2D<T> & operator = (Histogram2D<T> &other);
  void operator += (Histogram2D<T> &other);
  void operator -= (Histogram2D<T> &other);
  void operator *= (Histogram2D<T> &other);

  void marginal_x(Buffer<T> &marginal);
  void marginal_y(Buffer<T> &marginal);

  void marginal_x(Histogram<T> &marginal);
  void marginal_y(Histogram<T> &marginal);

  void print_format() { printf(YELLOW "Histogram(%lu,%lu) : x=[%g,%g] y=[%g,%g] (dx,dy)=(%g,%g)\n" NOCOLOR,_xbins,_ybins,_xmin,_xmax,_ymin,_ymax,_dx,_dy); }

  void plot(Gnuplot &p, const char * title);

  void smooth_add(T value, double x, double y, double smooth_dx, double smooth_dy);

  // If success==0 smoothing_size < bin: no point in smoothing.
  Matrix<T> gen_gaussian_kernel(T stddev_x, T stddev_y, bool *success); 

  void kernel_convolution(Matrix<T> &kernel, Matrix<T> &m);


 private:
  Matrix<T> *_m;
  double _dx, _dy, _xmin, _xmax, _ymin, _ymax;
  size_t _xbins, _ybins;
  HistogramBounds::Type _bound_type;
  T _dummy; // To return a bin reference with value=0 when there's out of bounds access in DiscardBeyondBound allocation mode
};


#endif // HISTOGRAM_2D_DECLARATION_H__
