// http://www.phyast.pitt.edu/~zov1/gnuplot/html/histogram.html // gnuplot cool drawings

#ifndef HISTOGRAM_2D_H__
#define HISTOGRAM_2D_H__

#include "Histogram.h"

#include "Matrix.h"

#include "color_codes.h" // For Histogram2D::print()

#include <fstream>
#include <cmath> // std::isnan // same in standard C library

// Fot the optional splot() command. Gnuplot_ipp must be added to the main project before Histogram is included.
class Gnuplot;

/// Auxiliary class for bin indexes: Point2D 
template <class T> class Point2D
{
 public:
 Point2D(T a=0, T b=0) : x(a), y(b) {};

  T x;
  T y;
};
template <class T> std::ostream &operator << (std::ostream &output, Point2D<T> &p)
{
  output << "(" << p.x << "," << p.y << ")";
  return output; // allows chaining
}
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

  int  get_bin_index (T x, T y, size_t &bin_x_index, size_t &bin_y_index);
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

  void fill_marginal_x(Histogram<T> &marginal);
  void fill_marginal_y(Histogram<T> &marginal);

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

template <class T>
Histogram2D<T>::Histogram2D(double bin_dx, double bin_dy, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type)
: _m(NULL)
{
  // Adjust so that they match
  size_t bins_x = (x_max-x_min)/bin_dx;
  size_t bins_y = (y_max-y_min)/bin_dy;
  
  reshape(bins_x, bins_y, x_min, x_max, y_min, y_max, bounds_type);
}


/*
// Needs to have a  different signature from the other constructor
template <class T>
Histogram2D<T>::Histogram2D(size_t bins_x, size_t bins_y, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type)
: _m(NULL)
{
  reshape(bins_x, bins_y, x_min, x_max, y_min, y_max, bounds_type);
}
*/

template <class T>
Histogram2D<T>::Histogram2D(const Histogram2D &cpy)
: _m(NULL), _dx(cpy._dx), _dy(cpy._dy), _xmin(cpy._xmin), _xmax(cpy._xmax), _ymin(cpy._ymin), _ymax(cpy._ymax), _xbins(cpy._xbins), _ybins(cpy._ybins), _bound_type(cpy._bound_type)
{
  _m = new Matrix<T>(cpy._m);
}

/*
template <class T>
void Histogram2D<T>::reshape(double bin_dx, double bin_dy, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type)
{
  // Histogram bin count must match the histogram area with the bins size, thus a first calculation is done, then the bin sizes are adjusted so that they fit
  size_t bins_x = (x_max-x_min)/bin_dx;
  size_t bins_y = (y_max-y_min)/bin_dy;

  reshape(bins_x, bins_y, x_min, x_max, y_min, y_max, bounds_type);
}
*/

template <class T>
void Histogram2D<T>::reshape(size_t bins_x, size_t bins_y, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type)
{
  Assert (x_max > x_min && y_max > y_min, "Histogram limits are reversed!");

  _xmin = x_min;
  _xmax = x_max;
  _ymin = y_min;
  _ymax = y_max;
  _xbins = bins_x;
  _ybins = bins_y;

  _dx = (_xmax-_xmin) / (T) _xbins;
  _dy = (_ymax-_ymin) / (T) _ybins;

  _bound_type = bounds_type;

  // Reallocate if already exists
  if (_m)
    delete _m;
  _m = new Matrix<T>(bins_x, bins_y);
}


/// Get read-write access to a bin directly by its coordinates
template <class T>
T & Histogram2D<T>::guarantee_bin(size_t binx, size_t biny)
{
  Guarantee (binx < _xbins && biny < _ybins, "Out of bounds bin access (%lu,%lu) for Histogram2D(%lu,%lu)!", binx,biny, _xbins, _ybins);

  return (*_m)(binx,biny);
}
/// Faster version of guaranteebin() without out of bounds runtime check.
template <class T>
inline T & Histogram2D<T>::bin(size_t binx, size_t biny)
{
  Assert (binx < _xbins && biny < _ybins, "Out of bounds bin access (%lu,%lu) for Histogram2D(%lu,%lu)!", binx,biny, _xbins, _ybins);
  return (*_m)(binx,biny);
}



template <class T>
int Histogram2D<T>::get_bin_index(T x, T y, size_t &bin_x_index, size_t &bin_y_index)
{
  Assert ( ! std::isnan(x) && ! std::isnan(y), "NaN coordinate given (%f,%f) to Histogram2D(%lu,%lu)!", x,y, _xbins,_ybins);

  // add to the boundary bins if the coordinate goes beyond
      if (x < _xmin)
	bin_x_index = 0;
      else if (x >= _xmax)
	bin_x_index = _xbins-1;
      else
	bin_x_index = (x - _xmin)/_dx;

      if (y < _ymin)
	bin_y_index = 0;
      else if (y >= _ymax)
	bin_y_index = _ybins-1;
      else
	bin_y_index = (y - _ymin)/_dy;

  if (_bound_type == HistogramBounds::Boundless) 
    return 1;
  else
    {
      if (x < _xmin || x >= _xmax || y < _ymin || y >= _ymax)
	{
	  if (_bound_type == HistogramBounds::DiscardBeyondBound)
	    return 0;
	  else
	    Guarantee(0, "Out of bounds access (x,y)=(%f,%f) on Bounded Histogram2D(%lu,%lu)!",x,y,_xbins,_ybins);
	}
      else
	return 1;
    }
}

template <class T>
Point2D<double> Histogram2D<T>::get_bin_center(size_t binx, size_t biny)
{
  Guarantee(binx < _xbins && biny < _ybins, "Requested bin(%lu,%lu) larger than Histogram(%lu,%lu) space.", binx,biny, _xbins,_ybins);
  return Point2D<double>(_xmin + _dx*(binx+0.5), _ymin + _dy*(biny+0.5));
}

/// Version that doesn't test if the result is inside bounds
template <class T>
void Histogram2D<T>::get_bin_center_(size_t binx, size_t biny, T *cx, T *cy)
{
  Assert(binx < _xbins && biny < _ybins, "Requested bin(%lu,%lu) larger than Histogram(%lu,%lu) space.", binx,biny, _xbins,_ybins);

  *cx = _xmin + _dx*(binx+0.5);
  *cy = _ymin + _dy*(biny+0.5);
}


template <class T>
Point2D<double> Histogram2D<T>::get_bin_center(Point2D<size_t> &bin_coordinates)
{
  Assert(bin_coordinates.x < _xbins && bin_coordinates.y < _ybins, "Requested bin(%lu,%lu) larger than Histogram(%lu,%lu) space.", bin_coordinates.x,bin_coordinates.y, _xbins,_ybins);
  return Point2D<double>(_xmin + _dx*(bin_coordinates.x+0.5), _ymin + _dy*(bin_coordinates.y+0.5));
}


template <class T>
T & Histogram2D<T>::operator()(double x, double y)
{
  size_t bin_x_index, bin_y_index;
	
  Assert ( ! std::isnan(x) && ! std::isnan(y), "NaN coordinate given (%f,%f) to Histogram2D(%lu,%lu)!", x,y, _xbins,_ybins);

  // add to the boundary bins if the coordinate goes beyond
  if (_bound_type == HistogramBounds::Boundless) 
    {
      if      (x < _xmin)
	bin_x_index = 0;
      else if (x >= _xmax)
	bin_x_index = _xbins-1;
      else
	bin_x_index = (x - _xmin)/_dx;

      if      (y < _ymin)
	bin_y_index = 0;
      else if (y >= _ymax)
	bin_y_index = _ybins-1;
      else
	bin_y_index = (y - _ymin)/_dy;
    }
  else
    {
      if (x < _xmin || x >= _xmax || y < _ymin || y >= _ymax)
	{
	  // Guarantee during runtime that the histogram is not Bounded!
	  Guarantee (_bound_type == HistogramBounds::DiscardBeyondBound,
			 "Out of bounds access (x,y)=(%f,%f) on Bounded Histogram2D(%lu,%lu)!", x,y,_xbins,_ybins);
	  // Return a "virtual bin" with value 0 so that changing it does not change the histogram itself and if it is read yields no counts at all (the value can be changed by the user though). This value will be reset at the next call in the same out of bounds situation on a DiscardBeyondBound Histogram.
	  _dummy = 0;
	  return _dummy;
	}
      else
	{
	  bin_x_index = (x - _xmin)/_dx;
	  bin_y_index = (y - _ymin)/_dy;
	}
    }
  return bin(bin_x_index, bin_y_index);
}


template <class T>
void Histogram2D<T>::operator += (Histogram2D<T> &other)
{
  static const double eps = 0.000001;
  Assert((&other == this) | (other._xbins==_xbins && other._ybins==_ybins && std::abs(other._xmax-_xmax)<eps && std::abs(other._xmin-_xmin)<eps && std::abs(other._ymax-_ymax)<eps && std::abs(other._ymin-_ymin)<eps), "Histograms coordinates don't match.");

  for (size_t i = 0; i < _xbins; ++i)
    for (size_t j = 0; j < _ybins; ++j)
      bin(i,j) += other.bin(i,j);
}

template <class T>
void Histogram2D<T>::operator -= (Histogram2D<T> &other)
{
  static const double eps = 0.000001;
  Assert((&other == this) | (other._xbins==_xbins && other._ybins==_ybins && std::abs(other._xmax-_xmax)<eps && std::abs(other._xmin-_xmin)<eps && std::abs(other._ymax-_ymax)<eps && std::abs(other._ymin-_ymin)<eps), "Histograms coordinates don't match.");

  for (size_t i = 0; i < _xbins; ++i)
    for (size_t j = 0; j < _ybins; ++j)
      bin(i,j) -= other.bin(i,j);
}

template <class T>
void Histogram2D<T>::operator *= (Histogram2D<T> &other)
{
  static const double eps = 0.000001;
  Assert((&other == this) | (other._xbins==_xbins && other._ybins==_ybins && std::abs(other._xmax-_xmax)<eps && std::abs(other._xmin-_xmin)<eps && std::abs(other._ymax-_ymax)<eps && std::abs(other._ymin-_ymin)<eps), "Histograms coordinates don't match.");

  for (size_t i = 0; i < _xbins; ++i)
    for (size_t j = 0; j < _ybins; ++j)
      bin(i,j) *= other.bin(i,j);
}


template <class T>
Histogram2D<T> & Histogram2D<T>::operator = (Histogram2D<T> &other)
{
  if (this == &other)
    return *this;

  static const double eps = 0.000001;
  Assert((&other == this) | (other._xbins==_xbins && other._ybins==_ybins && std::abs(other._xmax-_xmax)<eps && std::abs(other._xmin-_xmin)<eps && std::abs(other._ymax-_ymax)<eps && std::abs(other._ymin-_ymin)<eps), "Histograms coordinates don't match.");

  *_m = *(other._m); // Matrix copy

  return *this;
}
  
  template <class T>
  std::ostream &operator << (std::ostream &output, Histogram2D<T> &hist)
  {
    output << (*hist._m);
    
    return output; // allows chaining
  }

template <class T>
std::istream &operator >> (std::istream &input, Histogram2D<T> &hist)
{
  input >> (*hist._m);

  return input; // allows chaining
}


template <class T>
int Histogram2D<T>::write_to_gnuplot_pm3d_data (const char *filepath)
{
  std::ofstream dat;
  dat.open(filepath);

  if (! dat.is_open())
    {
      puts("Couldn't open file for output!");
      return 0;
    }

  T x, y;
  for (size_t i = 0; i < _xbins; ++i)
    {
      x = _xmin + _dx * (i+0.5); // center point
      for (size_t j = 0; j < _ybins; ++j)
	{
	  y = _ymin + _dy * (j+0.5); // center point
	  dat << x << " " << y << " " << bin(i,j) << std::endl;
	}

      dat << std::endl;
    }

  dat.close();
  return 1;
}

template <class T>
int Histogram2D<T>::write_to_gnuplot_pm3d_binary_data (const char *filepath)
{
  FILE *fh = fopen(filepath, "wb");
  if (fh==NULL) 
    { 
      printf("Couldn't open file for output of Histogram2D(%lu,%lu) binary data!", _xbins, _ybins); 
      return 0;
    }
	
  float x, y, z;
  for (size_t i = 0; i < _xbins; ++i)
    {
      x = _xmin + _dx * (i+0.5); // center point
      for (size_t j = 0; j < _ybins; ++j)
	{
	  y = _ymin + _dy * (j+0.5); // center point
			
	  z = (*_m)(i,j);

	  fwrite(&x, sizeof(x), 1, fh);
	  fwrite(&y, sizeof(y), 1, fh);
	  fwrite(&z, sizeof(z), 1, fh);			
	}
    }

  fclose(fh);
  return 1;
}

template <class T>
void Histogram2D<T>::plot(Gnuplot &p, const char * title)
{
  /* This function cannot blow up the compilation as it is optional 
     so it will be guaranteed at runtime without performance penalties. 
     It will require including gnuplot_ipp.h before Histogram2D.h though. */

#ifdef GNUPLOT_IPP_H__
  p.cmd("set pm3d; unset surface");
  p.splot(_m->raw(), _ybins, _xbins, _xmin, _xmax, _ymin, _ymax, title);
#else
  Guarantee(0, "Add gnuplot_ipp to the compilation chain and include gnuplot_ipp.h before Histogram2D.h.");
#endif
}

template <class T>
void Histogram2D<T>::smooth_add(T value, double x, double y, double smooth_dx, double smooth_dy)
{ 
  size_t binx,biny, minbinx,minbiny, maxbinx,maxbiny;

  // Wont't work for HistogramBounds::Bounded though!
  get_bin_index(x-smooth_dx, y-smooth_dy, minbinx, minbiny);
  get_bin_index(x+smooth_dx, y+smooth_dy, maxbinx, maxbiny);
  /*
  printf("(%g,%g)\n", x, y);
  printf("c::: %g %g :: %g %g\n", x-smooth_dx, y-smooth_dy, x+smooth_dx, y+smooth_dy);
  printf("c::: %g %g :: %g %g\n", get_bin_center(minbinx,minbiny).x, get_bin_center(minbinx,minbiny).y, get_bin_center(maxbinx,maxbiny).x, get_bin_center(maxbinx,maxbiny).y);
  */

  // The code used is an optimized version of:
  /*
  for (biny = minbiny; biny < maxbiny; ++biny)
    for (binx = minbinx; binx < maxbinx; ++binx)
      (*_m)(binx,biny) += value;
  return;
  */
  // Perform the looping using contigouous memory access and pointer arithmetic to reduce the amount of computational power wasted on the loop itself. Notice Matrix uses a transposed access.
  T *m_raw = _m->raw();
  T *m_row_ptr = m_raw + minbinx*_ybins;
  for (binx = minbinx; binx < maxbinx; ++binx)
    {
      for (biny = minbiny; biny < maxbiny; ++biny)
	m_row_ptr[biny] += value;

      m_row_ptr += _ybins;
    }

  // Add the marginals on the borders if it is a Boundless histogram
  if (_bound_type == HistogramBounds::Boundless)
    {
      // p = plus; m = minus.
      double xp_extra = x+smooth_dx - _xmax;
      double xm_extra = _xmin - (x-smooth_dx);
      double yp_extra = y+smooth_dy - _ymax;
      double ym_extra = _ymin - (y-smooth_dy);

      // Right side
      if (xp_extra > 0)
	{
	  binx = _xbins - 1;

	  T x_weight = xp_extra / _dx; // Proportional to the outside area in bin-counts.

	  if (ym_extra > 0)
	    {
	      T y_weight = ym_extra / _dy;

	      (*_m)(binx,0) += x_weight * y_weight;
	    }

	  for (biny = minbiny; biny < maxbiny; ++biny)
	    (*_m)(binx,biny) += x_weight;

	  if (yp_extra > 0)
	    {
	      T y_weight = yp_extra / _dy;
	      
	      (*_m)(binx,_ybins-1) += x_weight * y_weight;
	    } 
	}
      // Left side
      if (xm_extra > 0)
	{
	  binx = 0;

	  T x_weight = xm_extra / _dx; // Proportional to the outside area in bin-counts.

	  if (ym_extra > 0)
	    {
	      T y_weight = ym_extra / _dy;

	      (*_m)(binx,0) += x_weight * y_weight;
	    }

	  for (biny = minbiny; biny < maxbiny; ++biny)
	    (*_m)(binx,biny) += x_weight;

	  if (yp_extra > 0)
	    {
	      T y_weight = yp_extra / _dy;
	      
	      (*_m)(binx,_ybins-1) += x_weight * y_weight;
	    }
	}
      // Top
      if (yp_extra > 0)
	{
	  T y_weight = yp_extra / _dy;

	  for (binx = minbinx; binx < maxbinx; ++binx)
	    (*_m)(binx,_ybins-1) += y_weight;
	}
      // Bottom
      if (ym_extra > 0)
	{
	  T y_weight = ym_extra / _dy;
	  
	  for (binx = minbinx; binx < maxbinx; ++binx)
	    (*_m)(binx,0) += y_weight;
	}
    } // end of _bound_type == Boundless
}


template <class T>
void Histogram2D<T>::marginal_x(Buffer<T> &marginal)
{
  Assert(marginal.size() == _xbins, "Marginal_x size buffer doesn't match histogram marginal size.");

  for (size_t i=0; i < _xbins; ++i)
    {
      T tmp = 0;
    
      for (size_t j=0; j < _ybins; ++j)
	tmp += bin(i,j);

      marginal[i] = tmp;
    }
}

template <class T>
void Histogram2D<T>::fill_marginal_x(Histogram<T> &marginal)
{
  Assert(marginal.bins() == _xbins, "Marginal_x size buffer doesn't match histogram marginal size.");

  for (size_t i=0; i < _xbins; ++i)
    {
      T tmp = 0;
    
      for (size_t j=0; j < _ybins; ++j)
	tmp += bin(i,j);

      marginal.bin(i) = tmp;
    }
}

template <class T>
void Histogram2D<T>::marginal_y(Buffer<T> &marginal)
{
  Assert(marginal.size() == _ybins, "Marginal_y size buffer doesn't match histogram marginal size.");

  for (size_t j=0; j < _ybins; ++j)
    {
      T tmp = 0;

      for (size_t i=0; i < _xbins; ++i)
	tmp += bin(i,j);

      marginal[j] = tmp;
    }      
}

template <class T>
void Histogram2D<T>::fill_marginal_y(Histogram<T> &marginal)
{
  Assert(marginal.bins() == _ybins, "Marginal_y size buffer doesn't match histogram marginal size.");

  for (size_t j=0; j < _ybins; ++j)
    {
      T tmp = 0;

      for (size_t i=0; i < _xbins; ++i)
	tmp += bin(i,j);

      marginal.bin(j) = tmp;
    }      
}


template <class T>
Matrix<T> Histogram2D<T>::gen_gaussian_kernel(T stddev_x, T stddev_y, bool *success)
{
    // Full Width at Half Maximum: FWHM = 2sqrt(2ln2) stddev ~= 2.35482 stddev
  T FWHM_x = 2.35482 * stddev_x;
  T FWHM_y = 2.35482 * stddev_y;
  // Should be more than enough for precise gaussian blurring (kernel is very close to 0 at such edges)
  T length_x = 3*FWHM_x; 
  T length_y = 3*FWHM_y;

  // Enforce odd-size kernel (perfect centering)
  size_t size_x = length_x / _dx;
  size_t size_y = length_y / _dy;

  if (size_x > 1 || size_y > 1)
    {
      if (size_x > 1)
	size_x -= (size_x % 2 == 0); 
      else
	size_x = 1;
      if (size_y > 1)
	size_y -= (size_y % 2 == 0);
      else
	size_y = 1;

      Matrix<T> kernel(size_x, size_y);

      size_t center_x = size_x/2;
      size_t center_y = size_y/2;
  
      T norm_factor = 1 / (2*M_PI * stddev_x * stddev_y);
      for (size_t i=0; i < center_x; ++i)
	for (size_t j=0; j < center_y; ++j)
	  {
	    T X = (i*_dx) / stddev_x;
	    T Y = (j*_dy) / stddev_y;

	    kernel(center_x-i,center_y-j)   = 
	      kernel(center_x+i,center_y-j) = 
	      kernel(center_x+i,center_y+j) = 
	      kernel(center_x-i,center_y+j) = std::exp(-0.5 * (X*X + Y*Y)) / norm_factor;
	  }
  
      *success = true;
      return kernel;
    }
  else
    {
      *success = false;
      return Matrix<T>(1,1,1);
    }
}


template <class T>
void Histogram2D<T>::kernel_convolution(Matrix<T> &kernel, Matrix<T> &m)
{
  
  Assert(kernel.cols()%2 || kernel.rows()%2, "Kernel is not odd-sized!");
  Assert(m.rows() == _m->rows() && m.cols() && _m->cols(), "Object doesn't have the same number of bins.");
  
  const long int ksize_x = kernel.rows();
  const long int ksize_y = kernel.cols();
  const long int kcenter_x = ksize_x/2;
  const long int kcenter_y = ksize_y/2;

  m.clear();

  for (long int I=0; I < _xbins; ++I)
    for (long int J=0; J < _ybins; ++J)
      {
	for (long int kx = 0; kx < ksize_x; ++kx)
	  {
	    long int i = (I+kcenter_x)-kx;

	    if ( i < 0 || i >= _xbins )
	      continue;

	    for (long int ky = 0; ky < ksize_y; ++ky)
	      {
		long int j = (J+kcenter_y)-ky;

		if ( j < 0 || j >= _ybins )
		  continue;

		m(i,j) += kernel(kx,ky) * (*_m)(I,J);
	      }
	  }
      }

  (*_m) = m; // Copy avoidable by switching pointers with the external object (potentially dangerous).
}


#endif // HISTOGRAM_2D_H__
