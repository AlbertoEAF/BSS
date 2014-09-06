// http://www.phyast.pitt.edu/~zov1/gnuplot/html/histogram.html // gnuplot cool drawings

#ifndef HISTOGRAM_H__
#define HISTOGRAM_H__

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
  // Fixed number of bins (no small change is done)
  Histogram(size_t bins, HistogramBounds::Type bounds_type);
  Histogram(const Histogram<T> &copy);
  ~Histogram() { delete _m; }

  void stretch(double min, double max);

  inline T & bin(size_t ibin); // Access to bin directly by bin coordinates (faster than Histogram::(x,y))
  inline T & guarantee_bin (size_t ibinx); // Makes sure a bin is found (runtime assert)
  inline T & operator() (double x);          // Access to bin by (x,y) coordinates

  // (dx,dy) might undergo small changes so that the Histogram bins fit the area perfectly
  //  inline void reshape (double bin_dx, double bin_dy, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type); 
  inline void reshape (size_t bins, double min, double max, HistogramBounds::Type bounds_type);
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

  void plot  (Gnuplot &p, const char * title);
  void replot(Gnuplot &p, const char * title);

  void smooth_add(T value, double x, double smooth_dx);

  // If success==0 smoothing_size < bin: no point in smoothing.
  Buffer<T> gen_gaussian_kernel(T stddev, bool *success = NULL); 

  void kernel_convolution(Buffer<T> &kernel, Buffer<T> &m);

  T min_value() { return _m->min(); }
  T max_value() { return _m->max(); }

 private:
  Buffer<T> *_m;
  double _dx, _min, _max;
  size_t _bins;
  HistogramBounds::Type _bound_type;
  T _dummy; // To return a bin reference with value=0 when there's out of bounds access in DiscardBeyondBound allocation mode
};

template <class T>
Histogram<T>::Histogram(double bin_dx, double x_min, double x_max, HistogramBounds::Type bounds_type)
: _m(NULL)
{
  // Adjust so that they match
  size_t bins_x = (x_max-x_min)/bin_dx;
  
  reshape(bins_x, x_min, x_max, bounds_type);
}



// Needs to have a  different signature from the other constructor
template <class T> Histogram<T>::Histogram(size_t bins, HistogramBounds::Type bounds_type)
: _m(NULL)
{
  _m = new Buffer<T>(bins);
  _bins = bins;
  _min = 0;
  _max = 1;
  _dx = 1/(double)bins;
}


template <class T>
Histogram<T>::Histogram(const Histogram &cpy)
: _m(NULL), _dx(cpy._dx), _min(cpy._min), _max(cpy._max), _bins(cpy._bins), _bound_type(cpy._bound_type)
{
  _m = new Buffer<T>(*cpy._m);
}

/*
template <class T>
void Histogram<T>::reshape(double bin_dx, double bin_dy, double x_min, double x_max, double y_min, double y_max, HistogramBounds::Type bounds_type)
{
  // Histogram bin count must match the histogram area with the bins size, thus a first calculation is done, then the bin sizes are adjusted so that they fit
  size_t bins_x = (x_max-x_min)/bin_dx;
  size_t bins_y = (y_max-y_min)/bin_dy;

  reshape(bins_x, bins_y, x_min, x_max, y_min, y_max, bounds_type);
}
*/

template <class T>
void Histogram<T>::reshape(size_t bins, double min, double max, HistogramBounds::Type bounds_type)
{
  Assert (max > min, "Histogram limits are reversed!");

  _min = min;
  _max = max;
  _bins = bins;


  _dx = (_max-_min) / (T) _bins;

  _bound_type = bounds_type;

  // Reallocate if already exists
  if (_m)
    delete _m;
  _m = new Buffer<T>(bins);
}

/// Stretches the histogram area from min to max, resizing the bins. Clears the histogram.
template <class T> void Histogram<T>::stretch(double min, double max)
{
  Assert (max > min, "Histogram limits are reversed!");

  _min = min;
  _max = max;

  _dx = (_max-_min) / (T) _bins;

  _m->clear();
}


/// Get read-write access to a bin directly by its coordinates
template <class T>
T & Histogram<T>::guarantee_bin(size_t ibin)
{
  Guarantee (ibin < _bins, "Out of bounds bin access (%lu) for Histogram(%lu)!", ibin, _bins);

  return (*_m)(ibin);
}
/// Faster version of guaranteebin() without out of bounds runtime check.
template <class T>
inline T & Histogram<T>::bin(size_t ibin)
{
  Assert (ibin < _bins, "Out of bounds bin access (%lu) for Histogram(%lu)!", ibin, _bins);
  return (*_m)[ibin];
}



template <class T>
bool Histogram<T>::get_bin_index(double x, size_t &ibin)
{
  Assert ( ! std::isnan(x), "NaN coordinate given to Histogram(%lu)!", _bins);

  // add to the boundary bins if the coordinate goes beyond
      if (x < _min)
	ibin = 0;
      else if (x > _max)
	ibin = _bins-1;
      else
	ibin = (x - _min)/_dx;

  if (_bound_type == HistogramBounds::Boundless) 
    return true;
  else
    {
      if (x < _min || x > _max)
	{
	  if (_bound_type == HistogramBounds::DiscardBeyondBound)
	    return false;
	  else
	    Guarantee(0, "Out of bounds access (x,y)=(%f) on Bounded Histogram(%lu)!",x,_bins);
	}
      else
	return true;
    }
}

template <class T>
double Histogram<T>::get_bin_center(size_t bin)
{
  Assert(bin < _bins, "Requested bin(%lu) larger than Histogram(%lu) space.", bin, _bins);
  return _min + _dx*(bin+0.5);
}

template <class T>
T & Histogram<T>::operator()(double x)
{
  size_t bin_index;
	
  Assert ( ! std::isnan(x), "NaN coordinate given to Histogram(%lu)!", _bins);

  // add to the boundary bins if the coordinate goes beyond
  if (_bound_type == HistogramBounds::Boundless) 
    {
      if      (x < _min)
	bin_index = 0;
      else if (x >= _max)
	bin_index = _bins-1;
      else
	bin_index = (x - _min)/_dx;
    }
  else
    {
      if (x < _min || x >= _max)
	{
	  // Guarantee during runtime that the histogram is not Bounded!
	  Guarantee (_bound_type == HistogramBounds::DiscardBeyondBound,
		     "Out of bounds access (x=%g) on Bounded Histogram(%lu)!", x,_bins);
	  // Return a "virtual bin" with value 0 so that changing it does not change the histogram itself and if it is read yields no counts at all (the value can be changed by the user though). This value will be reset at the next call in the same out of bounds situation on a DiscardBeyondBound Histogram.
	  _dummy = 0;
	  return _dummy;

	}
      else
	{
	  bin_index = (x - _min)/_dx;
	}
    }
  return bin(bin_index);
}


template <class T>
void Histogram<T>::operator += (Histogram<T> &other)
{
  static const double eps = 0.000001;
  Assert((&other == this) | (other._bins==_bins && std::abs(other._max-_max)<eps && std::abs(other._min-_min)<eps), "Histograms coordinates don't match.");

  for (size_t i = 0; i < _bins; ++i)
    bin(i) += other.bin(i);
}

template <class T>
void Histogram<T>::operator -= (Histogram<T> &other)
{
  static const double eps = 0.000001;
  Assert((&other == this) | (other._bins==_bins && std::abs(other._max-_max)<eps && std::abs(other._min-_min)<eps), "Histograms coordinates don't match.");

  for (size_t i = 0; i < _bins; ++i)
      bin(i) -= other.bin(i);
}

template <class T>
void Histogram<T>::operator *= (Histogram<T> &other)
{
  static const double eps = 0.000001;
  Assert((&other == this) | (other._bins==_bins && std::abs(other._max-_max)<eps && std::abs(other._min-_min)<eps), "Histograms coordinates don't match.");

  for (size_t i = 0; i < _bins; ++i)
    bin(i) *= other.bin(i);
}


template <class T>
Histogram<T> & Histogram<T>::operator = (Histogram<T> &other)
{
  if (this == &other)
    return *this;

  static const double eps = 0.000001;
  Assert((&other == this) | (other._bins==_bins && std::abs(other._max-_max)<eps && std::abs(other._min-_min)<eps), "Histograms coordinates don't match.");

  *_m = *(other._m); // Data copy

  return *this;
}
  
template <class T>
std::ostream &operator << (std::ostream &output, Histogram<T> &hist)
{
  output << (*hist._m);
  
  return output; // allows chaining
}

template <class T>
std::istream &operator >> (std::istream &input, Histogram<T> &hist)
{
  input >> (*hist._m);

  return input; // allows chaining
}

template <class T>
void Histogram<T>::plot(Gnuplot &p, const char * title)
{
  /* This function cannot blow up the compilation as it is optional 
     so it will be guaranteed at runtime without performance penalties. 
     It will require including gnuplot_ipp.h before Histogram.h though. */

#ifdef GNUPLOT_IPP_H__
  p.plot((*_m)(),_bins, title);
#else
  Guarantee(0, "Add gnuplot_ipp to the compilation chain and include gnuplot_ipp.h before Histogram.h.");
#endif
}

template <class T>
void Histogram<T>::replot(Gnuplot &p, const char * title)
{
  /* This function cannot blow up the compilation as it is optional 
     so it will be guaranteed at runtime without performance penalties. 
     It will require including gnuplot_ipp.h before Histogram.h though. */

#ifdef GNUPLOT_IPP_H__
  p.replot((*_m)(),_bins, title);
#else
  Guarantee(0, "Add gnuplot_ipp to the compilation chain and include gnuplot_ipp.h before Histogram.h.");
#endif
}


template <class T>
void Histogram<T>::smooth_add(T value, double x, double smooth_dx)
{ 
  size_t binx, minbinx, maxbinx;
  
  get_bin_index(x-smooth_dx, minbinx);
  get_bin_index(x+smooth_dx, maxbinx);
    
  // Add the values normally inside the smooth region and inside the Histogram
  for (binx = minbinx; binx < maxbinx; ++binx)
    (*_m)[binx] += value;
  
  // Add the marginals beyond the borders if it is a Boundless histogram
  if (_bound_type == HistogramBounds::Boundless)
    {
  // p = plus; m = minus.
  double xp_extra = x+smooth_dx - _max;
  double xm_extra = _min - (x-smooth_dx);

      // Right side
      if (xp_extra > 0)
	(*_m)[binx] += xp_extra / _dx; // Proportional to the outside area in bin-counts.
      // Left side
      if (xm_extra > 0)
	(*_m)[0   ] += xm_extra / _dx;

    } // end of _bound_type == Boundless
}

/** Creates an odd-sized (centered) buffer kernel that spans 3 times the FWHM (enough precision).
    @warn Not realtime-safe! (allocates memory) */
template <class T>
Buffer<T> Histogram<T>::gen_gaussian_kernel(T stddev, bool *success)
{
  // Full Width at Half Maximum: FWHM = 2sqrt(2ln2) stddev ~= 2.35482 stddev
  T FWHM = 2.35482 * stddev;
  // Should be more than enough for precise gaussian blurring (kernel is very close to 0 at such edges)
  T length = 3*FWHM; 
  // Enforce odd-size kernel
  size_t size = length / _dx;

  if (size > 1)
    {
      size -= (size % 2 == 0); 

      Buffer<T> kernel(size);

      size_t center = size/2;
  
      T norm_factor = 1 / (stddev * std::sqrt(2*M_PI));
      for (size_t i=0; i < center; ++i)
	{
	  T arg = (i*_dx) / stddev;
	  kernel[center-i] = kernel[center+i] = std::exp(-0.5 * arg*arg) / norm_factor;
	}

      *success = true;
      return kernel;
    }
  else
    {
      *success = false;
      return Buffer<T>(1,1); // kernel
    }
}

/** Requires a Buffer of the same size as the internal storage buffer _m to perform the computations.
    This was chosen to allow sharing of a single extra storage layer instead of having one of those per histogram. 

OPTIMIZATION!
For now it calculates the convolution to the m layer and then copies it back into the histogram internal data however it could calculate to the m layer and then switch the pointers of both entities. */
template <class T>
void Histogram<T>::kernel_convolution(Buffer<T> &kernel, Buffer<T> &m)
{
  Assert(kernel.size()%2, "Kernel is not odd-sized!");
  Assert(m.size() == _bins, "Object doesn't have the same number of bins.");

  long int kcenter = kernel.size()/2;
  long int ksize = kernel.size();

  m.clear();

  for (long int I=0; I < _bins; ++I)
    {
      for (long int k = 0; k < ksize; ++k)
	{
	  long int i = (I+kcenter)-k;

	  if (i < 0 || i >= _bins)
	    continue;

	  m[i] += kernel[k] * (*_m)[I];
	}
    }

  (*_m) = m; // Copy avoidable by switching pointers with the external object (potentially dangerous).
}


#endif // HISTOGRAM_H__
