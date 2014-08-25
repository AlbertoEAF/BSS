#include <iostream>


#include <fftw3.h>
#include <cmath>

#include <stdlib.h> // strtof, strtod

#include "gnuplot_ipp/gnuplot_ipp.h" // Must be included before Histogram2D.h

//#include "Array.h"
#include "Buffer.h"
#include "Matrix.h"



#include "Histogram.h"
#include "Histogram2D.h"

#include "array_ops.h"

#include <fstream>

#include "types.h"
#include "libs/config_parser.h"
#include "wav.h"
#include "filters.h"
#include "extra.h"

#include <string.h> // memcpy

#include <limits.h>
#include <float.h> // float and double limits

#include "libs/timer.h"

#include "RankList.h"

#include "color_codes.h"

#include "Buffers.h"
 
#include "DoubleLinkedList.h"

#include "BufferPool.h"

#include "IdList.h"

using std::cout;
using std::cin;
using std::endl;


#include "libs/String.h" // to build the names of the .dat files for rendering

#include <complex>

#include <stdlib.h> // rand, srand

#include "CyclicCounter.h"


struct DUETcfg
{
  real noise_threshold;

  // Functions
  bool aggregate_clusters;

  real p;
  real q;

  idx FFT_slide, FFT_slide_percentage;
  idx FFT_N;
  idx FFT_pN;
  int FFT_p; // The FFT oversampling factor.

  int Fmax, Fmin;

  bool use_window; // we'll always use a window from now on
  //  int window_type;


  bool use_smoothing;  
  bool use_smoothing_2D;
  real sigma_alpha;
  real sigma_delta;

  // Peaks
  real max_peak_scale_disparity;
  real min_peak_fall;
  real min_peak_dalpha;
  real min_peak_ddelta;
};

void evenHC2magnitude(int samples, real *hc, real *magnitude);

class StreamSet // Non-thread-safe.
{
 public:
  StreamSet(unsigned int streams, size_t data_len, size_t spectrum_magnitude_size) : _streams(streams), _data(streams, data_len, fftw_malloc, fftw_free), _spectrum(streams, spectrum_magnitude_size), _last_buf(streams) {};

  unsigned int streams() { return _streams; }

  void clear (unsigned int id) { stream(id)->clear(); spectrum(id)->clear(); }


  unsigned int acquire_id() { _latest_id = _data.try_acquire_id(); Guarantee(_latest_id, "Impossible to allocate new stream."); return _latest_id; };
  void release_id(unsigned int id) { clear(id); _data.release_id(id); };

  void release_ids() { _data.release_ids(); }

  Buffer<real> *  spectrum(unsigned int id) { Assert(id, "Id must be larger than 0."); return _spectrum(id-1);      };
  Buffer<real> *& last_buf(unsigned int id) { Assert(id, "Id must be larger than 0."); return _last_buf[id-1];      };
  Buffer<real> *  stream  (unsigned int id) { return _data.get_buffer(id); };

  void stream_id_add_buffer_at(unsigned int id, Buffer<real> &buf, Buffer<real> &magnitude, size_t pos);

  real * last_buf_raw(unsigned int id, size_t pos = 0) { Assert(id, "Id must be larger than 0."); return &(*_last_buf[id-1])[pos];}

  const unsigned int _streams;

  unsigned int latest_id() { return _latest_id; }

  BufferPool<real>      _data;
  Buffers<real>         _spectrum;
  Buffer<Buffer<real>*> _last_buf;

 private:
  unsigned int _latest_id;
};

void StreamSet::stream_id_add_buffer_at(unsigned int id, Buffer<real> &buf, Buffer<real> &magnitude, size_t pos)
{
  stream(id)->add_at(buf, pos);
  last_buf(id) = &buf;
  (*spectrum(id)) += magnitude;
};
