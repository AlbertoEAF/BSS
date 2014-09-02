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

const real _2Pi = 2*M_PI;
#include "windows.h"
#include "separation_metrics.hpp"

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

  real max_Lambda_distance;
  

  // Stream behaviour
  int max_silence_blocks;
  int min_active_blocks;
  real a0min;
  int max_clusters;
  int max_active_streams;
};

void evenHC2magnitude(int samples, real *hc, real *magnitude);

enum class Status : std::int8_t { Unitialized, Active, Inactive, Dead };

class StreamSet // Non-thread-safe.
{
 public:
 StreamSet(unsigned int streams, size_t data_len, size_t spectrum_magnitude_size) : _streams(streams), _data(streams, data_len, fftw_malloc, fftw_free), _spectrum(streams, spectrum_magnitude_size), _pos(streams,Point2D<real>()), _last_buf(streams), _last_cluster(streams), _first_active_time_block(streams), _last_active_time_block(streams), _active_blocks(streams), _status(streams, Status::Unitialized), _latest_id(streams) {};

  unsigned int streams() { return _streams; }

  void clear (unsigned int id);


  unsigned int acquire_id() { _latest_id = _data.try_acquire_id(); Guarantee(_latest_id, "Impossible to allocate new stream."); return _latest_id; }
  void release_id(unsigned int id) { clear(id); _data.release_id(id); }

  void release_ids() { _data.release_ids(); }

  inline Buffer<real> *  spectrum(unsigned int id)     { Assert(id, "Id=0"); return _spectrum(id-1); }
  inline Buffer<real> *& last_buf(unsigned int id)     { Assert(id, "Id=0"); return _last_buf[id-1]; }
  inline Buffer<real> *  stream  (unsigned int id)     { return _data.get_buffer(id); }

  inline unsigned int  & first_active_time_block(unsigned int id) { Assert(id, "Id=0"); return _first_active_time_block[id-1]; }
  inline unsigned int  & last_active_time_block (unsigned int id) { Assert(id, "Id=0"); return  _last_active_time_block[id-1]; }
  inline unsigned int  & active_blocks          (unsigned int id) { Assert(id, "Id=0"); return           _active_blocks[id-1]; }
  inline int           & last_cluster           (unsigned int id) { Assert(id, "Id=0"); return            _last_cluster[id-1]; }
  inline Point2D<real> & pos                    (unsigned int id) { Assert(id, "Id=0"); return                     _pos[id-1]; }


  void add_buffer_at(unsigned int id, int cluster, Buffer<real> &buf, Buffer<real> &magnitude, unsigned int block, unsigned int block_size, Point2D<real> &cluster_pos);

  inline real * last_buf_raw(unsigned int id, size_t pos = 0) { Assert(id, "Id=0"); return &(*_last_buf[id-1])[pos];}

  const unsigned int _streams;

  unsigned int latest_id() { return _latest_id; }

  void print(unsigned int id);

  BufferPool<real>       _data;
  Buffers<real>          _spectrum;
  Buffer<Point2D<real> > _pos; // Cluster position (alpha,delta)
  Buffer<Buffer<real>*>  _last_buf; 
  Buffer<int>            _last_cluster; // Cluster index.
  Buffer <unsigned int>  _first_active_time_block, _last_active_time_block, _active_blocks;
  Buffer<Status>         _status;

 private:
  unsigned int _latest_id;
};

void StreamSet::clear(unsigned int id)
{
  stream(id)->clear(); 
  spectrum(id)->clear(); 
  pos(id) = Point2D<real>();
  last_buf(id) = NULL;
  last_cluster(id) = 0;
  first_active_time_block(id) = 0; 
  last_active_time_block(id) = 0;
  active_blocks(id) = 0;
}

void StreamSet::add_buffer_at(unsigned int id, int cluster, Buffer<real> &buf, Buffer<real> &magnitude, unsigned int block, unsigned int block_size, Point2D<real> &cluster_pos)
{
  stream(id)->add_at(buf, block*block_size);
  last_buf(id) = &buf;
  last_cluster(id) = cluster;
  (*spectrum(id)) += magnitude;

  if (! active_blocks(id))
    first_active_time_block(id) = block;
  last_active_time_block(id) = block;
  active_blocks(id) += 1;

  pos(id) = cluster_pos;
};


void StreamSet::print(unsigned int id)
{
  printf("Stream id = %u @ (%g,%g) k=%d: First t_block=%u  Last t_block=%u  Active_blocks=%u\n", 
	 id, pos(id).x, pos(id).y, last_cluster(id), first_active_time_block(id), last_active_time_block(id), active_blocks(id));
}
