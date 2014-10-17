#ifndef STREAMSET_H__
#define STREAMSET_H__

#include <iostream>
#include <fftw3.h>

#include "types.h"
#include "Buffers.h" // For Point2D<real> : we still cant handle such overloads

#include "BufferPoolDeclaration.h"

enum class Status : std::int8_t { Unitialized, Active, Inactive, Dead }; // Can go to the trash

class StreamSet // Non-thread-safe.
{
 public:
  StreamSet(unsigned int streams, size_t data_len, size_t spectrum_magnitude_size, unsigned int blocks);

  unsigned int streams();

  void clear (unsigned int id);


  unsigned int acquire_id();
  void release_id(unsigned int id);

  void release_ids();

  Buffer<real> *  spectrum(unsigned int id);
  Buffer<real> *& last_buf(unsigned int id);
  Buffer<real> *  stream  (unsigned int id);

  unsigned int  & first_active_time_block(unsigned int id);
  unsigned int  & last_active_time_block (unsigned int id);
  unsigned int  & active_blocks          (unsigned int id);
  int           & last_cluster           (unsigned int id);
  Point2D<real> & pos                    (unsigned int id);

  Buffer<Point2D<real> > *  trajectory(unsigned int id);

  void add_buffer_at(unsigned int id, int cluster, Buffer<real> &buf, Buffer<real> &magnitude, unsigned int block, unsigned int hop_size, Point2D<real> &cluster_pos);

  real * last_buf_raw(unsigned int id, size_t pos = 0);

  const unsigned int _streams;

  unsigned int latest_id();

  void print(unsigned int id);

  BufferPool<real>       _data;
  Buffers<real>          _spectrum;
  Buffer<Point2D<real> > _pos; // Cluster position (alpha,delta)
  Buffer<Buffer<real>*>  _last_buf; 
  Buffer<int>            _last_cluster; // Cluster index.
  Buffer <unsigned int>  _first_active_time_block, _last_active_time_block, _active_blocks;
  Buffer<Status>         _status;

  Buffers<Point2D<real> > _trajectory;

 private:
  unsigned int _latest_id;
};


#endif //STREAMSET_H__
