#include "StreamSet.h" 

StreamSet::StreamSet(unsigned int streams, size_t data_len, size_t spectrum_magnitude_size, unsigned int blocks)
: _streams(streams), _data(streams, data_len, fftw_malloc, fftw_free), _spectrum(streams, spectrum_magnitude_size), _pos(streams,Point2D<real>()), _last_buf(streams), _last_cluster(streams), _first_active_time_block(streams), _last_active_time_block(streams), _active_blocks(streams), _status(streams, Status::Unitialized), _trajectory(streams, blocks, Point2D<real>()), _latest_id(streams) 
{}

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


unsigned int StreamSet::streams() { return _streams; }


unsigned int StreamSet::acquire_id() { _latest_id = _data.try_acquire_id(); Guarantee(_latest_id, "Impossible to allocate new stream."); return _latest_id; }

void StreamSet::release_id(unsigned int id) { clear(id); _data.release_id(id); }

void StreamSet::release_ids() { _data.release_ids(); }

Buffer<real> *  StreamSet::spectrum(unsigned int id)     { Assert(id, "Id=0"); return _spectrum(id-1); }
Buffer<real> *& StreamSet::last_buf(unsigned int id)     { Assert(id, "Id=0"); return _last_buf[id-1]; }
Buffer<real> *  StreamSet::stream  (unsigned int id)     { return _data.get_buffer(id); }

unsigned int  & StreamSet::first_active_time_block(unsigned int id) { Assert(id, "Id=0"); return _first_active_time_block[id-1]; }
unsigned int  & StreamSet::last_active_time_block (unsigned int id) { Assert(id, "Id=0"); return  _last_active_time_block[id-1]; }
unsigned int  & StreamSet::active_blocks          (unsigned int id) { Assert(id, "Id=0"); return           _active_blocks[id-1]; }
int           & StreamSet::last_cluster           (unsigned int id) { Assert(id, "Id=0"); return            _last_cluster[id-1]; }
Point2D<real> & StreamSet::pos                    (unsigned int id) { Assert(id, "Id=0"); return                     _pos[id-1]; }

Buffer<Point2D<real> > *  StreamSet::trajectory(unsigned int id)     { Assert(id, "Id=0"); return _trajectory(id-1); }


real * StreamSet::last_buf_raw(unsigned int id, size_t pos) { Assert(id, "Id=0"); return &(*_last_buf[id-1])[pos];}

  

unsigned int StreamSet::latest_id() { return _latest_id; }




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

  (*trajectory(id))[block] = cluster_pos;
};


void StreamSet::print(unsigned int id)
{
  printf("Stream id = %u @ (%g,%g) k=%d: First t_block=%u  Last t_block=%u  Active_blocks=%u\n", 
	 id, pos(id).x, pos(id).y, last_cluster(id), first_active_time_block(id), last_active_time_block(id), active_blocks(id));
}



