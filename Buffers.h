// Simple collection of same-size buffers.
// Works like a matrix or single-threaded Bufferpool with the ability to swap buffers (allows permutations without copies)

#ifndef BUFFERS_H__
#define BUFFERS_H__

#include "Buffer.h"

template <class T>
class Buffers
{
public:
  Buffers(unsigned int buffers, size_t buffer_size, void * (*custom_alloc)(size_t) = malloc, void (*custom_free)(void *) = free);
  Buffers(unsigned int buffers, size_t buffer_size, T init_value, void * (*custom_alloc)(size_t) = malloc, void (*custom_free)(void *) = free);
  Buffers(const Buffers<T> & copy);
  
  ~Buffers();

  unsigned int buffers() { return _pool_size; }
  size_t buffer_size() { return _buffer_size; }
  Buffer<T> * operator() (unsigned int i);
  void swap(unsigned int i, unsigned int j);

  const Buffers & operator*= (const T factor);
  const Buffers & operator/= (const T factor);

  void clear() { for(unsigned int i=0;i<_pool_size;++i) _bufs[i]->clear(); }

  T * raw (unsigned int i) { Assert(i<_pool_size, "Out of bounds %u buffer request for Buffers(%u).",i,_pool_size); return (*_bufs[i])(); }
  T * raw (unsigned int i, size_t index_offset) { Assert(i<_pool_size, "Out of bounds %u buffer request for Buffers(%u).",i,_pool_size); return &raw(i)[index_offset]; }

  T max_abs();

private:
  const unsigned int _pool_size;
  const size_t _buffer_size;
  Buffer<T> ** _bufs;

  void *(*_custom_malloc)(size_t) ;
  void  (*_custom_free)  (void *p);
};


template <class T> Buffers<T>::Buffers (unsigned int buffers, size_t buffer_size, void * (*custom_malloc)(size_t), void (*custom_free)(void *))
: _pool_size(buffers), _buffer_size(buffer_size), _custom_malloc(custom_malloc), _custom_free(custom_free) 
{
  _bufs = (Buffer<T>**) malloc(sizeof(Buffer<T>*) * _pool_size);

  for (unsigned int i = 0; i < _pool_size; ++i)
    _bufs[i] = new Buffer<T>(buffer_size, 0, custom_malloc, custom_free);
}


template <class T> Buffers<T>::Buffers (unsigned int buffers, size_t buffer_size, T init_value, void * (*custom_malloc)(size_t), void (*custom_free)(void *))
: _pool_size(buffers), _buffer_size(buffer_size), _custom_malloc(custom_malloc), _custom_free(custom_free) 
{
  _bufs = (Buffer<T>**) malloc(sizeof(Buffer<T>*) * _pool_size);

  for (unsigned int i = 0; i < _pool_size; ++i)
    _bufs[i] = new Buffer<T>(buffer_size, init_value, custom_malloc, custom_free);
}


template <class T> Buffers<T>::Buffers (const Buffers<T> &copy)
: _pool_size(copy._pool_size), _buffer_size(copy._buffer_size), _custom_malloc(copy._custom_malloc), _custom_free(copy._custom_free) 
{
  _bufs = (Buffer<T>**) malloc(sizeof(Buffer<T>*) * _pool_size);

  for (unsigned int i = 0; i < _pool_size; ++i)
    {
      _bufs[i] = new Buffer<T>(_buffer_size, 0, _custom_malloc, _custom_free);
      
      _bufs[i]->copy( (*copy._bufs[i]) );
    }
}


template <class T> Buffers<T>::~Buffers ()
{
  for (unsigned int i = 0; i < _pool_size; ++i)
    delete _bufs[i];

  free(_bufs);
}


template <class T>
Buffer<T> *Buffers<T>::operator() (unsigned int i)
{
  return _bufs[i];
}

template <class T>
void Buffers<T>::swap(unsigned int i, unsigned int j)
{
  Buffer<T> * i_ptr_copy = _bufs[i];
  _bufs[i] = _bufs[j];
  _bufs[j] = i_ptr_copy;
}

template <class T>
const Buffers<T> & Buffers<T>::operator *= (const T factor)
{
  for (unsigned int i = 0; i < _pool_size ; ++i)
    (*_bufs[i]) *= factor;

  return *this;
}

template <class T>
const Buffers<T> & Buffers<T>::operator /= (const T factor)
{
  for (unsigned int i = 0; i < _pool_size ; ++i)
    (*_bufs[i]) /= factor;

  return *this;
}

template <class T>
T Buffers<T>::max_abs()
{
  T maxabs = 0;
  for (unsigned int i = 0; i < _pool_size ; ++i)
    {
      T maxabs_i = array_ops::max_abs(_bufs[i](), _buffer_size);
      if (maxabs_i > maxabs)
	maxabs = maxabs_i;
    }
  return maxabs;
}


#endif //BUFFERSET_H__
