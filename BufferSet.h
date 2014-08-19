// Works like a matrix or single-threaded Bufferpool with the ability to swap buffers (allows permutations without copies)

#ifndef BUFFERSET_H__
#define BUFFERSET_H__

#include "Buffer.h"

template <class T>
class BufferSet
{
public:
  BufferSet(unsigned int buffers, size_t buffer_size, void * (*custom_alloc)(size_t) = malloc, void (*custom_free)(void *) = free);
  
  ~BufferSet();

  Buffer<T> * operator() (unsigned int i);
  void swap(unsigned int i, unsigned int j);

private:
  const unsigned int _pool_size;
  Buffer<T> ** _bufs;
};

template <class T> BufferSet<T>::BufferSet (unsigned int buffers, size_t buffer_size, void * (*custom_alloc)(size_t), void (*custom_free)(void *))
  : _pool_size(buffers) 
{
  _bufs = (Buffer<T>**) malloc(sizeof(Buffer<T>*) * _pool_size);

  for (unsigned int i = 0; i < _pool_size; ++i)
    _bufs[i] = new Buffer<T>(buffer_size, 0, custom_alloc, custom_free);
}

template <class T> BufferSet<T>::~BufferSet ()
{
  for (unsigned int i = 0; i < _pool_size; ++i)
    delete _bufs[i];

  free(_bufs);
}


template <class T>
Buffer<T> *BufferSet<T>::operator() (unsigned int i)
{
  return _bufs[i];
}

template <class T>
void BufferSet<T>::swap(unsigned int i, unsigned int j)
{
  Buffer<T> * i_ptr_copy = _bufs[i];
  _bufs[i] = _bufs[j];
  _bufs[j] = i_ptr_copy;
}

#endif //BUFFERSET_H__
