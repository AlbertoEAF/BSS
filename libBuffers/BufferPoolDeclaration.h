#ifndef BUFFERPOOL_DECLARATION_H__
#define BUFFERPOOL_DECLARATION_H__

#ifndef BUFFERPOOL_H__
#include <stdio.h>
#include <iostream>
#endif
#include "Buffer.h"

#include "ifDebug.h"

//#include <atomic>

#include <mutex>

template <class T>
class BufferPool
{
public:
  BufferPool(unsigned int buffers, size_t buffer_size, void * (*custom_alloc)(size_t) = malloc, void (*custom_free)(void *) = free);
  
  ~BufferPool();

  unsigned int try_acquire_id(); 
  // Can deadlock if pool is not enough and the user's algorithm isn't lock-free and requires #buffers > _pool_size.
  unsigned int acquire_id();
  inline Buffer<T> * get_buffer (unsigned int id);
  void release_id(unsigned int id);
  
  void release_ids();

private:
  const unsigned int _pool_size;
  Buffer<T> ** _bufs;
  //  Buffer<std::atomic_bool> _used;
  Buffer<bool> _used;
  std::mutex _mutex;
  unsigned int _last_acquired_id;
};

#endif //BUFFERPOOL_DECLARATION_H__
