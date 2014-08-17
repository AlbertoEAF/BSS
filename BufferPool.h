#ifndef BUFFERPOOL_H__
#define BUFFERPOOL_H__

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
  
private:
  const unsigned int _pool_size;
  Buffer<T> ** _bufs;
  //  Buffer<std::atomic_bool> _used;
  Buffer<bool> _used;
  std::mutex _mutex;
  unsigned int _last_acquired_id;
};

template <class T> BufferPool<T>::BufferPool (unsigned int buffers, size_t buffer_size, void * (*custom_alloc)(size_t), void (*custom_free)(void *))
  : _pool_size(buffers), _used(buffers, false), _last_acquired_id(0)//, _used(buffers, std::atomic_bool(false)) 
{
  _bufs = (Buffer<T>**) malloc(sizeof(Buffer<T>*) * _pool_size);

  for (unsigned int i = 0; i < _pool_size; ++i)
    _bufs[i] = new Buffer<T>(buffer_size, 0, custom_alloc, custom_free);
}

template <class T> BufferPool<T>::~BufferPool ()
{
  for (unsigned int i = 0; i < _pool_size; ++i)
    {
      delete _bufs[i];
      Guarantee0(_used[i], "BufferPool destroyed with resources still in use.");
    }

  free(_bufs);
}


/// Can deadlock if no users free the resources (not enough buffers for concurrent usage)
template <class T> unsigned int BufferPool<T>::acquire_id ()
{

  _mutex.lock();
  unsigned int i = _last_acquired_id;

  while(_used[i])
    i = (i < _pool_size-1 ? i+1 : 0);

  _used[i] = true;
  _last_acquired_id = i;
  
  _mutex.unlock();

  return i+1; // valid id's start at 1
}


template <class T> unsigned int BufferPool<T>::try_acquire_id ()
{
  _mutex.lock();
  unsigned int i = _last_acquired_id;
  bool acquired = 0;

  int whole_turns_count = 0;

  while(whole_turns_count < 2)
    {
      if (!_used[i])
	{
	  acquired = 1;
	  break;
	}
      else
	{
	  ++i;
	  if (i == _pool_size-1)
	    {
	      i = 0;
	      ++whole_turns_count; // cheaper upper bound on turn count: avoids making a second test against the initial _last_acquired_id and uses 2 turns counted against the pool end as a comparison
	    }
	}
    }

  if (acquired)
    {
      _used[i] = true;
      _last_acquired_id = i;
    }

  _mutex.unlock();

  if (acquired)
    return i+1; 
  else 
    return 0; // couldn't acquire (more than 1 turn)
}

template <class T> Buffer<T> * BufferPool<T>::get_buffer (unsigned int id)
{
  Assert(id, "Invalid id=0");
  return _bufs[id-1];
}


template <class T> void BufferPool<T>::release_id (unsigned int id)
{
  _mutex.lock();
  _used[id-1] = false;
  _mutex.unlock();
}

#endif //BUFFERPOOL_H__
