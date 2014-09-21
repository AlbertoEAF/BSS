// Simple collection of same-size buffers.
// Works like a matrix or single-threaded Bufferpool with the ability to swap buffers (allows permutations without copies)

#ifndef BUFFERS_DECLARATION_H__
#define BUFFERS_DECLARATION_H__

/*
  WE CAN DO THIS TWO WAYS: 
  
  a) RECURSIVELY USING BASE COMPILATIONS IN THE CHAIN: Smaller binaries
  Limitation: Specializations must be manually done for all required items in the chain.

  b) Independent compilation using base headers: Bigger binaries
  Limitation: Much easier to use, if you specify specializations for the current header 
  they will work even if the base dependencies don't have the specializations required,
  they will be added in.
  
  (full discussion in the how-to)

  we're opting for way b for now. At final compilation optimization stages this is 
  easy to change but for project development way b is less error prone.
*/

// We suppose people don't want to mix compiled and non-compiled buffers for ease of use of non-compiled templates so either the whole chain is compiled or non-compiled.
/*
// WAY a
#ifndef BUFFERS_H__
#include "BufferDeclaration.h"
#else
#include "Buffer.h"
#endif
*/

// WAY b
#ifndef BUFFERS_H__
#include <stdio.h>
#include <iostream>
#endif
#include "Buffer.h" // Always include the header because we might want more specializations than those set for the base class.



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


#endif //BUFFERS_DECLARATION_H__
