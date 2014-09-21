/**

*****************************
Programmer: Alberto Ferreira
Project start-Date: 22/11/2012

License: GPL2

Disclaimer: You may use the code as long as you give credit to the 
copyright holder, Alberto Ferreira.
*****************************

The matrix is stored unidimensionally with malloc to avoid calling the 
constructors automatically so that the default value initialization is faster.
Changing to new is therefore considered a regression, only placement new would
be accepted.

*/

#ifndef BUFFER_DECLARATION_H__
#define BUFFER_DECLARATION_H__

#ifndef BUFFER_H__
#include <stdio.h>
#include <iostream>
#else
#include <iosfwd> // You must still include iostream afterwards.
#endif

#include <string.h> // memcpy

#include <stdlib.h>

#include "custom_assert.h"


#include "array_ops.h" 

// Templates das funcoes friend
template <class T> class Buffer;
template <class T> std::ostream &operator << (std::ostream &, const Buffer<T> &);
template <class T> std::istream &operator >> (std::istream &, Buffer<T> &);

//template <class T> Buffer<float> abs_matrix (Buffer<T> &);

template <class T>
class Buffer
{
  friend std::ostream &operator << <>(std::ostream &, const Buffer<T> &);
  friend std::istream &operator >> <>(std::istream &, Buffer<T> &);

 public:
  Buffer(const T *data, size_t size, T default_values = 0, void *(*custom_malloc)(size_t) = malloc, void (*custom_free)(void *) = free);
  Buffer (size_t size, T default_values = 0, void *(*custom_malloc)(size_t) = malloc, void (*custom_free)(void *) = free);
  Buffer (const Buffer<T> & copy);
  ~Buffer () { (*_custom_free)(m); }

  T & operator [] (size_t pos);
  T operator [] (size_t pos) const;
  
  inline T * operator () () { return m; }

  inline size_t size() const { return m_size; }
  inline size_t d()    const { return m_size; } // Consistent with Matrix.h

  void copy(const Buffer<T> &copy, size_t copy_size = 0);
  void copy(const T *copy, size_t copy_size);
  void clear() { for(size_t i=0;i<m_size;++i) m[i] = _default_value; }
  void fill(T value) { for(size_t i=0;i<m_size;++i) m[i] = value; }
  void fill_range(T min, T max);

  void add_at(Buffer<T> &, size_t pos);

  Buffer       & operator  = (const Buffer<T> &);
  const Buffer & operator += (const Buffer<T> &);
  const Buffer & operator -= (const Buffer<T> &);
  const Buffer & operator *= (const Buffer<T> &); // Hadamard-product
  const Buffer   operator +  (const Buffer<T> &);
  const Buffer   operator *  (const Buffer<T> &);



  const Buffer & operator *= (const T factor);
  const Buffer & operator /= (const T factor);



  // Prints the first elements.
  void print(size_t n=0);

  void normalize (const T value = 1); 

  // No-parameter array_ops integration.

  size_t min_index () { return array_ops::min_index(m, m_size); }
  size_t max_index () { return array_ops::max_index(m, m_size); }
  T min () { return array_ops::min(m, m_size); }
  T max () { return array_ops::max(m, m_size); }
  T sum () { return array_ops::sum(m, m_size); }
  T avg () { return array_ops::avg(m, m_size); }
  T max_abs() { return array_ops::max_abs(m, m_size); }
  T energy() { return array_ops::energy(m, m_size); }
  T rms() { return array_ops::rms(m, m_size); }


 private:
  // If memory alignment is required
  void *(*_custom_malloc)(size_t) ;
  void  (*_custom_free)  (void *p);

  const size_t m_size;
  T *m;
  T _default_value;
};

 
#endif // BUFFER_DECLARATION_H__
