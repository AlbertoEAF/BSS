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

#ifndef BUFFER_H__
#define BUFFER_H__


#include <iosfwd> // You must still include iostream afterwards.
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

  inline T & operator [] (size_t pos);
  inline T operator [] (size_t pos) const;
  
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

       
       
template <class T>
Buffer<T>::Buffer (size_t size, T default_value, void *(*custom_malloc)(size_t), void (*custom_free)(void *)) 
: m_size(size), m(NULL), _default_value(default_value)
{
  Guarantee(size, "Requested Buffer with size 0.");

  _custom_malloc = custom_malloc;
  _custom_free   = custom_free  ;

  //m = (T*)safe_malloc(size*sizeof(T)); 
  m = (T*) (*_custom_malloc)(size*sizeof(T));
  Guarantee(m, "Couldn't allocate memory with %lu entries!", size); // Better to check explicitly in case the user provides a custom allocator

  for (size_t i = 0; i < size; ++i)
    m[i] = default_value;
}

template <class T>
Buffer<T>::Buffer (const T *data, size_t size, T default_value, void *(*custom_malloc)(size_t), void (*custom_free)(void *)) 
: m_size(size), m(NULL), _default_value(default_value)
{
  Guarantee(size, "Requested Buffer with size 0.");

  _custom_malloc = custom_malloc;
  _custom_free   = custom_free  ;

  //m = (T*)safe_malloc(size*sizeof(T)); 
  m = (T*) (*_custom_malloc)(size*sizeof(T));
  Guarantee(m, "Couldn't allocate memory!"); // Better to check explicitly in case the user provides a custom allocator

  for (size_t i = 0; i < size; ++i)
    m[i] = data[i];
}


template <class T> 
Buffer<T>::Buffer (const Buffer<T> & copy) // USES MALLOC TO AVOID CALLING THE CONSTRUCTORS RIGHT AWAY
: m_size(copy.m_size), m(NULL), _default_value(copy._default_value)
{  
  _custom_malloc = copy._custom_malloc;
  _custom_free   = copy._custom_free  ;

  //m = (T*) safe_malloc(m_size*sizeof(T));
  m = (T*) (*_custom_malloc)(m_size*sizeof(T));
  Guarantee(m, "Couldn't allocate memory!"); // Better to check explicitly in case the user provides a custom allocator

  memcpy((void*)m, (void*)copy.m, m_size*sizeof(T));
}

template <class T> 
T & Buffer<T>::operator [] (size_t pos)
{
  Assert (pos < m_size, "Out of bounds access (%lu) for Buffer(%lu) ! *this=%p",pos, m_size, this);
  
  return m[pos];
}

template <class T> 
T Buffer<T>::operator []  (size_t pos) const
{
  Assert (pos < m_size, "Out of bounds access (%lu) for Buffer(%lu) ! *this=%p",pos, m_size, this);
  
  return m[pos];
}

template <class T>
std::ostream &operator << (std::ostream &output, const Buffer<T> &buffer)
{
  for (size_t i = 0; i < buffer.m_size ; ++i)
    output << buffer.m[i] << "  ";  
  output << std::endl;

  return output; // allows chaining
}


template <class T>
std::istream &operator >> (std::istream &input, Buffer<T> &buffer)
{
  for (size_t i = 0; i < buffer.m_size ; ++i)
    input >> buffer.m[i]; 

  return input; // allows chaining
}


template <class T> 
void Buffer<T>::print(size_t n)
{
  size_t I = ( n > 0 && n < m_size ? n : m_size ); 

  for (size_t i = 0; i < I ; ++i)
    std::cout << m[i] << "  ";  
  std::cout << std::endl;
}

template <class T> 
void Buffer<T>::copy(const Buffer<T> & copy, size_t copy_size)
{
  if (!copy_size)
    copy_size = copy.m_size;

  Assert(copy_size <= copy.m_size, "You are trying to copy more entries (%lu) than the ones the source Buffer(%lu) has.", copy_size, copy.m_size);
  Assert(copy_size <= m_size     , "You are trying to copy more entries (%lu) than the Buffer(%lu) size allows.", copy_size, m_size);

  memcpy((void*)m, (void*)copy.m, copy_size*sizeof(T));
  for (size_t i = copy_size; i < m_size; ++i)
    m[i] = 0;
}

template <class T> 
void Buffer<T>::copy(const T *copy, size_t copy_size)
{
  Assert(copy_size <= m_size, "You are trying to copy more entries (%lu) than the Buffer(%lu) size.", copy_size, m_size);
  Assert(copy_size, "You requested a copy of size zero to Buffer(%lu).", m_size);

  memcpy((void*)m, (void*)copy, copy_size*sizeof(T));
  for (size_t i = copy_size; i < m_size; ++i)
    m[i] = 0;
}

template <class T> 
void Buffer<T>::add_at(Buffer<T> &buf, size_t pos)
{
  size_t I = buf.size();
    
  Assert(pos+I <= m_size, "Buffer(%lu).add_at() added Buffer(%lu) at %lu resulting in %lu out of bounds accesses.", m_size, buf.m_size, pos, buf.size()+pos-m_size);


  for (size_t i=0; i<I; ++i)
    m[pos+i] += buf[i];
}

 
template <class T>
Buffer<T> & Buffer<T>::operator = (const Buffer<T> &o)
{
  // Safety checks
  if (this == &o)
    return *this;

  Guarantee (m_size == o.m_size , "Invalid assignment operation. Buffer sizes do not match!");

  memcpy((void*)m, (void*)o.m, sizeof(T)*m_size);

  return *this;
}
 

template <class T>
const Buffer<T> & Buffer<T>::operator += (const Buffer<T> &a)
{
  Assert (m_size == a.m_size, "Buffer sizes don not match!");
  for (size_t i = 0; i < m_size ; i++)
    m[i] += a.m[i];

  return *this;
}

template <class T>
const Buffer<T> & Buffer<T>::operator -= (const Buffer<T> &a)
{
  Assert (m_size == a.m_size, "Buffer sizes don not match!");
  for (size_t i = 0; i < m_size ; i++)
    m[i] -= a.m[i];

  return *this;
}

template <class T>
const Buffer<T> & Buffer<T>::operator *= (const T factor)
{
  for (size_t i = 0; i < m_size ; i++)
    m[i] *= factor;

  return *this;
}


template <class T>
const Buffer<T> & Buffer<T>::operator /= (const T factor)
{
  const T inv_factor = 1/factor;

  for (size_t i=0; i < m_size; ++i)
    m[i] *= inv_factor; // Multiplication is faster than division
  
  return *this;
}


template <class T>
const Buffer<T> & Buffer<T>::operator *= (const Buffer<T> &other)
{
  Assert(m_size == other.m_size, "Sizes must match.");

  for (size_t i = 0; i < m_size ; i++)
    m[i] *= other[i];

  return *this;
}


template <class T>
void Buffer<T>::normalize (const T value)
{
  const T factor = value/array_ops::max_abs(m,m_size);

  for (size_t i = 0; i < m_size ; i++)
    m[i] *= factor;
}

template <class T>
void Buffer<T>::fill_range(T min, T max)
{
  T dx = (max-min)/(T)m_size;

  for (size_t i=0; i < m_size; ++i)
    m[i] = min + i*dx;
}

#endif // BUFFER_H__
