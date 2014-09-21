#ifndef ARRAY_OPS_H__
#define ARRAY_OPS_H__

#include <iosfwd>
#include <cmath> // abs

namespace array_ops {

  template <class T, class T_int>
  T sum (const T *array, T_int size)
  {
    T s = 0;
    for (T_int i = 0; i < size; ++i)
      s += array[i];
	
    return s;
  }

  /// Careful with int arrays! (result = floor(true avg))
  template <class T, class T_int>
  T avg (const T *array, T_int size)
  {
    return sum(array, size) / (T)size;
  }

  template <class T, class T_int>
  void print(const T *array, T_int size)
  {
    for (T_int i=0; i < size; ++i)
      std::cout << array[i] << " ";
	
    std::cout << std::endl;
  }

  template <class T, class T_int> 
  T max_abs(const T *array, T_int size)
  {
    T max_abs_value = std::abs(array[0]);
    for (T_int i = 1; i < size; ++i)
      {
	if (std::abs(array[i]) > max_abs_value)
	  max_abs_value = std::abs(array[i]);
      }
    return max_abs_value;
  }	

  template <class T, class T_int>
  T max(const T *array, T_int size)
  {
    T max_value = array[0];
    for (T_int i = 1; i < size; ++i)
      {
	if (array[i] > max_value)
	  max_value = array[i];
      }
    return max_value;
  }	

  template <class T, class T_int>
  T min(const T *array, T_int size)
  {
    T min_value = array[0];
    for (T_int i = 1; i < size; ++i)
      {
	if (array[i] < min_value)
	  min_value = array[i];
      }
    return min_value;
  }	

  template <class T, class T_int>
  T_int max_index(const T *array, T_int size)
  {
    T_int max_index = 0, i = 0; 
    T max_value = array[0];
    for (i = 1; i < size; ++i)
      {
	if (array[i] > max_value)
	  {
	    max_value = array[i];
	    max_index = i;
	  }
      }
    return max_index;
  }	


  template <class T, class T_int>
  T_int min_index(const T *array, T_int size)
  {
    T_int min_index = 0, i = 0; 
    T min_value = array[0];
    for (i = 1; i < size; ++i)
      {
	if (array[i] < min_value)
	  {
	    min_value = array[i];
	    min_index = i;
	  }
      }
    return min_index;
  }	



  /**
     Calculates the inner product of two real arrays f and g.
  */
  template <class T, class T_int>
  T inner_product(const T *f, const T *g, T_int size)
  {
    T v = 0;
  
    for (T_int i = 0; i < size; ++i)
      v += f[i]*g[i];

    return v;
  }

  template <class T, class T_int>
  T energy(const T *f, T_int size)
  {
    T v = 0;
  
    for (T_int i = 0; i < size; ++i)
      v += f[i]*f[i];

    return v;
  }


  template <class T, class T_int>
  T a0(const T *f, const T *g, T_int size)
  {
    T inner = array_ops::inner_product(f,g,size);
    T E1 = array_ops::energy(f,size);
    T E2 = array_ops::energy(g,size);

    return inner / (std::sqrt(E1)*std::sqrt(E2));
  }
  

  template <class T, class T_int>
  void normalize(T *f, T value, T_int size)
  {
    const T factor = value/array_ops::max_abs(f,size);

    for (T_int i=0; i < size; ++i)
      f[i] *= factor;
  }

  template <class T, class T_int>
  T rms(const T *f, T_int size)
  {
    T v = 0;
  
    for (T_int i = 0; i < size; ++i)
      v += f[i]*f[i];

    v = sqrt(1/(T)size * v);

    return v;
  }

  template <class T, class T_int>
    void fill_range(T *array, T_int size, T min, T max)
  {
    T dx = (max-min)/(T)size;
  
    for (T_int i = 0; i < size; ++i)
      array[i] = min + i*dx;
  }


} // Namespace array_ops

#endif // ARRAY_OPS_H__
