#ifndef TYPES_H__
#define TYPES_H__

// Floating datatype
typedef double real;
/*
#ifdef DOUBLE_PRECISION
#define USING_DOUBLE_PRECISION 1
typedef double real;
#else
#define USING_DOUBLE_PRECISION 0
typedef float real;
#endif
*/


// Should support accessing large arrays. Positive values only
typedef long int idx; 

typedef unsigned long int uidx;

typedef unsigned int uint;


/// Auxiliary class for bin indexes: Point2D 
template <class T> class Point2D
{
 public:
 Point2D(T a=0, T b=0) : x(a), y(b) {};

  T x;
  T y;
};


template <class T> std::ostream &operator << (std::ostream &output, Point2D<T> &p)
{
  output << "(" << p.x << "," << p.y << ")";
  return output; // allows chaining
}
/* End of auxiliary class: Point2D */


#endif // TYPES_H__
