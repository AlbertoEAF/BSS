/*
  Only safe for integer variables

  Counts like the range operator in Python but cyclically: (0,3) -> 0,1,2, 0,1,2, ...
*/

#ifndef CyclicCounter_h__
#define CyclicCounter_h__

#include "custom_assert.h"

template <class Int>
class CyclicCounter
{
 public:
  CyclicCounter(Int max);
  CyclicCounter(Int min, Int max);
  
  inline Int value() { return _value; }

  Int operator++();

 private:
  Int _value;
  const Int _min, _max;
};

template <class Int>
CyclicCounter<Int>::CyclicCounter(Int max) : _value(0), _min(0), _max(max)
{
  Assert(max>0, "Invalid max! Must be greater than 0");
}

template <class Int>
CyclicCounter<Int>::CyclicCounter(Int min, Int max) : _value(min), _min(min), _max(max)
{
  Assert(min<max, "Invalid min and max!");
}

template <class Int>
Int CyclicCounter<Int>::operator++()
{
  static Int max = _max-1;

  if (_value < max)
    ++_value;
  else
    _value = _min;

  return _value;
}

#endif // CyclicCounter_h__
