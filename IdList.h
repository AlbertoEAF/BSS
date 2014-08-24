// List of integers to track usage of resources by int id's. Meant for small lists (not optimized at all).
// Assumes every entry is positive. 0 is for empty values. (id's > 0)

#ifndef IDLIST_H__
#define IDLIST_H__

#include <iostream> // Otherwise Buffer won't shut up
#include "Buffer.h"

class IdList
{
 public:
  IdList (unsigned int n);

  inline unsigned int size ();
 
  bool add(int value); // It is made to crash if can't add. No need to check the return value.
  bool del(int value);

  bool has(int value);

  void print();

 private:
  Buffer<int> _list;
  unsigned int _last, _size;
};





#endif // LIST_H__
