// List of integers to track usage of resources by int id's. Meant for small lists (not optimized at all).

#ifndef IDLIST_H__
#define IDLIST_H__

#include <iostream> // Otherwise Buffer won't shut up
#include "Buffer.h"

class IdList
{
 public:
  IdList (unsigned int n);

  unsigned int size();
  unsigned int last();
 
  int operator[] (unsigned int n);

  bool add(int value); // It is made to crash if can't add. No need to check the return value.
  bool del(int value);

  bool has(int value);

  void clear();

  void print();



 private:
  Buffer<int> _list;
  unsigned int _last, _size;
};





#endif // LIST_H__
