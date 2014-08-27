#include "IdList.h"

IdList::IdList(unsigned int n) 
  : _list(n), _N(0), _size(n)
{
}

unsigned int IdList::size ()
{
  return _size;
}

unsigned int IdList::N ()
{
  return _N;
}

int IdList::operator[] (unsigned int n)
{
  Guarantee (n < _N, "Out of bounds access IdList(%u)::operator[%u]. Don't go over IdList::N().", _size, n);

  return _list[n];
}


bool IdList::add(int value)
{
  Guarantee(_N < _size, "Can't add items anymore to IdList(%u).", _size);

  _list[_N] = value;
  ++_N;

  return true;
}

bool IdList::del(int value)
{
  for (unsigned int i = 0; i < _N; ++i)
    {
      if (_list[i] == value)
	{
	  for (unsigned int j = i; j+1 < _N; ++j)
	    _list[j] = _list[j+1];

	  --_N;

	  return true;
	}
    }

  return false;
}



bool IdList::has(int value)
{
  for (unsigned int i=0; i < _N; ++i)
    if (_list[i] == value)
      return true;
    
  return false;
}

void IdList::clear()
{
  _list.clear();
  _N = 0;
}


void IdList::print()
{
  for (unsigned int i=0; i < _N; ++i)
    std::cout << _list[i] << " ";
  std::cout << std::endl;
}


