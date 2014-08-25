#include "IdList.h"

IdList::IdList(unsigned int n) 
  : _list(n), _last(0), _size(n)
{
}

unsigned int IdList::size ()
{
  return _size;
}

unsigned int IdList::last ()
{
  return _last;
}

int IdList::operator[] (unsigned int n)
{
  Guarantee (n < _last, "Accessed out of the allowed range. Don't go over last().");

  return _list[n];
}


bool IdList::add(int value)
{
  Guarantee(_last < _size, "Can't add items anymore to IdList(%u).", _size);

  _list[_last] = value;
  ++_last;

  return true;
}

bool IdList::del(int value)
{
  for (unsigned int i = 0; i < _last; ++i)
    {
      if (_list[i] == value)
	{
	  for (unsigned int j = i; j+1 < _last; ++j)
	    _list[j] = _list[j+1];

	  --_last;

	  return true;
	}
    }

  return false;
}



bool IdList::has(int value)
{
  for (unsigned int i=0; i < _last; ++i)
    if (_list[i] == value)
      return true;
    
  return false;
}

void IdList::clear()
{
  _list.clear();
  _last = 0;
}


void IdList::print()
{
  for (unsigned int i=0; i < _last; ++i)
    std::cout << _list[i] << " ";
  std::cout << std::endl;
}


