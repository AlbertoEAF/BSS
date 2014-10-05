#include <atomic>

#include <iostream>
#include "BufferPool.h"
#include <unistd.h>


using std::cout;
using std::endl;

int main()
{
  
  
  BufferPool<int> b(4, 10);

  int id;

  while(1)
    {
      cout << "Id:";
      std::cin >> id;

      if (id < 0)
	break;

      if (! id)
	cout << b.acquire_id() << endl;
      else
	b.release_id(id);
    }

  
  
  
  return 0;
}
