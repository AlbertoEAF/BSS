#ifndef SAFEMEM_H__
#define SAFEMEM_H__

#include <stdlib.h>
#include <stdio.h>

void * safe_malloc(size_t size)
{
  void *ptr = malloc(size);
	
  if (ptr == NULL)
    {
      puts("CRITICAL ERROR: safe_malloc failed to allocate memory!");
      printf("%lu bytes requested\n", size);
      exit(EXIT_FAILURE);
    }

  return ptr;
}

#endif // SAFEMEM_H__
