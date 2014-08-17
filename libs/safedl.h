#ifndef SAFE_DL_H__
#define SAFE_DL_H__

#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include "safe.h"

/*
  safe_* : Try and close application in case of fail.

  try_*  : YOU must handle the errors. 
           In case of error NULL is returned. 

           Don't let errors propagate!
          
	   If you proceed and use NULL pointers you will get segmentation faults
	    ... if you're lucky.
*/ 


void *safe_dlopen(const char *file, int mode);
void *safe_dlsym (void *handle, const char *symbol_name);


void *try_dlopen(const char *file, int mode);
void *try_dlsym (void *handle, const char *symbol_name);

#endif
