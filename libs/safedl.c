#include "safedl.h"

void * _dlopen (const char *file, int mode, OnFail onfail);
void * _dlsym (void * handle, const char * symbol_name, OnFail onfail);


void *safe_dlopen(const char *file, int mode)
{
  return _dlopen(file, mode, Quit);
}

void *safe_dlsym(void *handle, const char *symbol_name)
{
  return _dlsym(handle, symbol_name, Quit);
}


void *try_dlopen(const char *file, int mode)
{
  return _dlopen(file, mode, Warn);
}

void *try_dlsym(void *handle, const char *symbol_name)
{
  return _dlsym(handle, symbol_name, Warn);
}







/// helper functions (generic)


void * _dlopen (const char *file, int mode, OnFail onfail)
{
  dlerror(); // flush errors

  void * handle = dlopen (file, mode);

  if (!handle)
    {
      fprintf(stderr,
	      "\nsafe_dlopen():: \n\t %s\n", dlerror());

      if (onfail == Quit)
	  exit(1);

      return NULL;
    }

  return handle;
}

// WARNING: dlsym uses *restrict instead of *. See if it doesn't bring trouble!
void * _dlsym (void * handle, const char * symbol_name, 
		   OnFail onfail)
{
  dlerror(); // flush errors

  void *function = dlsym(handle, symbol_name);

  const char *dlsym_error = dlerror();

  if (dlsym_error)
    {
      fprintf(stderr, 
	      "\nsafe_dlsym():: \n\tCannot open symbol 'f': %s\n", dlsym_error);
      dlclose(handle);

      if (onfail == Quit)
	exit(1);

      return NULL;
    }

  return function;
}
