#ifndef SAFE_H__
#define SAFE_H__

typedef enum OnFail 
{ 
  Unspecified, // useful for default arguments
  Ignore,
  Warn,
  Quit
} OnFail;


#endif 
