#ifndef TYPES_H__
#define TYPES_H__

// Floating datatype
typedef double real;
/*
#ifdef DOUBLE_PRECISION
#define USING_DOUBLE_PRECISION 1
typedef double real;
#else
#define USING_DOUBLE_PRECISION 0
typedef float real;
#endif
*/


// Should support accessing large arrays. Positive values only
typedef long int idx; 

typedef unsigned long int uidx;

typedef unsigned int uint;


#endif // TYPES_H__
