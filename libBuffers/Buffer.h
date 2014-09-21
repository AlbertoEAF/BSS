/**

*****************************
Programmer: Alberto Ferreira
Project start-Date: 22/11/2012

License: GPL2

Disclaimer: You may use the code as long as you give credit to the 
copyright holder, Alberto Ferreira.
*****************************

The matrix is stored unidimensionally with malloc to avoid calling the 
constructors automatically so that the default value initialization is faster.
Changing to new is therefore considered a regression, only placement new would
be accepted.

*/

#ifndef BUFFER_H__
#define BUFFER_H__


#include "BufferDeclaration.h"

#include "Buffer.cpp"

#endif // BUFFER_H__
