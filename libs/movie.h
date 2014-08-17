#ifndef MOVIE_H__
#define MOVIE_H__

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include "Matrix.h"
#include "Complex.h"
#include "gnuplot_call.h"

#include "String.h"

/* 

// Now included in String.h

// creates a string with N digits
std::string itosNdigits (int number, int N_digits);

// converte int para string
std::string to_string(int i);

*/

template <class T>
int print_to_gnuplot(Matrix<T> m, std::string filepath);
template <class T>
int render_frame(Matrix<T> m, const std::string &cfg_filepath, unsigned int frame_number, unsigned int max_digits);
template <class T>
int render_movie(unsigned int max_digits, int ffmpeg_deprecated_mode = 0);




#endif
