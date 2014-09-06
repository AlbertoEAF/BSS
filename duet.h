#include <iostream>
#include <complex>
#include <stdlib.h> // rand, srand
#include <cmath>
#include <stdlib.h> // strtof, strtod
#include <fstream>
#include <limits.h>
#include <float.h> // float and double limits
#include <string.h> // memcpy
#include <algorithm> // sort

#include <fftw3.h>

#include "types.h"

#include "gnuplot_ipp/gnuplot_ipp.h" // Must be included before Histogram2D.h to enable Histogram plotting support.
#include "Histogram2D.h" // Includes "Buffer.h" "Matrix.h" "Histogram.h" "array_ops.h"
#include "libs/config_parser.h"
#include "wav.h"
#include "filters.h"
#include "extra.h"
#include "libs/timer.h"
#include "RankList.h"
#include "color_codes.h"
#include "Buffers.h"
#include "DoubleLinkedList.h"
#include "BufferPool.h"
#include "IdList.h"
#include "libs/String.h" // to build the names of the .dat files for rendering
#include "CyclicCounter.h"


#include "constants.h"

#include "DUETstruct.h"
#include "abs.h"
#include "separation_metrics.h" 
#include "StreamSet.h"
#include "windows.h" 
#include "complex_ops.h"
#include "clustering.h"



using std::cout;
using std::cin;
using std::endl;







void RENDER_HIST(const std::string &filepath, const std::string &title, bool pause)
{
  std::string cmd("gnuplot -e \"splot \\\"");

  cmd += filepath;
  cmd += "\\\" u 1:2:3 w pm3d title \\\"";
  cmd += title;
  cmd += "\\\", \\\"s.dat\\\"  pt 7 ps .9 title \\\"Simulation clusters\\\", \\\"s_duet.dat\\\" pt 8 ps .8 title \\\"DUET clusters\\\"; set xlabel \\\"alpha\\\"; set ylabel \\\"delta (s)\\\";";
  if (pause)
    cmd += "pause -1";
  cmd += "\"";

  system(cmd.c_str()); 
}

template <class T> void print(T o) { cout << o << endl; }

template <class T> 
void swap (T &a, T &b)
{
  T tmp = a;
  a = b;
  b = tmp;
}

template <class T> T div_up(T num, T den) { return num/den + (num%den?1:0); }


/// Returns the success state of the input and prints [DONE] or [FAIL] accordingly.
bool print_status (bool success)
{
  if (success)
    puts(GREEN "[DONE]" NOCOLOR);
  else
    puts(RED "[FAIL]" NOCOLOR);

  return success;
}



real Lambda_distance(Point2D<real> &a, Point2D<real> &b)
{
  return std::abs(a.y-b.y);
}

template <class T> T blocks (T n, T block_size)
{
  return n/block_size + ( n % block_size ? 1:0 );
}
