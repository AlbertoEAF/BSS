#include <iostream>
#include <complex>
#include <stdlib.h> // rand, srand
#include <cmath>
#include <stdlib.h> // strtof, strtod
#include <fstream>
#include <limits.h>
#include <float.h> // float and double limits
#include <string.h> // memcpy

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


using std::cout;
using std::cin;
using std::endl;

#include "constants.h"

#include "DUETstruct.h"
#include "abs.h"
#include "separation_metrics.h" 
#include "StreamSet.h"
#include "windows.h" 
#include "complex_ops.h"
#include "clustering.h"










