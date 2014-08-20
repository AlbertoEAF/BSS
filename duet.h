#include <iostream>


#include <fftw3.h>
#include <cmath>

#include <stdlib.h> // strtof, strtod

#include "gnuplot_ipp/gnuplot_ipp.h" // Must be included before Histogram2D.h

//#include "Array.h"
#include "Buffer.h"
#include "Matrix.h"

#include "Histogram.h"
#include "Histogram2D.h"

#include "array_ops.h"

#include <fstream>

#include "types.h"
#include "libs/config_parser.h"
#include "wav.h"
#include "filters.h"
#include "extra.h"

#include <string.h> // memcpy

#include <limits.h>
#include <float.h> // float and double limits

#include "libs/timer.h"

#include "RankList.h"

#include "color_codes.h"

#include "Buffers.h"
 
#include "DoubleLinkedList.h"


using std::cout;
using std::endl;


#include "libs/String.h" // to build the names of the .dat files for rendering

#include <complex>

#include <stdlib.h> // rand, srand

#include "CyclicCounter.h"


struct DUETcfg
{
  real noise_threshold;

  // Functions
  bool aggregate_clusters;

  real p;
  real q;

  idx FFT_slide, FFT_slide_percentage;
  idx FFT_N;
  idx FFT_pN;
  int FFT_p; // The FFT oversampling factor.

  bool use_window; // we'll always use a window from now on
  //  int window_type;


  bool use_smoothing;  
  bool use_smoothing_2D;
  real sigma_alpha;
  real sigma_delta;

  // Peaks
  real max_peak_scale_disparity;
  real min_peak_fall;
  real min_peak_dalpha;
  real min_peak_ddelta;
};
