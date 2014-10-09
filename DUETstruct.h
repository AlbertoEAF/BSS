#ifndef DUETSTRUCT_HPP__
#define DUETSTRUCT_HPP__

#include "types.h"

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

  int Fmax, Fmin; // For speech usage this is more than enough (short int goes up to 32767)

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

  real max_Lambda_distance;
  

  // Stream behaviour
  int max_silence_blocks;
  int min_active_blocks;
  real a0min;
  int max_clusters;
  int max_active_streams;
  bool multiple_assign;
};

#endif // DUETstruct
