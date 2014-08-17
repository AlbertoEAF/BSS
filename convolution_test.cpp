
#include <fstream>
#include <stdlib.h>

#include "gnuplot_ipp/gnuplot_ipp.h"
#include "wav.h"
#include "types.h"
#include "convolution.h"
#include "filters.h"
#include "extra.h"
#include "libs/config_parser.h"
#include "libs/timer.h"
using std::cout;



void sound2txt(const char *filepath, real *data, uint size, real dt)
{
  std::ofstream file;
  file.open(filepath);

  for (uint sample = 0; sample < size; ++sample)
  {
    file << sample*dt << "\t" << data[sample] << "\n";
  }

  file.close();
}

/** Only works for floats */
template <class T>
void center_data(T *array, uidx size)
{
  T mean = sum(array, size) / (T) size;

  for (uidx i = 0; i < size; ++i)
    array[i] -= mean;
}

using std::endl;


int main()
{
  Gnuplot pg, ph, pgh;
  // Read mono .wav file
  SndfileHandle input_wav("mike.wav");
  
  Options o("settings.cfg", Quit, 1);

  long int g_size = input_wav.frames();
  real *g = new real[g_size];
  input_wav.read(g, g_size);

  uint sample_rate_Hz = input_wav.samplerate();
  real dt = 1.0 / (real) sample_rate_Hz;

  pg.plot_y(g, g_size, "g[]");
  sound2txt("g.txt", g, g_size, dt);
  write_mono_wav ("g.wav", g, g_size, sample_rate_Hz);

  printf("Processing sound file with %u samples @ %uHz (%f s)\n", (uint)g_size, (uint)sample_rate_Hz, g_size*dt);




  long int h_size;  

  real A = o.stof("h_A");
  real tau = o.stof("h_tau"); 
  real max_filter_time = o.stof("h_max_time");

  real *h = decay_filter (A, tau, 0.0, max_filter_time, sample_rate_Hz, &h_size);
  /*
  long int h2_size;
  real *h2 = decay_filter(A, 0.1, 0, 1., sample_rate_Hz, &h2_size);
  add_h2_in_h1 (h, h2, h_size, h2_size, 10000, 2);
  //for (uidx t = 0; t < h_size; ++t) h[t+600] += h[t];
  normalize_filter (h, h_size);
  */
  printf("Filter generated with size %u.", (uint)h_size);
 // sound2txt("h.txt", h, h_size, dt);
  ph.plot_y(h, h_size, "h[]");

  //return 0;

  printf("Computing %u terms.\n", (uint)g_size*(uint)h_size);

  

  long int gh_size = g_size + h_size;  // Auralization task. Ending is not required.
  //long int gh_size = g_size; // in the RT implementation we don't care for what comes next.
  real *gh = new real[gh_size];

  //center_data(g, g_size);
  Timer convolution_timer;
  convolution_timer.start();
  causal_convolution(g, h, gh, g_size, h_size, gh_size);
  convolution_timer.stop();
  printf("Convolution execution time: %lf (s)\n", convolution_timer.getElapsedTimeInSec());
  normalize_to(gh, gh_size, max(gh, gh_size));

  cout << max(g, g_size)<<endl;
  cout << max(gh, gh_size)<<endl;

  //sound2txt("gh.txt", gh, gh_size, dt);
  write_mono_wav ("gh.wav", gh, gh_size, sample_rate_Hz);
  pgh.plot_y(gh, gh_size, "gh[]");

  wait();
  delete[] h;
  delete[] gh;
  return 0;
}
