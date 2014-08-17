
#include <cmath>

#include <iostream>

#include "Buffer.h"

#include "types.h"

#include "wav.h"

//#include "extra.h"

#include <string.h> // memcpy

#include <limits.h>

#include <omp.h>


using std::cout;
using std::endl;

#include <complex>


int main(int argc, char **argv)
{

  if(argc < 3)
    {
      puts("\nMissing program options:\n \txcorr <input> <output> [max_t(ms)] [max_corr_t(ms)]\n");
      return 1;
    }

  SndfileHandle input_wav(argv[1]);

  uint sample_rate_Hz = input_wav.samplerate();
  real Tsampling = 1/(real)sample_rate_Hz;

  Guarantee(wav::ok(input_wav), "File doesn't exist.");
  Guarantee(wav::mono(input_wav), "Input must be mono.");


  Buffer<real> I(input_wav.frames());

  input_wav.read(I(), input_wav.frames());

  long int tmax, tcorrmax, N = I.size();

  if (argc >=4)
    tmax = strtof(argv[3],NULL) / (1000.0 * Tsampling);
  else
    tmax = N;

  if (argc >=5)
    tcorrmax = strtof(argv[4],NULL) / (1000.0 * Tsampling);
  else
    tcorrmax = N;


  if (tmax > N)
    tmax = N;
  if (tcorrmax > N)
    tcorrmax = N;

  Buffer<real> O(tmax);

  for (long int t = 0; t < tmax; ++t)
    {
      for(long int i=0; t+i < N && i < tcorrmax; ++i)
	O[t] += I[t+i] * I[i];    
    }


  write_mono_wav (argv[2], O(), O.size(), sample_rate_Hz);

  return 0;
}
