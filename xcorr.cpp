
#include <cmath>

#include <iostream>

#include "libBuffers/Buffer.h"

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
      puts("\nMissing program options:\n \txcorr <input1> <input2> <output> [max_t(ms)] [max_corr_t(ms)]\n");
      return 1;
    }

  SndfileHandle input_wav1(argv[1]);
  SndfileHandle input_wav2(argv[2]);

  uint sample_rate_Hz = input_wav1.samplerate();
  real Tsampling = 1/(real)sample_rate_Hz;

  Guarantee(input_wav1.samplerate() == input_wav2.samplerate(), "Sampling rates must match.");
  Guarantee(wav::ok(input_wav1) && wav::ok(input_wav2), "File doesn't exist.");
  Guarantee(wav::mono(input_wav1) && wav::mono(input_wav2), "Inputs must be mono.");


  Buffer<real> I1(input_wav1.frames()), I2(input_wav2.frames());

  input_wav1.read(I1(), input_wav1.frames());
  input_wav2.read(I2(), input_wav2.frames());

  long int tmax, tcorrmax, N1 = I1.size(), N2 = I2.size(), N = std::min(N1,N2);

  if (argc >=5)
    tmax = strtof(argv[4],NULL) / (1000.0 * Tsampling);
  else
    tmax = N;

  if (argc >=6)
    tcorrmax = strtof(argv[5],NULL) / (1000.0 * Tsampling);
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
	O[t] += I1[t+i] * I2[i];    
    }


  wav::write_mono (argv[3], O(), O.size(), sample_rate_Hz);

  return 0;
}
