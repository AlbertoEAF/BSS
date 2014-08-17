
#include <cmath>

#include <iostream>

#include "Buffer.h"

#include "types.h"

#include "wav.h"

//#include "extra.h"

#include <string.h> // memcpy

#include <limits.h>




using std::cout;
using std::endl;

#include <complex>


int main(int argc, char **argv)
{

  if(argc != 3)
    {
      puts("\nMissing program options:\n \treverse <input> <output>\n");
      return 1;
    }

  SndfileHandle input_wav(argv[1]);

  uint sample_rate_Hz = input_wav.samplerate();

  Guarantee(wav::ok(input_wav), "File doesn't exist.");

  Guarantee(wav::mono(input_wav), "Input must be mono.");


  Buffer<real> I(input_wav.frames()), O(I);

  input_wav.read(I(), input_wav.frames());

  size_t N = O.size()-1;
  for(size_t i=0; i < I.size(); ++i)
    O[N-i] = I[i];

  wav::write_mono (argv[2], O(), O.size(), sample_rate_Hz);

  return 0;
}
