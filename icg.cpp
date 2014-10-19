#include <iostream>

#include "libBuffers/Buffer.h"
#include "wav.h"

#include <cmath>

struct stereo_t
{
  double l;
  double r;
};

struct stereo_frame_t {

  stereo_frame_t() { item_left = 0; item_right = 0; }
  float item_left;
  float item_right;
};

#include <stdlib.h>


int main(int argc, char **argv)
{
  Guarantee(argc>=3, "Usage:\n\t icg <skip_frames> <stereo.wav>");

  size_t skip_frames = atoi(argv[1]);    

  for (int wave_arg=2; wave_arg<argc; ++wave_arg)
    {
      SndfileHandle sndfile(argv[wave_arg]);
      //  const uint sample_rate_Hz = sndfile.samplerate();
      const uint samples = sndfile.frames(); 
      //  Buffer<stereo_frame_t> wav(samples, stereo_frame_t());
      Buffer<double> wav(samples*2);

      sndfile.read(wav(), samples*2); 

      Buffer<double> l(samples), r(samples);


      Guarantee(sndfile.channels() == 2, "Stereo file required");

      const int channels = 2;

      for (size_t i=skip_frames*channels; i < samples; i+=2)
	{
	  l[i] = wav[i];
	  r[i] = wav[i+1];
	}

      //  std::cout << std::sqrt( l.energy() ) / float(samples) << std::endl;
      // std::cout << std::sqrt( r.energy() ) / float(samples) << std::endl;
  
      double al = std::sqrt( l.energy() ) / float(samples-skip_frames);
      double ar = std::sqrt( r.energy() ) / float(samples-skip_frames);

      double g = ar/al;

      /*
	if (g > 1)
	std::cout << "2 " << 1/g << std::endl; // If the signal is stronger on the channel 2 reduce its volume.
	else
	std::cout << "1 " << g   << std::endl; // If the signal is stronger on the channel 1 reduce its volume.
      */

      std::cout << argv[wave_arg] << std::endl;
      std::cout << g << std::endl;
  
    }

  return 0;
}
