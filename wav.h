#ifndef WAV_H__
#define WAV_H__

#include <algorithm> // std::max

#include <sndfile.hh>
#include <stdio.h> // check() error printing

#include <string>

#include "array_ops.h" // for normalization (max())
//#include "Buffer.h" // Just to avoid changing the input: copy the data and perform normaliztion on the copy
#include "MatrixDeclaration.h" // Version for matrix of outputs write

#include "Buffers.h"

/** Reads a .wav file from disk. STUB: better use manually */
/*int read_mono_wav (const char *filename)
  {
  SndfileHandle file(filename);

  printf("File %s has %u samples at a sample rate of %u.\n", filename, (uint)file.frames(), (uint)file.samplerate());


  // STUB

  return 0;
  }*/


const double EPSILON = std::numeric_limits<double>::epsilon();






/// Check for the validity of a sound file and print the error report if an error occured.
namespace wav 
{
  int ok(SndfileHandle s)
  {
    if (s.error())
      {
        printf("\nError opening soundfile::%s\n\n", s.strError());
        return 0;
      }
    return 1;
  }
  
  int mono(SndfileHandle s)
  {
    if (s.channels() == 1)
      return 1;
    else
      return 0;
  }


  /** Writes a .wav file to disk. 
    
      If explicit_normalization == 0, the normalization factor is found automatically. Set a specific value in case you want to perform the normalization across several wav files.

  */
  template <class T>
    bool write_mono (const std::string &filename, const T *data, uint size, const int sample_rate_Hz, T explicit_normalization = 0)
    {
      const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_16;
      const int channels=1;


      // Normalization performed on the copy and not on the data itself.
    
      Guarantee(explicit_normalization >= 0, "Normalization must be > 0.");

      T max;
      if (explicit_normalization < EPSILON) // == 0
	max = array_ops::max_abs(data, size);
      else
	max = explicit_normalization;


      Buffer<T> data_copy(data, size); // Could be more efficient by not allocating before testing if normalization is required.
      // If it is so close to 1: no point in normalizing
      if (std::abs(max-1) > EPSILON) 
	data_copy /= max;

      // Write the data to disk

      SndfileHandle outfile(filename, SFM_WRITE, format, channels, sample_rate_Hz);
      if (not outfile) 
        return false;

      outfile.write(data_copy(), size);

      return true;
    }

  // Every row in the matrix counts as a .wav for output.
  template <class T>
    bool write (const std::string &name, const Matrix<T> &data, const int sample_rate_Hz, const bool global_normalization = true)
    {
      const uint N_waves = data.rows();
      const uint samples = data.cols();

      Matrix<T> data_copy(data);

      if (global_normalization)
	data_copy /= array_ops::max_abs(data.raw(), data.size());

      for (uint n=0; n < N_waves; ++n)
	{
	  bool write = 0;
	  if (global_normalization)
	    write = wav::write_mono(name+std::to_string(n)+".wav", data_copy(n), samples, sample_rate_Hz, 1.0);
	  else
	    write = wav::write_mono(name+std::to_string(n)+".wav", data_copy(n), samples, sample_rate_Hz); // Let it normalize

	  if (! write)
	    return false;
	}


      return true;
    }


  template <class T>
    bool write (const std::string &name, const Buffers<T> &data, const int sample_rate_Hz, const bool global_normalization = true)
    {
      const uint N_waves = data.buffers();
      const uint samples = data.buffer_size();

      Buffers<T> data_copy(data);

      if (global_normalization)
	data_copy /= data_copy.max_abs();

      for (uint n=0; n < N_waves; ++n)
	{
	  bool write = 0;
	  if (global_normalization)
	    write = wav::write_mono(name+std::to_string(n)+".wav", data_copy(n)(), samples, sample_rate_Hz, 1.0);
	  else
	    write = wav::write_mono(name+std::to_string(n)+".wav", data_copy(n)(), samples, sample_rate_Hz); // Let it normalize

	  if (! write)
	    return false;
	}

      return true;
    }


} // End of namespace wav

#endif // WAV_H__
