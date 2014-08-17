#ifndef FILTERS_H__
#define FILTERS_H__

#include <cmath>

#include "types.h"

//////////////////////////////////////////////////////////////////////
// Manipulate filters | Filter Operations
//////////////////////////////////////////////////////////////////////

/** Adds h2 to h1 with delay but doesn't increase h1 size. */
void add_h2_in_h1 (real *h1, real *h2, uint h1_size, uint h2_size, uint h2_delay_frames, real scale_h2)
{
	for (uint t = 0; t < h2_size && t+h2_delay_frames < h1_size; ++t)
		h1[t+h2_delay_frames] += scale_h2*h2[t];
}

template <class T>
T sum (T *array, uidx size)
{
	T total = 0;

	for (uidx i = 0; i < size; ++i)
		total += array[i];

	return total;
}

template <class T>
T max (T *array, uidx size)
{
	T max = 0;

	for (uidx i = 0; i < size; ++i)
		if (array[i]>max)
			max = array[i];

	return max;
}

template <class T>
void normalize_to (T *array, uidx size, T norm)
{
	for (uidx t = 0; t < size; ++t)
		array[t] /= norm;
}

template <class T>
void normalize_filter (T *array, uidx size)
{
	T norm = sum(array, size);

	for (uidx i = 0; i < size; ++i)
		array[i] /= norm;
}


//////////////////////////////////////////////////////////////////////
// Generate filters
//////////////////////////////////////////////////////////////////////

/** 
	Calculates a FIR with A*exp(-lambda*t') shape with t' = silence_time + t

	@warn Requires explicit delete[] 
*/
real * decay_filter (real A, real tau, real silence_time, real time, uint sample_rate_Hz, long int *gh_size)
{
	real lambda = 1.0/tau;
	real dt = 1.0 / (real) sample_rate_Hz;
	uint silence_frames = (uint) (silence_time / dt);

	uint filter_frames = (uint) (time / dt);

	uint frames = silence_frames + filter_frames;

	real *h = new real[frames];

	uint t;

	for (t = 0; t < silence_frames; ++t)
		h[t] = 0.0;

	for (t = silence_frames; t < frames; ++t)
		h[t] = A*exp(-lambda*(t*dt));

	*gh_size = frames;
	return h;
}



#endif // FILTERS_H__