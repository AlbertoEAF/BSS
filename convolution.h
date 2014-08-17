#ifndef CONVOLUTION_H__
#define CONVOLUTION_H__

#include <cmath>
#include <cstdlib>



/*!
	Performs the convolution operation. Of course, the size of the returned array must be g_size+h_size (to calculate the response of the filter even to the last element only of g).

	@warn: For speed, g_size > h_size or it will access out of bounds

	@param[in] g Sound signal to convolve.
	@param[in] h Convolution filter function.
*/

#include <cstdio>

#include "types.h"

#include <iostream>

using std::cout;

template <class realT, class intT>
void causal_convolution (realT *g, realT *h, realT *gh, intT g_size, intT h_size, uint gh_size)
{
	intT t, v;
	puts("stage 1");
	// Process the first elements ( g(t<0) := 0 )	
	for (t = 0; t < h_size; ++t)
	{
		gh[t] = g[t]*h[0]; // instead of setting to 0 we compute a value
		for (v = 1; v <= t; ++v)
		{
			//cout << t << " " << v << "    " << gh[t] << std::endl;
			gh[t] += h[v]*g[t-v];
		}
	}
	puts("stage 2");
	// Process the remaining datapoints until t = g_size ( g is always accessible )
	for (t = h_size; t < g_size; ++t)
	{
		gh[t] = h[0]*g[t];
		for (v = 1; v < h_size; ++v)
			gh[t] += h[v]*g[t-v];
	}
	puts("stage 3");
	// Process the response for the virtual zeros ( g(t>g_size) := 0 )
	for (t = g_size; t < gh_size; ++t)
	{
		gh[t] = h[t-g_size]*g[g_size-1];
		for (v = t-g_size+1; v < h_size; ++v)
			gh[t] += h[v]*g[t-v];
	}
}



#endif // CONVOLUTION_H__