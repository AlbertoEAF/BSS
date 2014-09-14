#ifndef SEPARATION_METRICS_HPP__
#define SEPARATION_METRICS_HPP__

#include "types.h"
#include "Buffer.h" // Already includes <cmath>
#include "IdList.h"

bool in (int val, Buffer<int> &list)
{
  size_t size = list.size();

  for (size_t i=0; i < size; ++i)
    if (val == list[i])
      return true;

  return false;
}

/** Dtotal metric to compare the original signal with the extracted one.

    @param[in] e - Estimated signal
    @param[in] o - Original  signal
    @param[in] samples - Number of samples

*/
real Dtotal(const real *e,  const real *o, idx samples)
{
  real O2 = array_ops::energy(o, samples);
  real E2 = array_ops::energy(e, samples);

  real eo = array_ops::inner_product(e, o, samples);
  real eo2 = eo*eo;

  real D = (E2*O2 - eo2) / eo2; // Takes into account the modulus of both signals

  return D;
}


real SNR(const real *e, const real *o, idx samples)
{
  Buffer<real> e_normalized(e,samples), o_normalized(o,samples);

  // This normalization guarantees that the energies of each signal is 1.
  e_normalized /= std::sqrt(array_ops::energy(e_normalized(), samples));
  o_normalized /= std::sqrt(array_ops::energy(o_normalized(), samples));
  
  // 2 stands for square(d)
  Buffer<real> x2(samples), x_diff2(samples);

  for (size_t i=0; i < samples; ++i)
    {
      x2[i] = o_normalized[i]*o_normalized[i];
      
      x_diff2[i] = e_normalized[i]-o_normalized[i]; 
      x_diff2[i] *= x_diff2[i];
    }

  real sum_x2 = array_ops::sum(x2(),samples);
  real sum_x_diff2 = array_ops::sum(x_diff2(), samples);

  return 10 * std::log10(sum_x2/sum_x_diff2);
}


/** Relative Energy

    @param[in] e - Estimated signal
    @param[in] o - Original  signal
    @param[in] samples - Number of samples

*/
real Energy_ratio(const real *e, const real *o, idx samples)
{
  real E_e = array_ops::energy(e, samples);
  real E_o = array_ops::energy(o, samples);

  return E_e/E_o;
}



/** Provides the separation stats of the mixtures.   
 */
int separation_stats(Buffers<real> &s, Buffers<real> &o, int N, idx samples)
{
  if (N == 0)
    return -1;

  real dtotal;
  Buffer<real> dtotals(std::max<unsigned int>(N,o.buffers()), FLT_MAX); // Store the temporary Dtotal results to find the minimum (best candidate)
  IdList o2s(o.buffers()), s2o(N); // Indices of the estimated mixtures that already have the minimum distortion measure (match found) (maps)

  // Look now for the pair from each true source to the closest of the estimates.
  dtotals.clear();
  int unmixed_results = 0;
  for (int i_o = 0; i_o < o.buffers(); ++i_o)
    {
      for (int i_s = 0; i_s < N; ++i_s)      
	dtotals[i_s] = Dtotal(s.raw(i_s), o.raw(i_o), samples);	  
	
      int s_match = dtotals.min_index();

      if (o2s.has(s_match))
	++unmixed_results;

      o2s.add(s_match); 
      dtotal = dtotals[s_match];

      // Other stats
      real E_r = Energy_ratio(s.raw(s_match), o.raw(i_o), samples);
      real snr = SNR(s.raw(s_match), o.raw(i_o), samples);
    
      printf(BLUE "o%d->s%d : Dtotal=%g (%g/sample) SNR=%gdB E_r=%g\n" NOCOLOR, i_o, s_match, dtotal, dtotal/(real)samples, snr, E_r);     
    }
  if (unmixed_results) 
    printf(RED "%d Unmixed outputs.\n" NOCOLOR, unmixed_results);

  // Now look the other way around, from each estimate to the closest true source to it.
  dtotals.clear();
  unmixed_results = 0;
  for (int i_s = 0; i_s < N; ++i_s)
    {
      for (int i_o = 0; i_o < o.buffers(); ++i_o)
	dtotals[i_o] = Dtotal(s.raw(i_s), o.raw(i_o), samples);	  

      int o_match = dtotals.min_index();

      if (s2o.has(o_match))
	++unmixed_results;
	
      s2o.add(o_match);
      dtotal = dtotals[o_match];

      // Other stats
      real E_r = Energy_ratio(s.raw(i_s), o.raw(o_match), samples);
      real snr = SNR(s.raw(i_s), o.raw(o_match), samples);
      printf(CYAN "s%d->o%d : Dtotal=%g (%g/sample) SNR=%gdB E_r=%g\n" NOCOLOR, i_s, o_match, dtotal, dtotal/(real)samples, snr, E_r);     
    }
  if (unmixed_results) 
    printf(RED "%d Unmixed outputs.\n" NOCOLOR, unmixed_results);

  return unmixed_results;
}


#endif//SEPARATION_METRICS_HPP__
