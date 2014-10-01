#ifndef SEPARATION_METRICS_HPP__
#define SEPARATION_METRICS_HPP__

#include <fstream>
#include "types.h"
#include "BufferDeclaration.h" // Already includes <cmath>
#include "IdList.h"

bool in (const int val, Buffer<int> &list)
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
real Dtotal(const real *e,  const real *o, const idx samples)
{
  real O2 = array_ops::energy(o, samples);
  real E2 = array_ops::energy(e, samples);

  real eo = array_ops::inner_product(e, o, samples);
  real eo2 = eo*eo;

  real D = (E2*O2 - eo2) / eo2; // Takes into account the modulus of both signals

  return D;
}


real SNR(const real *e, const real *o, const idx samples)
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
real Energy_ratio(const real *e, const real *o, const idx samples)
{
  real E_e = array_ops::energy(e, samples);
  real E_o = array_ops::energy(o, samples);

  return E_e/E_o;
}



/** Provides the separation stats of the mixtures.   
 */
int separation_stats(Buffers<real> &e, Buffers<real> &o, const int Ne, const idx samples)
{
  if (Ne == 0)
    return -1;

  const int N = o.buffers();

  std::ofstream log ("ecoduet.log", std::ios::app/*trunc*/);
  log << "N = " << N << " Ne = " << Ne << std::endl;


  real dtotal;
  Buffer<real> dtotals(std::max<unsigned int>(Ne,N), FLT_MAX); // Store the temporary Dtotal results to find the minimum (best candidate)
  IdList o2e(N), e2o(Ne); // Indices of the estimated mixtures that already have the minimum distortion measure (match found) (maps)

  // Look now for the pair from each true source to the closest of the estimates.
  log << "e_match Dtotal Dtotal/sample SNR\n";
  dtotals.clear();
  int unmixed_results = 0;
  for (int i_o = 0; i_o < N; ++i_o)
    {
      for (int i_e = 0; i_e < Ne; ++i_e)      
	dtotals[i_e] = Dtotal(e.raw(i_e), o.raw(i_o), samples);	  
	
      int e_match = dtotals.min_index();

      if (o2e.has(e_match))
	++unmixed_results;

      o2e.add(e_match); 
      dtotal = dtotals[e_match];

      // Other stats
      real E_r = Energy_ratio(e.raw(e_match), o.raw(i_o), samples);
      real snr = SNR(e.raw(e_match), o.raw(i_o), samples);
    
      printf(BLUE "o%d<-e%d : Dtotal=%g (%g/sample) SNR=%gdB E_r=%g\n" NOCOLOR, i_o, e_match, dtotal, dtotal/(real)samples, snr, E_r);     

      log << e_match+1 << " " << dtotal << " " << dtotal/(real)samples << " " << snr << std::endl;
    }
  if (unmixed_results) 
    printf(RED "%d Unmixed outputs.\n" NOCOLOR, unmixed_results);

  log << "#degeneracies=" << unmixed_results << std::endl;

  // Now look the other way around, from each estimate to the closest true source to it.
  log << "o_match Dtotal Dtotal/sample SNR\n";
  dtotals.clear();
  unmixed_results = 0;
  for (int i_e = 0; i_e < Ne; ++i_e)
    {
      for (int i_o = 0; i_o < N; ++i_o)
	dtotals[i_o] = Dtotal(e.raw(i_e), o.raw(i_o), samples);	  

      int o_match = dtotals.min_index();

      if (e2o.has(o_match))
	++unmixed_results;
	
      e2o.add(o_match);
      dtotal = dtotals[o_match];

      // Other stats
      real E_r = Energy_ratio(e.raw(i_e), o.raw(o_match), samples);
      real snr = SNR(e.raw(i_e), o.raw(o_match), samples);
      printf(CYAN "e%d<-o%d : Dtotal=%g (%g/sample) SNR=%gdB E_r=%g\n" NOCOLOR, i_e, o_match, dtotal, dtotal/(real)samples, snr, E_r);     

      log << o_match+1 << " " << dtotal << " " << dtotal/(real)samples << " " << snr << std::endl;      
    }
  if (unmixed_results) 
    printf(RED "%d Unmixed outputs.\n" NOCOLOR, unmixed_results);

  log << "#degeneracies=" << unmixed_results << std::endl;

  log.close();
  return unmixed_results;
}


#endif//SEPARATION_METRICS_HPP__
