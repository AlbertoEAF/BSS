#ifndef SEPARATION_METRICS_HPP__
#define SEPARATION_METRICS_HPP__

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

  real eo = std::abs(array_ops::inner_product(e, o, samples));
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
void separation_stats(Buffers<real> &s, Buffers<real> &o, int N, idx samples)
{
  real dtotal;
  static Buffer<real> dtotals(N); // Store the results to find the minimum
  static Buffer<int> matches(N); // Indices of the estimated mixtures that already have the minimum distortion measure (match found)


  matches.clear();
  for (int i = 0; i < N; ++i)
    {
      // Compare the score against all the signals and choose the smallest.
      // Notice that the larger peak in the histogram should get the clearest reseults so we start with it since this is already sorted by score.
      // Notice that two signals can't be compared against the same original.
      for (int i_original = 0; i_original < N; ++i_original)
	{/*
	 // UNCOMMENT THIS PART IF THE GUARANTEE IS REMOVED! (***)
	 if (in(i_original+1, matches)) // 0's can't be used thus +1 is applied
	 dtotals[i_original] = FLT_MAX;
	 else*/
	  // The original signal wasn't matched yet.
	  dtotals[i_original] = Dtotal(s.raw(i), o.raw(i_original), samples);	  
	}

      //cout << "dtotals:" << dtotals;

      

      // Find the matching index and add it to the exlusion list for the next signals.
      size_t match_index = array_ops::min_index(dtotals(), N);

      // IF YOU REMOVE THIS GUARANTEE ADD THE REGION WITH THE COMMENT (***)
      //Guarantee0(in(match_index+1, matches), "Dtotal was minimal for two sources: Probably signals are mixed!");
      if (in(match_index+1, matches))
	printf(RED "Dtotal was minimal for two sources (match=%lu): Probably signals weren't unmixed successfuly!\n" NOCOLOR, match_index);

      matches[i] = match_index+1; // 0's cant be used
      dtotal = dtotals[match_index];

      // Other stats
      real E_r = Energy_ratio(s.raw(i), o.raw(match_index), samples);
      real snr = SNR(s.raw(i), o.raw(match_index), samples);
    
      printf(BLUE "s%d : Dtotal=%g (%g/sample) SNR=%gdB E_r=%g\n" NOCOLOR, i, dtotal, dtotal/(real)samples, snr, E_r);     
    }
}


#endif//SEPARATION_METRICS_HPP__
