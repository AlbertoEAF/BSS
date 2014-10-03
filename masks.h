#ifndef MASKS_H__
#define MASKS_H__

// Indicator function for IBM mask. Ms is the magnitude of the desired source at a frequency, My, the same in respect to the mixture minus the source and x the threshold of Phi.
bool Phi(real Ms_f, real My_f, real x)
{
  if (20.0*std::log10(Ms_f/My_f) >= x)
    return true;
  return false;
}

// Same but receives the power instead of the magnitude (square of magnitude)
bool Phi_power(real Ps_f, real Py_f, real x)
{
  if (10.0*std::log10(Ps_f/Py_f) >= x)
    return true;
  return false;
}


/*		  
void build_mono_ibm_masks(Buffer<int> &masks, Buffer<real> &WDO, Buffers<real> &xoriginals, size_t t_offset, real *X, int FFT_N, fftw_plan &fft, Buffer<real> &W)
{
  masks.clear();

  int N = xoriginals.buffers(); // True Sources.

  static Buffer<real> Sn(FFT_N,0,fftw_malloc,fftw_free), Yn(Sn), Msn(FFT_N/2), Myn(FFT_N/2);

  WDO.clear();

  for(int n = 0; n < N; ++n)
    {
      fftw_execute_r2r(fft, xoriginals.raw(n,t_offset), Sn());
      // Y = X*W - S // X hasn't been windowed. Do so!
      Yn.copy(X,FFT_N);
      Yn *= W;
      Yn -= Sn;
      evenHC2magnitude(Sn, Msn);
      evenHC2magnitude(Yn, Myn);

      for (int k = 0,K=FFT_N/2; k < K; ++k)
	{
	  // if Phi is positive assign it to the current source (n)
	  if ( Phi(Msn[k], Myn[k], 0) )
	    {
	      masks[k] = n; 
	      WDO[n] += Msn[k] - Myn[k];
	    }
	}
      WDO[n] /= Msn.sum();
    }
}
*/

void build_mono_ibm_masks(Buffer<int> &masks, Buffers<real> &WDOs, Buffers<real> &Es, Buffers<real> &masked_S, Buffers<real> &xoriginals, size_t tb, size_t t_offset, real *X, int FFT_N, fftw_plan &fft, Buffer<real> &W, real Phi_x, real wdo_threshold)
{
  masks.clear();
  masked_S.clear();

  int N = xoriginals.buffers(); // True Sources.

  static Buffer<real> Sn(FFT_N,0,fftw_malloc,fftw_free), Yn(Sn), Msn(FFT_N/2), Myn(FFT_N/2);
  static Buffer<real> xoriginal(Sn);
  for(int n = 0; n < N; ++n)
    {
      //Guarantee(t_offset+FFT_N < xoriginals(0)->size(), "Larger by %lu", t_offset+FFT_N - xoriginals(0)->size());
      xoriginal.copy(xoriginals.raw(n,t_offset), FFT_N);
      xoriginal *= W; // The original stream wasn't windowed yet.
      fftw_execute_r2r(fft, xoriginal(), Sn());
      // Y = X - S 
      Yn.copy(X,FFT_N); Yn -= Sn;
      
      evenHC2magnitude(Sn, Msn);
      evenHC2magnitude(Yn, Myn);

      real &wdo = *WDOs.raw(n,tb);
      real psr=0, sir=0, sir_den=0;
      if ( Phi(Msn[0], Myn[0], Phi_x) )
	{
	  masks[0] = n;
	  
	  masked_S.raw(n)[0] = X[0];

	  psr     += Msn[0]*Msn[0];
	  sir_den += Myn[0]*Myn[0];
	}
      for (int k = 1,K=FFT_N/2; k < K; ++k)
	{
	  // if Phi is positive assign it to the current source (n)
	  if ( Phi(Msn[k], Myn[k], Phi_x) )
	    {
	      masks[k] = n; 

	      masked_S.raw(n)[k      ] = X[k      ];
	      masked_S.raw(n)[FFT_N-k] = X[FFT_N-k];
	    
	      psr     += Msn[k]*Msn[k];
	      sir_den += Myn[k]*Myn[k];
	    }
	}
      
      sir = psr;
      
      real En = *Es.raw(n,tb) = Sn.energy();

      psr /= En;
      sir /= sir_den;
      
      wdo = psr - psr/sir;
      
      // If wdo is invalid set it to a negative value that can quickly be discarded.
      if ( wdo<wdo_threshold || std::isnan(wdo) )
	wdo=-1;
    }
}

#endif // MASKS_H__
