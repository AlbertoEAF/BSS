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

void build_mono_ibm_masks(Buffer<int> &masks, Buffers<real> &WDOs, Buffers<real> &masked_S, Buffers<real> &xoriginals, size_t tb, size_t t_offset, real *X, int FFT_N, fftw_plan &fft, Buffer<real> &W, real Phi_x)
{
  masks.clear();
  masked_S.clear();

  int N = xoriginals.buffers(); // True Sources.

  static Buffer<real> Sn(FFT_N,0,fftw_malloc,fftw_free), Yn(Sn), Msn(FFT_N/2), Myn(FFT_N/2);
  static Buffer<real> xoriginal(Sn);
  for(int n = 0; n < N; ++n)
    {
      xoriginal.copy(xoriginals.raw(n,t_offset), FFT_N);
      xoriginal *= W; // The original stream wasn't windowed yet.
      fftw_execute_r2r(fft, xoriginal(), Sn());
      // Y = X - S 
      Yn.copy(X,FFT_N); Yn -= Sn;
      
      evenHC2magnitude(Sn, Msn);
      evenHC2magnitude(Yn, Myn);
      /*
      evenHC2power(Sn, Msn);
      evenHC2power(Yn, Myn);
      */
      real &wdo = *WDOs.raw(n,tb);
      real psr=0, sir=0, sir_den=0;
      for (int k = 0,K=FFT_N/2; k < K; ++k)
	{
	  // if Phi is positive assign it to the current source (n)
	  if ( Phi(Msn[k], Myn[k], Phi_x) )
	    {
	      masks[k] = n; 

	      masked_S.raw(n)[k      ] = X[k      ];
	      masked_S.raw(n)[FFT_N-k] = X[FFT_N-k];

	      wdo += Msn[k]*Msn[k] - Myn[k]*Myn[k];	      
	      psr += Msn[k]*Msn[k];
	      sir_den += Myn[k]*Myn[k];
	      /*
	      wdo += Msn[k] - Myn[k];
	      psr += Msn[k];
	      */
	    }
	}
      
      sir = psr;
      /*
      wdo /= Msn.sum();
      psr /= Msn.sum();
      */
      wdo /= Sn.energy();
      psr /= Sn.energy();
      sir /= sir_den;
      

      printf(RED "wdo psr sir \t%g \t%g \t%g   %g\n" NOCOLOR, wdo, psr, sir, psr-psr/sir);
    }
}

#endif // MASKS_H__
