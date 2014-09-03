#include "duet.h"

/// Calculates the energy profile of overlapping windows. Make sure E.size()>=2*W.size()
void calcE(Buffer<real> &W, int slide, Buffer<real> &E)
{
  Assert(slide, "Slide>0");
  E.clear();

  int N = W.size();
  for (int i=0; i < N; ++i)
    {
      for (int offset = -N; offset < N; offset+=slide)
	{
	  if (i-offset >=0 && i-offset < N)
	    E[i] += W[i-offset];
	}
    }
}

/// Calculates the energy fluctuation ratio R. R.size()==W.size().
void calcR(Buffer<real> &W, Buffer<real> &R)
{
  int N = W.size();
  // Buffer to store the energy of the overlapping windows
  Buffer<real> E(N); // To avoid caring with bounds.
 
  R[0] = 1;
  for (int slide = 1; slide < N; ++slide)
    {
      calcE(W, slide, E);
      R[slide] = E.min()/E.max();
    }
}

int main(int argc, char **argv)
{
  Options _o("presets.cfg", Quit, 1); // Choose the preset file.
  Options o (_o("preset").c_str(), Quit, 0); // True configuration file (preset)
  
  Guarantee(argc == 2, "Usage: window_plots <FFT_N>");
  
  
  size_t FFT_N = stoi(argv[1]);
  printf("FFT_N = %lu\n",FFT_N);

  Buffer<real> W(FFT_N);
  Buffer<real> R(FFT_N);
  Buffer<real> E(FFT_N); // Enough to avoid out-of-bound ops.

  // Windows plot
  static Gnuplot Wplot; // Window plot
  static Gnuplot Rplot; // Ratio=Emin/Emax plot


  Wplot.set_labels("i", "W[i]");
  Rplot.set_labels("FFT_slide", "R (Emin/Emax)");
  

  build_window(W,Hann);
  Wplot.plot(W(),W.size(),"Hann");
  calcR(W,R);
  Rplot.plot(R(),R.size(),"Hann");

  build_window(W,myHamming);
  Wplot.plot(W(),W.size(),"myHamming");
  calcR(W,R);
  Rplot.plot(R(),R.size(),"myHamming");


  build_window(W,Hamming);
  Wplot.plot(W(),W.size(),"Hamming");
  calcR(W,R);
  Rplot.plot(R(),R.size(),"Hamming");

  build_window(W,weakHamming);
  Wplot.plot(W(),W.size(),"weakHamming");
  calcR(W,R);
  Rplot.plot(R(),R.size(),"weakHamming");

  build_window(W,weakerHamming);
  Wplot.plot(W(),W.size(),"weakerHamming");
  calcR(W,R);
  Rplot.plot(R(),R.size(),"weakerHamming");

  build_window(W,Hamming0);
  Wplot.plot(W(),W.size(),"Hamming0");
  calcR(W,R);
  Rplot.plot(R(),R.size(),"Hamming0");

  build_window(W,Blackman);
  Wplot.plot(W(),W.size(),"Blackman");
  calcR(W,R);
  Rplot.plot(R(),R.size(),"Blackman");

  build_window(W,BlackmanNutall);
  Wplot.plot(W(),W.size(),"BlackmanNutall");
  calcR(W,R);
  Rplot.plot(R(),R.size(),"BlackmanNutall");

  build_window(W,Nutall);
  Wplot.plot(W(),W.size(),"Nutall");
  calcR(W,R);
  Rplot.plot(R(),R.size(),"Nutall");

  build_window(W,Rectangular);
  Wplot.plot(W(),W.size(),"Rectangular");
  calcR(W,R);
  Rplot.plot(R(),R.size(),"Rectangular");

  wait();

  return 0;
}
