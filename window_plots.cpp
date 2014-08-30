#include "duet.h"

int main()
{
  Options _o("presets.cfg", Quit, 1); // Choose the preset file.
  Options o (_o("preset").c_str(), Quit, 0); // True configuration file (preset)

  const size_t FFT_N = o.i("FFT_N");

  Buffer<real> W(FFT_N);

  // Windows plot
  static Gnuplot Wplot;

  build_window(W,Hann);
  Wplot.plot(W(),W.size(),"Hann");

  build_window(W,myHamming);
  Wplot.plot(W(),W.size(),"myHamming");

  build_window(W,Hamming);
  Wplot.plot(W(),W.size(),"Hamming");

  build_window(W,weakHamming);
  Wplot.plot(W(),W.size(),"weakHamming");

  build_window(W,weakerHamming);
  Wplot.plot(W(),W.size(),"weakerHamming");

  build_window(W,Hamming0);
  Wplot.plot(W(),W.size(),"Hamming0");

  build_window(W,Blackman);
  Wplot.plot(W(),W.size(),"Blackman");

  build_window(W,BlackmanNutall);
  Wplot.plot(W(),W.size(),"BlackmanNutall");

  build_window(W,Nutall);
  Wplot.plot(W(),W.size(),"Nutall");

  build_window(W,Rectangular);
  Wplot.plot(W(),W.size(),"Rectangular");

  wait();

  return 0;
}
