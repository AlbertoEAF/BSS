#include "duet.h"


int merged_streams = 0; // Just for debug / printing (seems something's wrong: more merges than streams)

//#define OLD_MASK_BUILD
//#define OLD_PEAK_ASSIGN



// After giving one buffer in one call (new_buffer), at the end of the next (blocking) call, it can be released.
// Cannot be reassigned to a different output because it stores the previous buffer and it would have conflicts, thus all the data must be passed 
size_t write_data(Buffers<real> &o, Buffers<real> *new_buffers, const size_t FFT_N, const size_t FFT_slide)
{
  static int overlapping_buffers = 1; // how many buffers are in current use (state variable)
  static Buffers<real> *a=new_buffers, *b=NULL;
  static size_t i = 0, p = 0;
  unsigned int buffers = o.buffers();

  if (FFT_slide < FFT_N) // Up to 50% overlap
    {
      if (overlapping_buffers == 2)
	{
	  b = new_buffers;
	  while (i < FFT_N)
	    {
	      //o[p] = a[i] + b[i-FFT_slide];
	      for (uint buf = 0; buf < buffers; ++buf)
		(*o(buf))[p] = (*(*a)(buf))[i] + (*(*b)(buf))[i-FFT_slide];
	      ++p;
	      ++i;
	    }
	  // RELEASE(a)
	  i = FFT_N-FFT_slide;
	  a = b;
	  overlapping_buffers = 1;
	}

      // overlapping_buffers == 1
      while (i < FFT_slide)
	{
	  // o[p] = a[i]
	  for (uint buf = 0; buf < buffers; ++buf)
	    (*o(buf))[p] = (*(*a)(buf))[i];
	  ++p;
	  ++i;
	}
      overlapping_buffers = 2;
      // Now wait for new call with new_buffer 
    }
  else // No overlap
    {
      i = 0;
      a = new_buffers;
      while (i < FFT_slide) // == FFT_N
	{
	  // o[p] = a[i]
	  for (uint buf = 0; buf < buffers; ++buf)
	    (*o(buf))[p] = (*(*a)(buf))[i];	      
	  ++p;
	  ++i;
	}
    }

  return p;
}
   
struct Cfg
{
  unsigned int FFT_N, FFT_slide;
  
  real eta, beta, alpha;
};

/**
   Arguments: prgm [FFT_N] [x1_wav] [x2_wav]
*/
void StationaryNoisePSD(Matrix<real> &X_history, Buffer<real> &NPSD, idx tb_start, idx tb_end)
{
  static const idx K = NPSD.size(), FFT_N = K*2;
  static Buffer<real> P(NPSD);

  real *X = NULL;

  NPSD.clear();
  for (idx tb = tb_start; tb < tb_end; ++tb)
    {
      X = X_history(tb);

      evenHC2power(FFT_N, X, P());
      
      NPSD += P;
    }
  NPSD /= tb_end - tb_start; 
}

void SSub(real *X, Buffer<real> &NPSD, const Cfg &cfg)
{
  static const idx K = NPSD.size(), FFT_N = K*2;
  
  for (idx k = 0; k < K; ++k)
    {
      std::complex<real> C(X[k], X[FFT_N-k]);

      real Xk_power = std::norm(C);
      real Yk_power = Xk_power - cfg.beta * NPSD[k];

      if (Yk_power < 0)
	{
	  X[k]       = cfg.eta * X[k]      ;
	  X[FFT_N-k] = cfg.eta * X[FFT_N-k];
	}
      else
	{
	  std::complex<real> Ck ( std::polar<real>(std::sqrt(Yk_power), std::arg(C)) );
	  X[k]       = Ck.real();
	  X[FFT_N-k] = Ck.imag();
	}
    }
}

int main(int argc, char **argv)
{

  /* Name convention throughout this file:
  	
     i - input
     o - output
     m - magnitude

     and capital letters for the frequency domain
  */	

  Options _o("presets.cfg", Quit, 1); // Choose the preset file.
  Options o (_o("ssub_preset").c_str(), Quit, 1); // True configuration file (preset)

  OptionParser opt;
  opt.setFlag("noSTFTcorrection");
  opt.setFlag('p'); // Plot the clean signal

  int arg0 = opt.parse(argc, argv);

  const bool APPLY_STFT_CORRECTION = ! opt.getFlag("noSTFTcorrection");

  unsigned int FFT_N = o.i("FFT_N");
  unsigned int FFT_slide = FFT_N * (o.i("FFT_slide_percentage")/100.0);

  Cfg cfg;
  cfg.FFT_N = FFT_N;
  cfg.FFT_slide = FFT_slide;
  cfg.beta = o.f("beta");
  cfg.eta = o.f("eta");
  cfg.alpha = o.f("alpha");

  // Choose mic input files
  Guarantee(argc-arg0 == 2, "Usage: ssub <in> <out>");
  std::string I_filepath = argv[arg0]; 


  SndfileHandle I(I_filepath);
  Guarantee(wav::ok(I)  , "Input file doesn't exist.");
  Guarantee(wav::mono(I), "Input files must be mono.");

  const uint sample_rate_Hz = I.samplerate();
  const idx  samples_input  = I.frames(); 
  const real Ts             = 1.0/(real)sample_rate_Hz;
  	
  printf(YELLOW "\nProcessing input file with %lu frames @ %u Hz.\n\n" NOCOLOR, samples_input, sample_rate_Hz);	
  Guarantee0(FFT_N % 2, "System implemented for FFTs with even size.");
  Guarantee(FFT_slide <= FFT_N, "FFT_slide(%u) > FFT_N(%u)", FFT_slide, FFT_N);

  const idx time_blocks = div_up<size_t>(samples_input-FFT_N,FFT_slide) + 1;
  const size_t samples_all = time_blocks*FFT_slide+(FFT_N-FFT_slide);

  const size_t skip_samples = o.f("skip_time_s")*sample_rate_Hz;

  // Silence blocks
  const unsigned int Sb = div_up<size_t>(skip_samples,FFT_slide);


  //// Storage allocation ///////

  Buffer<real> wav(samples_all);
  I.read(wav(), samples_input);


  // Initialize the buffers all with the same characteristics and aligned for FFTW use.
  Buffer<real> x(FFT_N, 0, fftw_malloc, fftw_free), X(x), M(FFT_N/2), NPSD(M);
  

  // We're going to save at least one of the microphone transforms for all time blocks for the static heuristic reconstruction
  Matrix<real> X_history(time_blocks, FFT_N);

  Buffer<real> wav_out(time_blocks*FFT_slide+(FFT_N-FFT_slide), 0, fftw_malloc, fftw_free);
  
  const real FFT_df = sample_rate_Hz / (real) FFT_N;

  Buffer<real> f_axis(FFT_N/2);
  f_axis.fill_range(0,0.5*(real)sample_rate_Hz);

  fftw_plan xXplan, Xxplan;
  cout << "Estimating FFT plan..." << endl;
  cout << "The fast way!\n";
  int FFT_flags = FFTW_ESTIMATE;
  xXplan = fftw_plan_r2r_1d(FFT_N, x(), X(), FFTW_R2HC, FFT_flags); 
  Xxplan = fftw_plan_r2r_1d(FFT_N, X(), x(), FFTW_HC2R, FFT_flags); 
  cout << "DONE" << endl;

  Buffer<real> W(FFT_N);
  select_window(o("window"), W);
  Buffer<real> W2(W); W2 *= W; // W2 = W*W
  Buffer<real> invWenvelope(samples_all);
  for (idx tb=0; tb < time_blocks; ++tb)
    invWenvelope.add_at(W2,tb*FFT_slide);
  for (size_t t=1; t < samples_all-1; ++t) // Beginning and ending samples will be 0.
    invWenvelope[t] = 1/invWenvelope[t];
  

  for (idx time_block = 0; time_block < time_blocks; ++time_block)
    {
      idx block_offset = time_block*FFT_slide;

      for (idx i = 0; i < FFT_N; ++i)
	x[i] = wav[i+block_offset] * W[i];


      fftw_execute(xXplan);

      // Keep the record of X1 for all time for later audio reconstruction
      for (idx f = 0; f < FFT_N; ++f)
	X_history(time_block,f) = X[f];
    } // End of blockwise processing
  

  // Estimate noise and subtract weakly several times.
  for (int iterations = 0; iterations < o.i("iterations"); ++iterations)
    {
      if (o.i("use_final_silence"))
	StationaryNoisePSD(X_history, NPSD, (long int)time_blocks-(long int)Sb, (long int)time_blocks);	  
      else
	  StationaryNoisePSD(X_history, NPSD, 0, Sb);

      for (idx tb = 0; tb < time_blocks; ++tb)
	SSub(X_history(tb), NPSD, cfg);
    }



  for (idx tb = 0; tb < time_blocks; ++tb)
    {
      fftw_execute_r2r(Xxplan, X_history(tb), x());
      if (APPLY_STFT_CORRECTION)
	x *= W;
      wav_out.add_at(x, tb*FFT_slide);
    }

  wav_out /= FFT_N;

  if (APPLY_STFT_CORRECTION)
    wav_out *= invWenvelope;

  if (opt.getFlag('p'))
    {
      Gnuplot p;
      p.plot(wav_out(),wav_out.size(),"wav_out");
      wait();
    }

  std::string wav_filepath(argv[arg0+1]);
  printf("%s...", wav_filepath.c_str());
  print_status( wav::write_mono(wav_filepath, wav_out(), samples_input, sample_rate_Hz) );
  
	

  fftw_destroy_plan(xXplan); 
  fftw_destroy_plan(Xxplan);

  return 0;
}

