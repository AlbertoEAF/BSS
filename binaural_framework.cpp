// Performs late dereverberation (Blind Reverberation Mitigation for Robust Speaker Identification)

#include "duet.h"


//#define OLD_MASK_BUILD
//#define OLD_PEAK_ASSIGN

const int MAX_PRECLUSTERS = 12;
const int MAX_MARGINAL_PEAKS = 16;


const real _2Pi = 2*M_PI;

template <class T> void print(T o) { cout << o << endl; }

template <class T> 
void swap (T &a, T &b)
{
  T tmp = a;
  a = b;
  b = tmp;
}


real norm(real a, real b) { return std::sqrt(a*a + b*b); }

/// Returns the success state of the input and prints [DONE] or [FAIL] accordingly.
bool print_status (bool success)
{
  if (success)
    puts(GREEN "[DONE]" NOCOLOR);
  else
    puts(RED "[FAIL]" NOCOLOR);

  return success;
}

/* READ!!: http://www.fftw.org/doc/The-Halfcomplex_002dformat-DFT.html */
/// Specialization for even sizes
void evenHC2magnitude(int samples, real *hc, real *magnitude)
{
  magnitude[0] = hc[0];
  idx I = samples/2;
  for (idx i=1; i < I; ++i)
    magnitude[i] = norm(hc[i], hc[samples-i]); // Not true for odd samples!!!
}

int valid_FFT_convolution(idx h_nonzero_size, idx FFT_N)
{
  idx g_nonzero_size = FFT_N - h_nonzero_size + 1;
  printf(
	 "FFT convolution with \n"
	 "    FFT_N  = %ld\n"
	 "    g_size = %ld\n"
	 "    h_size = %ld\n",
	 FFT_N, g_nonzero_size, h_nonzero_size);
  if (g_nonzero_size < 1)
    {
      puts("Invalid configuration!");
      return 0;
    }
  return 1;
}


inline void fillFFTblock(real *data, idx data_size, real *block, idx block_size)
{
  idx i = 0;
  for (; i < data_size; ++i)
    block[i] = data[i];
  for (; i < block_size; ++i)
    block[i] = 0.0;
  // memset((void*)wav_out, 0, sizeof(real) * (N_wav+h_size-1));
  // memcpy()
}

/**
   Z = Z1*Z2
   @param[in] re1 - Re{Z1}
   @param[in] im1 - Im{Z1}
   @param[in] re2 - Re{Z2}
   @param[in] im2 - Im{Z2}
   @param[out] re - Re{Z}
   @param[out] im - Im{Z}

*/
inline void complex_multiply(real re1, real im1, real re2, real im2, real *re, real *im)
{
  *re = re1*re2 - im1*im2;
  *im = re1*im2 + im1*re2;
}

// z = z1/z2
inline void complex_divide(real re1, real im1, real re2, real im2, real *re, real *im)
{
  real denominator = re2*re2 + im2*im2;

  *re = (re1*re2 + im1*im2) / denominator;
  *im = (im1*re2 - re1*im2) / denominator;
}

/**
   HalfComplex representation multiply
   @param[in] z1 - Input  HC array
   @param[in] z2 - Input  HC array
   @param[out] z - Output HC array
   @param[in] size - Size of the HC array

   @warn: ONLY FOR EVEN TRANSFORMATIONS!!!
*/
void hc_multiply (real *z1, real *z2, real *z, idx size)
{
  z[0] = z1[0]*z2[0];

  const idx max_i = size/2;
  for (idx i=1; i < max_i; ++i)
    complex_multiply(z1[i], z1[size-i],
		     z2[i], z2[size-i],
		     &z[i], &z[size-i]);
}

template <class T> T blocks (T terms, T block_size)
{
  return terms/block_size + ( terms % block_size ? 1:0 );
}




// L2-norm for a vector with start and end point a, b
real distance(const Point2D<real> &a, const Point2D<real> &b)
{
  return norm(b.x-a.x, b.y-a.y);
}



// Window functions
real Hann(idx n, idx N) 
{ 
  return 0.5 * (1.0 - std::cos(_2Pi*n/(N-1.0))); 
}
real Hamming0(idx n, idx N) 
{ 
  return 0.53836 + 0.46164*std::cos(_2Pi*n/(N-1.0)); 
}
real Hamming(idx n, idx N) 
{ 
  //  return 1 - (0.53836 + 0.46164*std::cos(_2Pi*n/(N-1.0))); 
  return 0.46164 - 0.46164*std::cos(_2Pi*n/(N-1.0));
}
real Rectangular(idx n, idx N)
{
  return 1;
}

#define RELEASE(x) {}

/*
/// Copy one column from one matrix to another
void copycol(Matrix<real> &b, Matrix<real> &a, size_t to_col, size_t from_col)
{
  Assert(b.cols() == a.cols() && b.size() == a.size(), "Different sizes!");
  
  for (size_t row = 0; row < a.rows(); ++row)
    b(row,to_col) = a(row,from_col);
}
*/

// After giving one buffer, at the end of the next (blocking) call, it can be released
// Cannot be reassigned to a different output because it stores the previous buffer and it would have conflicts, thus all the data must be passed 
size_t write_data(Matrix<real> &o, Matrix<real> *new_buffer, const size_t FFT_N, const size_t FFT_slide)
{
  static int buffers = 1; // how many buffers are in current use and need to be summed (state variable)
  static Matrix<real> *a=new_buffer, *b=NULL;
  static size_t i = 0, p = 0;
  
  if (FFT_slide < FFT_N) // Up to 50% overlap
    {
      if (buffers == 2)
	{
	  b = new_buffer;
	  while (i < FFT_N)
	    {
	      //o[p] = a[i] + b[i-FFT_slide];
	      for (uint row = 0; row < o.rows(); ++row)
		o(row, p) = (*a)(row,i) + (*b)(row,i-FFT_slide);
	      ++p;
	      ++i;
	    }
	  RELEASE(a);
	  i = FFT_N-FFT_slide;
	  a = b;
	  buffers = 1;
	}

      // Buffers == 1
      while (i < FFT_slide)
	{
	  // o[p] = a[i]
	  for (uint row = 0; row < o.rows(); ++row)
	    o(row, p) = (*a)(row, i);
	  ++p;
	  ++i;
	}
      buffers = 2;
      // Now wait for new call with new_buffer 
    }
  else // No overlap
    {
      i = 0;
      a = new_buffer;
      while (i < FFT_slide) // == FFT_N
	{
	  // o[p] = a[i]
	  for (uint row = 0; row < o.rows(); ++row)
	    o(row,p) = (*a)(row, i);	      
	  ++p;
	  ++i;
	}
    }

  return p;
}




void build_window(Buffer<real> &W, real (*Wfunction)(idx n, idx N))
{
  idx N = W.size();
  for (idx n=0; n < N; ++n)
      W[n] = Wfunction(n,N);
}


void late_dereverberation(Buffer<real> &x)
{
  for (size_t i=0; i < x.size(); ++i)
    {
      
    }
}


/**
   Arguments: prgm [FFT_N] [x1_wav] [x2_wav]
*/
int main(int argc, char **argv)
{
  /* Name convention throughout this file:
  	
     i - input
     o - output
     m - magnitude

     and capital letters for the frequency domain
  */	
  /*
    Histogram2D<real> hi(3,5, -2,2,-5,5, HistogramBounds::Boundless);
    hi.bin(1,0) = 3;
    print(hi);
    return 1;
  */
  
  /*
  // Test output overlap
  Matrix<real> a(1,1000), b(1,10000);
  for (int i=0; i < 1000; ++i)
    a(0,i) = Hann(i,1000);
  for (int loop=0; loop<5;++loop)
    write_data(b, &a, 1000, 990);
  Gnuplot p;
  p.plot_y(b(),10000,"Hann");
  wait();
  return 0;
  */

  Options o("dereverberation.cfg", Quit, 1);
  DUETcfg _DUET; // Just to initialize, then a const DUET is initialized from this one.

  // Convolution Smoothing tests //////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  /*
  Histogram<real> 
    halpha(o.d("hist.dalpha"), o.d("alpha.min"), o.d("alpha.max"), HistogramBounds::Boundless),
    hdelta(o.d("hist.ddelta"), o.d("delta.min"), o.d("delta.max"), HistogramBounds::Boundless);

  static Buffer<real> 
    conv_kernel_alpha(halpha.gen_gaussian_kernel(o.f("hist.smoothing_Delta_alpha"))),
    conv_kernel_delta(hdelta.gen_gaussian_kernel(o.f("hist.smoothing_Delta_delta"))),
    conv_halpha(halpha.bins()), 
    conv_hdelta(hdelta.bins());



  Gnuplot ppa,ppd;

  halpha(0) += 1;
  hdelta(0.0001) += 1;
  
  halpha.kernel_convolution(conv_kernel_alpha, conv_halpha);
  hdelta.kernel_convolution(conv_kernel_delta, conv_hdelta);

  Buffer<real> delta_axis(hdelta.bins());
  for (size_t i=0; i<delta_axis.size(); ++i)
    delta_axis[i] = hdelta.min() + i*hdelta.dx();

  ppa.plot((*halpha.raw())(),halpha.bins(),"alpha");
  ppd.plot(delta_axis(),(*hdelta.raw())(),hdelta.bins(),"delta");
  */
  /////////////////////////////////////////////////////////////////////////////////////////////

  
  int WAIT = o.i("wait");

  fftw_plan xX1_plan, xX2_plan, Xxo_plan;
  int FFT_flags;

  
  


  // Choose mic input files
  std::string x1_filepath = (argc == 4 ? argv[2] : o("x1_wav"));
  std::string x2_filepath = (argc == 4 ? argv[3] : o("x2_wav"));




  SndfileHandle x1_file(x1_filepath), x2_file(x2_filepath);
  Guarantee(wav::ok(x1_file) && wav::ok(x2_file) , "Input file doesn't exist.");
  Guarantee(wav::mono(x1_file) && wav::mono(x2_file), "Input files must be mono.");

  const uint sample_rate_Hz = x1_file.samplerate();
  const idx  samples        = x1_file.frames(); 

  Buffer<real> x1_wav(samples), x2_wav(samples);
  x1_file.read(x1_wav(), samples);
  x2_file.read(x2_wav(), samples);

  // Only x1's are needed since that's the chosen channel for source separation
  	
  printf("\nProcessing input file with %lu frames @ %u Hz.\n\n", 
	 samples, sample_rate_Hz);	
  printf("Max int: %d\n"
	 "Max idx: %ld\n", INT_MAX, LONG_MAX);
  printf("Indexing usage: %.2f%%\n\n", 0.01*(float)x1_file.frames()/(float)LONG_MAX);
	
  const idx FFT_N = (argc > 1 ? (idx)strtol(argv[1], NULL, 10) : o.i("FFT_N"));
  _DUET.FFT_N = FFT_N;
  Guarantee0(FFT_N % 2, "System implemented for FFTs with even size.");
 
  _DUET.FFT_slide_percentage = o.i("FFT_slide_percentage", Warn);
  if (! _DUET.FFT_slide_percentage)
    _DUET.FFT_slide_percentage = 100;

  _DUET.FFT_slide = FFT_N * (_DUET.FFT_slide_percentage/100.);
  Guarantee(_DUET.FFT_slide <= _DUET.FFT_N, "FFT_slide(%ld) > FFT_N(%ld)", _DUET.FFT_slide, _DUET.FFT_N);
  printf(YELLOW "FFT_N = %ld\n" "FFT_slide = %ld (%ld%%)\n" NOCOLOR, FFT_N, _DUET.FFT_slide, _DUET.FFT_slide_percentage);

  const idx FFT_slide = _DUET.FFT_slide;

  // This will require triple-buffering
  Guarantee(FFT_slide >= FFT_N/2, "FFT_slide(%ld) > FFT_N/2(%ld)", FFT_slide, FFT_N/2);

  _DUET.use_window = 1;
    
  // Frequency oversampling
  _DUET.FFT_p = o.i("FFT_oversampling_factor");
  _DUET.FFT_pN = _DUET.FFT_p * _DUET.FFT_N;
  const idx FFT_pN = _DUET.FFT_pN;

  const uint time_blocks = 1 + blocks(samples, FFT_slide);

  //// Storage allocation ///////

  // Initialize the buffers all with the same characteristics and aligned for FFTW use.
  Buffer<real> x1(FFT_pN, 0, fftw_malloc, fftw_free), x2(x1), X1(x1), X2(x1), xo(x1), Xo(x1);


  // We're going to save at least one of the microphone transforms for all time blocks for the static heuristic reconstruction
  Matrix<real> X1_history(time_blocks, FFT_pN), X2_history(time_blocks, FFT_pN);

  // Organized as (time, frequency)
  Matrix<real,MatrixAlloc::Rows> 
    alpha(time_blocks, FFT_pN/2), 
    delta(time_blocks, FFT_pN/2), 
    wav_out(2, time_blocks*FFT_slide);

  const real FFT_df = sample_rate_Hz / (real) FFT_N;
  FFT_flags = FFTW_ESTIMATE; // Use wisdom + FFTW_EXHAUSTIVE later!

  cout << "Estimating FFT plan..." << endl;
  cout << "The fast way!\n";
  FFT_flags = FFTW_ESTIMATE;
  xX1_plan = fftw_plan_r2r_1d(FFT_pN, x1(), X1(), FFTW_R2HC, FFT_flags); 
  xX2_plan = fftw_plan_r2r_1d(FFT_pN, x2(), X2(), FFTW_R2HC, FFT_flags);
  Xxo_plan = fftw_plan_r2r_1d(FFT_pN, Xo(), xo(), FFTW_HC2R, FFT_flags); 
  cout << "DONE" << endl;

  const HistogramBounds::Type hist_bound_type = ( o.i("hist.bounds") ? HistogramBounds::Boundless : HistogramBounds::DiscardBeyondBound ); 

  Histogram2D<real> hist(o.d("hist.dalpha"), o.d("hist.ddelta"),
			 o.d("alpha.min"), o.d("alpha.max"), 
			 o.d("delta.min"), o.d("delta.max"), 
			 hist_bound_type);

  if (hist.bins() > 1e6)
    {
      puts(RED "Exiting: Too many bins" NOCOLOR);
      exit(1);
    }

  Histogram<real> 
    hist_alpha(o.d("hist.dalpha"), o.d("alpha.min"), o.d("alpha.max"), hist_bound_type),
    hist_delta(o.d("hist.ddelta"), o.d("delta.min"), o.d("delta.max"), hist_bound_type);
  Histogram2D<real> cumulative_hist(hist), old_hist(hist);
  Buffer<real> alpha_range(hist_alpha.bins()), delta_range(hist_delta.bins()); // Values for the axis of alpha and delta
  alpha_range.fill_range(o.d("alpha.min"), o.d("alpha.max"));
  delta_range.fill_range(o.d("delta.min"), o.d("delta.max"));

  hist.print_format();

  /*
  cumulative_hist.clear();
  cumulative_hist(-0.12,5e-6) += 10;
  cumulative_hist.smooth_add(1, -0.12, 5e-6, 1.1e-2, 6e-7);
  Gnuplot cumulative_hist_plot;
  cumulative_hist_plot.cmd("set xlabel 'alpha'; set ylabel 'delta (s)'");
  cumulative_hist.plot(cumulative_hist_plot, "Cumulative Histogram");
  wait();
  return 1;
  */






  const DUETcfg DUET = _DUET; // Make every parameter constant to avoid mistakes



  Buffer<real> W(FFT_N);
  if (o("window",Ignore) == "Hamming0")
    {
      puts(YELLOW "W=Hamming0" NOCOLOR);
      build_window(W,Hamming0);      
    }
  else if (o("window",Ignore) == "Hamming")
    {
      puts(YELLOW "W=Hamming" NOCOLOR);
      build_window(W,Hamming);      
    }
  else if (o("window",Ignore) == "Hann")
    {
      puts(YELLOW "W=Hann" NOCOLOR);
      build_window(W,Hann);
    }
  else
    {
      puts(YELLOW "W=Rectangular" NOCOLOR);
      build_window(W,Rectangular);
    }

  /*
  Gnuplot Wplot;
  Wplot.plot_y(W(),W.size(),"W");
  wait();
  */
  for (idx time_block = 0; time_block < time_blocks; ++time_block)
    {
      idx block_offset = time_block*FFT_slide;

      for (idx i = 0; i < FFT_N; ++i)
	{
	  idx offset_i = i+block_offset;

	  if (offset_i < samples)
	    {
	      x1[i] = x1_wav[offset_i] * W[i];
	      x2[i] = x2_wav[offset_i] * W[i]; 	
	    }
	  else // end of file: fill with zeros
	    {
	      x1[i] = 0;
	      x2[i] = 0;
	    }
	}


      fftw_execute(xX1_plan);
      fftw_execute(xX2_plan);
  
      // Keep the record of X1 for all time for later audio reconstruction
      for (idx f = 0; f < FFT_pN; ++f)
	{
	  X1_history(time_block,f) = X1[f];
	  X2_history(time_block,f) = X2[f];
	}
    }



  // 2 sets of buffers are needed to allow up to 50% overlapping.
  Matrix<real> 
    bufs1(2, FFT_pN), *bufs_ptr  = &bufs1,
    bufs2(2, FFT_pN), *bufs2_ptr = &bufs2;



  // Process the signal and reassemble it
  for (idx t_block = 0; t_block < time_blocks; ++t_block)
    {
      // Signal processing upon bufs_ptr
      

      write_data(wav_out, bufs_ptr, FFT_N, FFT_slide);
      swap(bufs_ptr, bufs2_ptr);
    }		

  std::string wav_filepath("late_dereverberated_x");
  print_status( wav::write(wav_filepath, wav_out, sample_rate_Hz) );
  
  fftw_destroy_plan(xX1_plan); 
  fftw_destroy_plan(xX2_plan);
  fftw_destroy_plan(Xxo_plan);
	

  if (WAIT)
    wait();

  return 0;
}
