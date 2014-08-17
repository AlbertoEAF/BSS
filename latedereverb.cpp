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

// Shifted Rayleigh distribution: Could be more efficient by pre-computing the whole window once
real w(int int_m, int int_b)
{
  if (int_m <= -int_b)
    return 0;
  else
    {
      real m = int_m;
      real b = int_b;
      real b2 = b*b;
      
      return (m+b)/b2*std::exp( -(m+b)*(m+b)/(2*b2) );      
    }
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



real lambda_k_l(int k, real mu, int b, int rho, int m, Matrix<real> &R)
{
  long int N = R.cols();

  real lambda = 0;

  int t;


  for (int t=0; t<rho && m-t>=0; ++t)
    {

      lambda += w(t-rho,b) * std::abs(std::complex<real>(R(m-t)[k],R(m-t)[N-k]));
    }
  lambda *= mu;



  return lambda;
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
	
  real FFT_length_ms = o.f("FFT_length_ms");

  idx FFT_N = (real)sample_rate_Hz * FFT_length_ms / 1000.0;
  FFT_N += (FFT_N%2);


  _DUET.FFT_N = FFT_N;
 
  _DUET.FFT_slide_percentage = o.i("FFT_slide_percentage", Warn);
  if (! _DUET.FFT_slide_percentage)
    _DUET.FFT_slide_percentage = 100;

  _DUET.FFT_slide = FFT_N * (_DUET.FFT_slide_percentage/100.);
  Guarantee(_DUET.FFT_slide <= _DUET.FFT_N, "FFT_slide(%ld) > FFT_N(%ld)", _DUET.FFT_slide, _DUET.FFT_N);
  printf(YELLOW "FFT_N = %ld\n" "FFT_slide = %ld (%ld%%)\n" NOCOLOR, FFT_N, _DUET.FFT_slide, _DUET.FFT_slide_percentage);

  const idx FFT_slide = _DUET.FFT_slide;

  // Otherwise would require triple-buffering
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
  Matrix<real> wav_out(2, time_blocks*FFT_slide);

  const real FFT_df = sample_rate_Hz / (real) FFT_N;
  FFT_flags = FFTW_ESTIMATE; // Use wisdom + FFTW_EXHAUSTIVE later!

  cout << "Estimating FFT plan..." << endl;
  cout << "The fast way!\n";
  FFT_flags = FFTW_ESTIMATE;
  xX1_plan = fftw_plan_r2r_1d(FFT_pN, x1(), X1(), FFTW_R2HC, FFT_flags); 
  xX2_plan = fftw_plan_r2r_1d(FFT_pN, x2(), X2(), FFTW_R2HC, FFT_flags);
  Xxo_plan = fftw_plan_r2r_1d(FFT_pN, Xo(), xo(), FFTW_HC2R, FFT_flags); 
  cout << "DONE" << endl;


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

  const real mu = o.f("mu");
  const int b = o.i("b");
  const real eta = o.f("eta");
  const real alpha = o.f("alpha");
  const real beta = o.f("beta");
  const int rho = o.i("rho");
  const real Gf = o.f("Gf");

  const int K = FFT_N/2;
  Buffer<real> last_xi(K), xi(K);
  Buffer<real> G(K);


  int m = 0;

  // Process the signal and reassemble it
  for (idx m = 0; m < time_blocks; ++m)
    {
      
      ///////////////// Signal processing upon bufs_ptr ////////////////////
      for (int k=0; k < K; ++k)
	{
	  real gamma_k = X1_history(m,k)*X1_history(m,k) / lambda_k_l(k,mu, b, rho, m, X1_history);
	  xi[k] = eta*last_xi[k] + (1-eta)*std::max(gamma_k - 1.0, 0.0);
	  
	  G[k] = std::pow(xi[k]/(xi[k]+alpha), beta);
	  if (G[k] < Gf)
	    G[k] = Gf;
	}
      last_xi.copy(xi);

      for (int k=0; k<K; ++k)
	if (std::isnan(last_xi[k]) or std::isinf(last_xi[k]))
	  last_xi[k] = 0;

      //cout << last_xi;
      for (int k=0; k<K; ++k)
	Xo[k] = G[k]*X1_history(m,k);
      fftw_execute(Xxo_plan);
      /*
      static Gnuplot p;
      p.replot((*bufs_ptr)(0),xo.size(),"xo");
      usleep(10000);
      */
    

      for (int k=0; k<2*K; ++k)
	(*bufs_ptr)(0)[k] = xo[k];

      //////////////////////////////////////////////////////////////////////

      write_data(wav_out, bufs_ptr, FFT_N, FFT_slide);
      swap(bufs_ptr, bufs2_ptr);
    }		


  for (size_t i=0; i < wav_out.cols(); ++i)
    if (std::isnan(wav_out(0)[i]))
      wav_out(0)[i] = 0;



  //  cout << Buffer<real>(wav_out(0),wav_out.cols()) ;

  Buffer<real> x1_wav_normalized(x1_wav,x1_wav.size()), wav_out1_normalized(wav_out(0),wav_out.cols());

  x1_wav_normalized /= array_ops::max_abs(x1_wav(),x1_wav.size());
  wav_out1_normalized /= array_ops::max_abs(wav_out(),wav_out.cols());

  std::string wav_filepath("late_dereverberated_x");
  puts(RED "Review wav::write -> wrong normalization procedure as well as write_mono. In all programs.\n" NOCOLOR);
  print_status( wav::write_mono(wav_filepath+"0.wav", wav_out(0), wav_out.cols(), sample_rate_Hz, array_ops::max_abs(x1_wav( ),x1_wav.size()) ));
  
  fftw_destroy_plan(xX1_plan); 
  fftw_destroy_plan(xX2_plan);
  fftw_destroy_plan(Xxo_plan);
	
  Gnuplot pInL, pOutL;
  pInL.plot(x1_wav_normalized(),x1_wav.size(), "in L");
  pInL.plot(wav_out1_normalized(0),wav_out.cols(), "out L");

  if (o.i("wait"))
    wait();

  return 0;
}
