#include "duet.h"


//#define OLD_MASK_BUILD
//#define OLD_PEAK_ASSIGN


const int MAX_MARGINAL_PEAKS = 16;

void RENDER_HIST(const std::string &filepath, const std::string &title, bool pause)
{
  std::string cmd("gnuplot -e \"splot \\\"");

  cmd += filepath;
  cmd += "\\\" u 1:2:3 w pm3d title \\\"";
  cmd += title;
  cmd += "\\\", \\\"s.dat\\\"  pt 7 ps .9 title \\\"Simulation clusters\\\", \\\"s_duet.dat\\\" pt 8 ps .8 title \\\"DUET clusters\\\"; set xlabel \\\"alpha\\\"; set ylabel \\\"delta (s)\\\";";
  if (pause)
    cmd += "pause -1";
  cmd += "\"";

  system(cmd.c_str()); 
}

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


void heuristic_pre_filter_clusters (Histogram<real> &hist, RankList<real,real> &preclusters, real min_peak_fall)
{
  static const size_t skip_bins = 0; // skip the next bin if this one is below the noise threshold for faster performance (for sure a peak will not arise in the next bins)

  const size_t max_bin = hist.bins() - 1;	

  preclusters.clear();
  // Exclude bins on the border (Borderless histogram and interior region = interest region)
  for (size_t bin=1; bin < max_bin; ++bin)
    {
      static real min_peak_diff = 0; // (10) sets the minimum (-)derivative around the peak.
      static real min_score     = 0; // (50) Minimum threshold, below which is noise.

      real score = hist.bin(bin);

      if (score > min_score)
	{
	  if (score - hist.bin(bin-1) >= min_peak_fall && 
	      score - hist.bin(bin+1) >= min_peak_fall)
	    preclusters.add(score, hist.get_bin_center(bin));
	}
      else
	bin += skip_bins; // skip (faster but might not be safe - we don't want to skip over a peak) 
    }
}


void heuristic_pre_filter_clusters2D (Histogram2D<real> &hist, RankList<real, Point2D<real> > &preclusters, real min_peak_fall)
{
  static const size_t skip_bins = 0; // skip the next bin if this one is below the noise threshold for faster performance (for sure a peak will not arise in the next bins)

  const size_t max_alpha_bin = hist.xbins() - 1;	
  const size_t max_delta_bin = hist.ybins() - 1;

  preclusters.clear();
  // Bins on the histogram border are filtered out! We only want peaks in the histogram's interior region (since we're using a borderless histogram we filter out peaks from outside the chosen interest region this way)
  for (size_t alphabin=1; alphabin < max_alpha_bin; ++alphabin)
    {
      for (size_t deltabin=1; deltabin < max_delta_bin; ++deltabin)
	{				
	  static real min_peak_diff = 0; // (10) sets the minimum (-)derivative around the peak.
	  static real min_score     = 0; // (50) Minimum threshold, below which is noise.

	  real score = hist.bin(alphabin,deltabin);

	  if (score > min_score)
	    {
	      if (score - hist.bin(alphabin-1,deltabin  ) >= min_peak_fall && 
		  score - hist.bin(alphabin+1,deltabin  ) >= min_peak_fall &&
		  score - hist.bin(alphabin  ,deltabin-1) >= min_peak_fall &&
		  score - hist.bin(alphabin  ,deltabin+1) >= min_peak_fall)
		preclusters.add(score, hist.get_bin_center(alphabin,deltabin));
	    }
	  else
	    deltabin += skip_bins; // skip the next one (faster) (if there's no points at this bin there isn't a peak in the next bin for sure) 
	}
    }
}

/// Returns |a-b| for unsigned int type size_t
inline size_t subabs(size_t a, size_t b)
{
  return ( a > b ? a-b : b-a );
}



// Checks if the distance between the 2 points is smaller than d_delta and d_alpha for each dimension respectively. Independent Box coordinates!
bool belongs (const Point2D<real> &a, const Point2D<real> &b, real max_distance_alpha, real max_distance_delta)
{
  return (std::abs(b.x-a.x) <= max_distance_alpha && std::abs(b.y-a.y) <= max_distance_delta ? 1 : 0);
}

/// Aggregates clusters to the biggest cluster inside a certain radius (in box coordinates)
void heuristic_aggregate_preclusters (RankList<real,real> &preclusters, const DUETcfg &DUET, real min_peak_distance)
{
  size_t size = preclusters.eff_size(DUET.noise_threshold);

  real max_score = preclusters.scores[0];
  for (size_t i=0; i < size; ++i)
    if (preclusters.scores[i] * DUET.max_peak_scale_disparity < max_score)
      {
	preclusters.del(i,DUET.noise_threshold);
	--size;
	--i; // just deleted, check again
      }
  if (DUET.aggregate_clusters) // if for test purposes: always ON at release.
    {
      for (size_t i=0; i < size; ++i)
	{
	  for (size_t cluster=0; cluster < i; ++cluster)
	    {
	      if (std::abs(preclusters.values[i]-preclusters.values[cluster]) <= min_peak_distance)
		{
		  preclusters.del(i,DUET.noise_threshold);
		  --size; // Just deleted an element. No need to process extra 0-score entries.
		  --i; // The rest of the list was pushed up, process the next entry which is in the same position.
		}
	    }
	}
    }
}

void heuristic_clustering(Histogram<real> &hist, RankList<real,real> &preclusters, const DUETcfg &DUET, real min_peak_distance)
{
  heuristic_pre_filter_clusters(hist, preclusters, DUET.min_peak_fall);
  //	cout << preclusters;
  heuristic_aggregate_preclusters(preclusters, DUET, min_peak_distance);
}


/// Aggregates clusters to the biggest cluster inside a certain radius (in box coordinates)
void heuristic_aggregate_preclusters2D (RankList<real,Point2D<real> > &preclusters, const DUETcfg &DUET)
{
  size_t size = preclusters.eff_size(DUET.noise_threshold);

  real max_score = preclusters.scores[0];
  for (size_t i=0; i < size; ++i)
    if (preclusters.scores[i] * DUET.max_peak_scale_disparity < max_score)
      {
	preclusters.del(i,DUET.noise_threshold);
	--size;
	--i; // just deleted, check again
      }
  if (DUET.aggregate_clusters) // if for test purposes: always ON at release.
    {
      for (size_t i=0; i < size; ++i)
	{
	  for (size_t cluster=0; cluster < i; ++cluster)
	    {
	      if (belongs(preclusters.values[i], preclusters.values[cluster], DUET.min_peak_dalpha, DUET.min_peak_ddelta))
		{
		  preclusters.del(i,DUET.noise_threshold);
		  --size; // Just deleted an element. No need to process extra 0-score entries.
		  --i; // The rest of the list was pushed up, process the next entry which is in the same position.
		}
	    }
	}
    }
}

void heuristic_clustering2D(Histogram2D<real> &hist, RankList<real, Point2D<real> > &preclusters, const DUETcfg &DUET)
{
  heuristic_pre_filter_clusters2D(hist, preclusters, DUET.min_peak_fall);
  //	cout << preclusters;
  heuristic_aggregate_preclusters2D(preclusters, DUET);
}

// L2-norm for a vector with start and end point a, b
real distance(const Point2D<real> &a, const Point2D<real> &b)
{
  return norm(b.x-a.x, b.y-a.y);
}

// Returns the index of the closest cluster in clusters.
size_t closest_cluster(const Point2D<real> &point, Buffer<Point2D<real> > &clusters)
{
  const size_t size = clusters.size();
	
  real dist, min_distance = FLT_MAX;
  size_t min_i = 0;

  // Find the closest cluster
  for (size_t i=0; i < size; ++i)
    {
      dist = distance(point, clusters[i]);
      if (dist < min_distance)
	{
	  min_distance = dist;
	  min_i = i;
	}
    }

  return min_i;
}

/// Returns the score for the DUET histogram based on the parameters p and q
real DUEThist_score(real x1re, real x1im, real x2re, real x2im, real omega, real p, real q)
{
  real s_re, s_im, s_abs;
  complex_multiply(x1re,x1im, x2re,x2im, &s_re,&s_im);

  s_abs = norm(s_re,s_im);

  return std::pow(s_abs,p)*std::pow(omega,q); // The tables of the powers of omega could be reused.
}

/** Taken from bessel.c, also distributed in this folder. Calculates the modified Bessel function I0. */
double bessi0( double x )
{
  double ax,ans;
  double y;

  if ((ax=fabs(x)) < 3.75) {
    y=x/3.75,y=y*y;
    ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
					 +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
  } else {
    y=3.75/ax;
    ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
					  +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
									     +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
														  +y*0.392377e-2))))))));
  }
  return ans;
}

/**
   @param[in] K - Number of active sources: theta.size() >= K
   @param[in] x - Must have fewer than RAND_MAX elements.
*/
void RANSAC (Buffer<real> &theta, Buffer<real> &x, int K, int RANSAC_samples_per_source)
{
  const int N = x.size();
  const int M = K * RANSAC_samples_per_source;

  const int MAX_N = 5000;
  const int MAX_M = 200; // Maximum number of RANSAC samples.
  static Buffer<int> y(MAX_M);

  Assert(MAX_N > N && MAX_M > M, "RANSAC compile-time constants too small.");

  static Buffer<bool> I(MAX_N * MAX_M); // Since memory must be reused it will be dynamically addressed as if it was a linear 2D-array of dimensions (N,M).

  I.clear();
  for (int m=0; m < M; ++m)
    {
      y[m] = rand() % N;
      //      Fit(y[m])
    }

}



void DUET_hist_add_score(Histogram2D<real> &hist, Histogram<real> &hist_alpha, Histogram<real> &hist_delta, real alpha, real delta, real X1_re, real X1_im, real X2_re, real X2_im, real omega, const DUETcfg &DUET)
{

  if (std::isnan(alpha))
    {
      // This test can be taken out in the final system supposedly because it won't be padded with 0's, instead there will always be noise in the signal, and whatever the noise is it will produce frequencies. The thesis on the circular duet talks about this, that the window should have at least the size of the biggest consecutive chain of 0's possible to appear in the data ( so this doesn't happen ).
      /*
	static size_t alpha_isnan_count = 0;
	++alpha_isnan_count;
	printf("alpha=nan occurred %lu times.\n", alpha_isnan_count);
      */
      //printf("nan value in alpha for t_block=%lu, f=%lu\n", time_block, f);
      return;
    }

  real score = DUEThist_score(X1_re,X1_im, X2_re, X2_im, omega, DUET.p, DUET.q);
	
  /*
    if (DUET.use_smoothing)
    {
    hist.smooth_add(score, alpha, delta, DUET.smoothing_Delta_alpha, DUET.smoothing_Delta_delta);

    hist_alpha.smooth_add(score, alpha, DUET.smoothing_Delta_alpha);
    hist_delta.smooth_add(score, delta, DUET.smoothing_Delta_delta);
    }
    else
    {
    hist(alpha, delta) += score;     

    hist_alpha(alpha) += score;
    hist_delta(delta) += score;
    }
  */
  hist(alpha, delta) += score;     
  hist_alpha(alpha)  += score;
  hist_delta(delta)  += score;
}

/// Calculates (alpha,delta) for a time block and adds to the histogram.
void calc_alpha_delta(idx time_block, idx pN, idx sample_rate_Hz,
		      Buffer<real> &X1, Buffer<real> &X2,
		      Matrix<real,MatrixAlloc::Rows> &alpha, 
		      Matrix<real,MatrixAlloc::Rows> &delta, 
		      Histogram2D<real> &hist, 
		      Histogram<real> &hist_alpha, Histogram<real> &hist_delta, 
		      const DUETcfg &DUET)
{
  static real df = sample_rate_Hz/(real)pN;

  /*
    f = 0 Hz:
    d_Re = X1(0) / X2(0);
    d_Im = 0;
  */
  real a = X2[0] / X1[0];
  alpha(time_block, 0) = a - 1/a;
  delta(time_block, 0) = 0.0;

  idx fI; // imaginary part index
  real _alpha, _delta; // aliases to avoid unneeded array re-access.
  real omega;

  if (DUET.FFT_p > 1) // Use the phase-aliasing correction extension.
    {
      for (idx f = 1; f < pN/2 - 1; ++f)
	{
	  fI = pN-f; // imaginary part index

	  std::complex<real> F   (std::complex<real>(X2[f  ],X2[fI  ])/std::complex<real>(X1[f  ],X1[fI  ]));
	  std::complex<real> Feps(std::complex<real>(X2[f+1],X2[fI-1])/std::complex<real>(X1[f+1],X1[fI-1]));

	  omega = _2Pi * f*df; //  f_Hz = f*df

	  a = std::abs(F);
      
	  _alpha = alpha(time_block, f) = a - 1/a;
	  _delta = delta(time_block, f) = std::fmod(pN/_2Pi * (std::arg(F)-std::arg(Feps)), pN); 

	  DUET_hist_add_score(hist, hist_alpha, hist_delta, _alpha, _delta, X1[f],X1[fI], X2[f],X2[fI], omega, DUET);
	}
    }
  else // Standard DUET without phase-aliasing correction.
    {
      for (idx f = 1; f < pN/2; ++f)
	{
	  idx fI = pN-f; // imaginary part index
		
	  std::complex<real> F(std::complex<real>(X2[f],X2[fI])/std::complex<real>(X1[f],X1[fI]));

	  omega = _2Pi * f*df; //  f_Hz = f*df
	  a = std::abs(F);
	  
	  _alpha = alpha(time_block, f) = a - 1/a;
	  _delta = delta(time_block, f) = - std::arg(F)/omega;

	  DUET_hist_add_score(hist, hist_alpha, hist_delta, _alpha, _delta, X1[f],X1[fI], X2[f],X2[fI], omega, DUET);
	}
    }


  // For the convolution of the histograms

  if (DUET.use_smoothing)
    {
      static Buffer<real> 
	conv_kernel_alpha(hist_alpha.gen_gaussian_kernel(DUET.sigma_alpha)),
	conv_kernel_delta(hist_delta.gen_gaussian_kernel(DUET.sigma_delta)),
	conv_hist_alpha  (hist_alpha.bins()), 
	conv_hist_delta  (hist_delta.bins());

      static Matrix<real> 
	conv_kernel(hist.gen_gaussian_kernel(DUET.sigma_alpha, DUET.sigma_delta)),
	conv_hist  (hist.xbins(),hist.ybins());

      // New blurring method: convolution
      hist_alpha.kernel_convolution(conv_kernel_alpha, conv_hist_alpha);
      hist_delta.kernel_convolution(conv_kernel_delta, conv_hist_delta);
      // WARNING: VERY SLOW OPERATION
      if (DUET.use_smoothing_2D)
	hist.kernel_convolution(conv_kernel, conv_hist); ///SLOW!!
    }
}

void ransac_test(idx time_block, idx pN, idx sample_rate_Hz,
		 Buffer<real> &X1, Buffer<real> &X2,
		 Matrix<real,MatrixAlloc::Rows> &alpha, 
		 Matrix<real,MatrixAlloc::Rows> &delta, 
		 Histogram2D<real> &hist, 
		 Histogram<real> &hist_alpha, Histogram<real> &hist_delta, 
		 const DUETcfg &DUET)
{
  static real df = sample_rate_Hz/(real)pN;

  /*
    f = 0 Hz:
    d_Re = X1(0) / X2(0);
    d_Im = 0;
  */
  real a = X2[0] / X1[0];
  alpha(time_block, 0) = a - 1/a;
  delta(time_block, 0) = 0.0;

  idx fI; // imaginary part index
  real _alpha, _delta; // aliases to avoid unneeded array re-access.
  real omega;

  // ELIMINATE THIS LATER
  static Gnuplot pransac("points");
  pransac.set_labels("delta", "f index");
  static Buffer<real> f_axis(pN/2), delta_axis(pN/2);
  ///////////////////////


  for (idx f = 1; f < pN/2; ++f)
    {
      idx fI = pN-f; // imaginary part index
		
      std::complex<real> F1(X1[f],X1[fI]), F2(X2[f],X2[fI]), F(F2/F1);

      omega = _2Pi * f*df; //  f_Hz = f*df
      a = std::abs(F);
	  

      // Do not override alpha and delta, for those are for DUET right now
      _alpha /*=alpha(time_block, f)*/ = a - 1/a;
      // Wrong

      //_delta = delta(time_block, f) = std::fmod(std::arg(F1) - std::arg(F2), M_PI);///omega;
      //_delta /*= delta(time_block, f)*/ = std::arg(std::polar<real>(1,std::arg(F1) - std::arg(F2)));///omega;
      _delta = std::fmod(std::arg(F1)-std::arg(F2) + M_PI, 2*M_PI) - M_PI;

      //      _delta = std::fmod(std::arg(F1)-std::arg(F2),M_PI);
      f_axis[f] = f;
      delta_axis[f] = _delta;

      //DUET_hist_add_score(hist, hist_alpha, hist_delta, _alpha, _delta, X1[f],X1[fI], X2[f],X2[fI], omega, DUET);
    }
  

  pransac.replot(delta_axis(), f_axis(), pN/2, "Frame RANSAC");
  //wait();

 
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
	  RELEASE(a);
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

/// Transforms alpha back to a.
real alpha2a (real alpha)
{
  return (alpha + std::sqrt(alpha*alpha + 4.0)) * 0.5;
}

/// Fills a buffer of size FFT_N/2 // To each bin will be assigned the number of the source. values < 0 indicate that the bin won't be assigned a source (noise or intentional algorithm rejection/discard). 
/// Thus, a single buffer is required to hold all the masks
/// tmp must have size = max(N_clusters)
void build_masks(Buffer<int> &masks, real *alpha, real *delta, real *X1, real *X2, Buffer<Point2D<real> > &clusters, int N_clusters, idx FFT_N, idx FFT_half_N, real FFT_df, Buffer<real> &tmp)
{
  Buffer<int> old_masks(masks);
  idx masks_diffs = 0;

  for (idx f = 0; f < FFT_half_N; ++f)
    {
      real omega = _2Pi * f * FFT_df;
      idx f_im = FFT_N - f;
      
      for (int k=0; k < N_clusters; ++k)
	{
	  real a_k = alpha2a(clusters[k].x);
	  real delta_k = clusters[k].y;

	  tmp[k] = std::norm(a_k*std::polar<real>(1,-delta_k*omega) * std::complex<real>(X1[f],X1[f_im]) - std::complex<real>(X2[f],X2[f_im])) / (1.0 + a_k*a_k);
	}
      masks[f] = array_ops::min_index(tmp(), N_clusters);

      old_masks[f] = closest_cluster(Point2D<real>(alpha[f],delta[f]), clusters);

      if (masks[f]!=old_masks[f])
	masks_diffs += 1;
    }
#ifdef OLD_MASK_BUILD
  masks = old_masks;
  cout << "#Mask diffs = " << masks_diffs << endl;
#endif // OLD_MASK_BUILD
  //  cout << RED << masks_diffs << NOCOLOR << endl;
}

void apply_masks(Buffers<real> &buffers, real *alpha, real *X1, real *X2, Buffer<int> &masks, Buffer<Point2D<real> > &clusters, uint active_sources, idx FFT_N, idx FFT_half_N, real FFT_df, fftw_plan &FFTi_plan, Buffer<real> &Xo)
{
  /*
    for (uint source = 0; source < active_sources; ++source)
    {
    Xo.clear();

    if (masks[0] == source)
    Xo[0] = X[0];
    for (uint f = 1, f_max = FFT_N/2; f < f_max; ++f)
    {
    if (masks[f] == source)
    {
    uint f_im = FFT_N - f;
    Xo[f   ] = X[f   ];
    Xo[f_im] = X[f_im];
    }
    }
    fftw_execute_r2r(FFTi_plan, Xo(), buffers(source));
    }

    buffers /= (real)FFT_N;
  */

  // Rebuild one source per iteration to reuse the FFT plan (only 1 needed).
  for (uint source = 0; source < active_sources; ++source)
    {
      Xo.clear();


      if (masks[0] == source)
	{
	  real a_k = alpha2a(clusters[source].x);
	  Xo[0] = a_k*X1[0]-X2[0];
	  Xo[0] *= Xo[0] / (1 + a_k*a_k);
	}
      for (uint f = 1, f_max = FFT_N/2; f < f_max; ++f)
	{
	  if (masks[f] == source)
	    {
	      uint f_im = FFT_N - f;
	      real a_k = alpha2a(clusters[source].x);
	      real delta_k = clusters[source].y;
	      real omega = _2Pi * f * FFT_df;

	      std::complex<real> X(std::complex<real>(X1[f],X1[f_im])+std::polar<real>(a_k,delta_k*omega) * std::complex<real>(X2[f],X2[f_im]));

#ifdef OLD_PEAK_ASSIGN
	      Xo[f   ] = X1[f   ];
	      Xo[f_im] = X1[f_im];
#else
	      Xo[f   ] = X.real();
	      Xo[f_im] = X.imag();
#endif // OLD_PEAK_ASSIGN
	    }
	}
      fftw_execute_r2r(FFTi_plan, Xo(), buffers.raw(source));
    }

  buffers /= (real)FFT_N;
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


bool in (int val, Buffer<int> &list)
{
  size_t size = list.size();

  for (size_t i=0; i < size; ++i)
    if (val == list[i])
      return true;

  return false;
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
  /*

    for (int i = 0; i < N; ++i)
    {
    real dtotal = Dtotal(s(i),o(i),samples);

    printf(BLUE "s%d : Dtotal = %f\n" NOCOLOR, i, dtotal);     
    }
  */
}

void build_window(Buffer<real> &W, real (*Wfunction)(idx n, idx N))
{
  idx N = W.size();
  for (idx n=0; n < N; ++n)
    W[n] = Wfunction(n,N);
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

  Options o("settings.cfg", Quit, 1);
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

  const int N_max = o.i("N_max");
  int N;
  Buffer<real> tmp_real_buffer_N_max(N_max); // For calculations in sub-functions but we must allocate the space already

  // Choose mic input files
  std::string x1_filepath = (argc == 4 ? argv[2] : o("x1_wav"));
  std::string x2_filepath = (argc == 4 ? argv[3] : o("x2_wav"));

  // Estimated and simulation (true) centroids 
  real calpha[N_max], true_alpha[N_max];
  real cdelta[N_max], true_delta[N_max];

  // Read simulation parameters
  std::ifstream sim; 
  sim.open("simulation.log");
  Guarantee(sim.is_open(), "Couldn't open simulation log!");
  sim >> N;
  printf(YELLOW "N=%d" NOCOLOR, N);
  for (uint i = 0; i < N; ++i)
    sim >> true_alpha[i] >> true_delta[i];
  // If N_max > N: Make the remaining true locations invisible by drawing over the same position of one of the active sources
  for (uint i = N; i < N_max; ++i)
    {
      true_alpha[i] = true_alpha[0];
      true_delta[i] = true_delta[0];
    }
  sim.close();

  // Write data for gnuplot real source positions overlay.
  std::ofstream sim_log;
  sim_log.open("s.dat");
  for (idx i=0; i < N; ++i)
    // The last column is required for splot (0 height suffices for the 2D hist, not the 3D, which should have the height at that point).
    sim_log << true_alpha[i] << " " << true_delta[i] << " 0\n\n";
  sim_log.close();



  SndfileHandle x1_file(x1_filepath), x2_file(x2_filepath);
  Guarantee( wav::ok(x1_file) && wav::ok(x2_file) , "Input file doesn't exist.");
  Guarantee(wav::mono(x1_file) && wav::mono(x2_file), "Input files must be mono.");

  const uint sample_rate_Hz = x1_file.samplerate();
  const idx  samples        = x1_file.frames(); 

  Buffer<real> x1_wav(samples), x2_wav(samples);
  x1_file.read(x1_wav(), samples);
  x2_file.read(x2_wav(), samples);

  // Only x1's are needed since that's the chosen channel for source separation
  Buffers<real> original_waves_x1(N, samples);
  for (int i = 0; i < N; ++i)
    {
      SndfileHandle wav_file("sounds/"+std::to_string(i)+"x0.wav");
      if (! wav::ok (wav_file))
	return EXIT_FAILURE;

      wav_file.read(original_waves_x1.raw(i), samples);
    }
  	
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
    delta(time_blocks, FFT_pN/2);

  // 2 sets of buffers [optional: +1] are needed to allow up to 50% overlapping. If writing and computing is done simultaneously instead of writing and waiting for the old  buffer that is freed at the next write_data call end an additional buffer is needed to store current computations.
  Buffers<real> 
    wav_out(N_max, time_blocks*FFT_slide, fftw_malloc, fftw_free), 
    bufs1(N_max,FFT_pN,fftw_malloc,fftw_free), bufs2(bufs1), bufs3(bufs1);
  Buffers<real> *old_buffers=NULL, *new_buffers=NULL;
  // Convenient interface to handle bufs pointers.
  DoubleLinkedList<Buffers<real>*> bufs;
  bufs.append(&bufs1); bufs.append(&bufs2); bufs.append(&bufs3);
  

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
  // Buffers for the axis of alpha and delta
  Buffer<real> alpha_range(hist_alpha.bins()), delta_range(hist_delta.bins()); 
  alpha_range.fill_range(o.d("alpha.min"), o.d("alpha.max"));
  delta_range.fill_range(o.d("delta.min"), o.d("delta.max"));

  hist.print_format();

  RankList<real, Point2D<real> > 
    preclusters(o.d("max_preclusters"),0.0,Point2D<real>()), old_preclusters(preclusters);
  RankList<real, Point2D<real> > cumulative_clusters(N_max,0.0,Point2D<real>());
  RankList<real, real> delta_preclusters(MAX_MARGINAL_PEAKS, 0.0), alpha_preclusters(delta_preclusters);

  //// Each of the clusters should now belong to a source: create masks and separate the sources.
  Buffer<int> masks(FFT_pN/2); 

  Gnuplot palpha,pdelta;


  _DUET.p = o.f("hist.p");
  _DUET.q = o.f("hist.q");
  _DUET.sigma_alpha = o.f("hist.sigma_alpha");
  _DUET.sigma_delta = o.f("hist.sigma_delta");
  _DUET.use_smoothing    = o.i("hist.use_smoothing")   ;
  _DUET.use_smoothing_2D = o.i("hist.use_smoothing_2D");

  const int RENDER = o.i("render");

  _DUET.aggregate_clusters = o.i("DUET.aggregate_clusters");
  _DUET.min_peak_fall   = o.d("DUET.min_peak_fall");
  _DUET.min_peak_dalpha = o.d("DUET.min_peak_dalpha");
  _DUET.min_peak_ddelta = o.d("DUET.min_peak_ddelta");
  _DUET.max_peak_scale_disparity = o.d("DUET.max_peak_scale_disparity");//10; // If smaller_peak * scale_disparity < max_peak it is rejected for being noise.

  _DUET.noise_threshold = o.d("DUET.noise_threshold");

  const DUETcfg DUET = _DUET; // Make every parameter constant to avoid mistakes


  /////////////////////////// TEST HISTOGRAMS /////////////////////////////////////////////7
  /*
    hist(.2,-.0001)+= 1;
    hist(-.2,-.0001)+=1;
    hist(0,-.0001)+=1;
  
    hist(.2,.0001) += 1;
    hist(-.2,.0001)+=1;

    hist(0,0)+=1;

    hist(-0.23,0.00002)+=0.8;

    hist(.3,0.00015)+=3;

    hist(-.4,0.00013)+=2;

    hist.fill_marginal_x(hist_alpha);
    hist.fill_marginal_y(hist_delta);
  
      
    static Buffer<real> 
    conv_kernel_alpha(hist_alpha.gen_gaussian_kernel(DUET.sigma_alpha)),
    conv_kernel_delta(hist_delta.gen_gaussian_kernel(DUET.sigma_delta)),
    conv_hist_alpha  (hist_alpha.bins()), 
    conv_hist_delta  (hist_delta.bins());

    static Matrix<real> 
    conv_kernel(hist.gen_gaussian_kernel(DUET.sigma_alpha, DUET.sigma_delta)),
    conv_hist  (hist.xbins(),hist.ybins());

    // New blurring method: convolution
    hist_alpha.kernel_convolution(conv_kernel_alpha, conv_hist_alpha);
    hist_delta.kernel_convolution(conv_kernel_delta, conv_hist_delta);
    hist.kernel_convolution(conv_kernel, conv_hist); 

    palpha.replot(alpha_range(), (*hist_alpha.raw())(),hist_alpha.bins(), "alpha");
    pdelta.replot(delta_range(), (*hist_delta.raw())(),hist_delta.bins(), "delta");	  

    hist.write_to_gnuplot_pm3d_data("hist.dat");
    RENDER_HIST("hist.dat", "Hist", 1); 


    while(1)
    {
    real Alpha,Delta;
    cout << "(alpha,delta) = ";
    cin >> Alpha >> Delta;

    hist(Alpha,Delta) += 1;
    hist_alpha(Alpha) += 1;
    hist_delta(Delta) += 1;

    palpha.replot(alpha_range(), (*hist_alpha.raw())(),hist_alpha.bins(), "alpha");
    pdelta.replot(delta_range(), (*hist_delta.raw())(),hist_delta.bins(), "delta");	  

    hist.write_to_gnuplot_pm3d_data("hist.dat");
    RENDER_HIST("hist.dat", "Hist", 1); 
    }
  */
  ///////////////////////////////////////////////////////////////////////////////////////////////


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

  if (RENDER >= 0)
    {
      puts("Calculating and writing histograms...");
      system("make cleanhists");
    }
  else
    {
      puts("Calculating histograms...");      
    }
  /*
    Gnuplot Wplot;
    Wplot.plot_y(W(),W.size(),"W");
    wait();
  */

  static Gnuplot pM1;
  static Buffer<real> M1(FFT_pN/2);
  static Histogram<real> M1hist(1,0,FFT_pN/2,HistogramBounds::Bounded);

  for (idx time_block = 0; time_block < time_blocks; ++time_block)
    {
      idx block_offset = time_block*FFT_slide;

      for (idx i = 0; i < FFT_N; ++i)
	{
	  idx offset_i = i+block_offset;

	  if (offset_i < samples)
	    {
	      if (DUET.use_window)
		{
		  x1[i] = x1_wav[offset_i] * W[i];
		  x2[i] = x2_wav[offset_i] * W[i]; 	
		}
	      else
		{
		  x1[i] = x1_wav[offset_i];
		  x2[i] = x2_wav[offset_i];
		}
	    }
	  else // end of file: fill with zeros
	    {
	      x1[i] = 0;
	      x2[i] = 0;
	    }
	}
      for (idx i=FFT_N; i < FFT_pN; ++i)
	{
	  // Just to make sure the padding region is clean but should only need to be done once.
	  x1[i] = 0;
	  x2[i] = 0;
	}
      /*
	Gnuplot x1_plot;
	x1_plot.plot_y(x1(),x1.size(), "x1");
	wait();
      */

      fftw_execute(xX1_plan);
      fftw_execute(xX2_plan);



      evenHC2magnitude(FFT_pN, X1(),M1());
      if (o.i("show_each_hist"))      
	pM1.plot(/*f_axis(),*/M1(),FFT_pN/2,"M1");
      for (idx f=0; f < FFT_pN/2; ++f)
	M1hist(f) += M1[f];

      /*
	Gnuplot Mplot;
	evenHC2magnitude(FFT_pN, X1(), x1());
	Mplot.plot_y(x1(),x1.size()/2,"M1");
	usleep(300000);
      */
      // Keep the record of X1 for all time for later audio reconstruction
      for (idx f = 0; f < FFT_pN; ++f)
	{
	  X1_history(time_block,f) = X1[f];
	  X2_history(time_block,f) = X2[f];
	}

      hist.clear();
      hist_alpha.clear();
      hist_delta.clear();
      calc_alpha_delta(time_block, FFT_pN, sample_rate_Hz, X1, X2, alpha, delta, hist, hist_alpha, hist_delta, DUET);
      //ransac_test(time_block, FFT_pN, sample_rate_Hz, X1, X2, alpha, delta, hist, hist_alpha, hist_delta, DUET);
      
      /*
	static Histogram2D<real> prod_hist(hist), diff_hist(hist);

	Gnuplot ph, po, pp, pd;
	prod_hist = hist;
	prod_hist *= old_hist;

	diff_hist = hist;
	diff_hist -= old_hist;
      */
      /*
	old_hist.plot(po, "Old");
	prod_hist.plot(pp, "Prod");
	diff_hist.plot(pd, "Diff");
      */
      //hist.plot(ph, "Hist");
      //wait();
      
      if (o.i("cc") >= 0)
	{
	  static CyclicCounter<int> cc(o.i("cc"));
	  cout << BLUE << cc.value() << NOCOLOR << endl;
	  ++cc;
	  old_hist += hist;
	  if (! cc.value())
	    {
	      old_hist.write_to_gnuplot_pm3d_data("old_hist.dat");
	      RENDER_HIST("old_hist.dat","Old", 1);

	      old_hist.clear();	  
	    }
	}

      //old_hist = hist;
      cumulative_hist += hist;
      	
      ///////// pre-Filter histogram clusters ///////////////////
		
      heuristic_clustering2D(hist, preclusters, DUET);
      heuristic_clustering(hist_alpha, alpha_preclusters, DUET, DUET.min_peak_dalpha);
      heuristic_clustering(hist_delta, delta_preclusters, DUET, DUET.min_peak_ddelta);

      cout << preclusters << alpha_preclusters << delta_preclusters << YELLOW "########\n" NOCOLOR;

      /*
	Buffer<Point2D<real> > clusters(preclusters.eff_size(DUET.noise_threshold));
	clusters.copy(preclusters.values(), clusters.size());

	// Write the clusters to the plot overlay
	std::ofstream clusters_dat;
	clusters_dat.open("s_duet.dat");
	for (idx i=0; i < clusters.size(); ++i)
	clusters_dat << clusters[i].x << " " << clusters[i].y << " 0\n\n";
	clusters_dat.close();
      */
      /*
	prod_hist.write_to_gnuplot_pm3d_data("prod_hist.dat");
	diff_hist.write_to_gnuplot_pm3d_data("diff_hist.dat");
	RENDER_HIST("prod_hist.dat", "Prod", 0);
	RENDER_HIST("diff_hist.dat", "Diff", 0);
      */


      ///////// Apply masks and rebuild current frame to audio and add it to the appropriate outputs
      if (! o.i("DUET.static_rebuild"))
	{
	  int N_clusters = preclusters.eff_size(DUET.noise_threshold); // clusters = preclusters.values
      		
	  old_buffers = bufs.read();
	  new_buffers = bufs.next();
	
	  build_masks(masks, alpha(time_block), delta(time_block), X1_history(time_block), X2_history(time_block), preclusters.values, N_clusters, FFT_pN, FFT_pN/2, FFT_df, tmp_real_buffer_N_max);
	  apply_masks(*new_buffers, alpha(time_block), X1_history(time_block), X2_history(time_block), masks, preclusters.values, N_clusters, FFT_pN, FFT_pN/2, FFT_df, Xxo_plan, Xo);
	
	
	  

	  write_data(wav_out, new_buffers, FFT_N, FFT_slide); // Explicitly use the initial region FFT_N and exclude the padding FFT_pN.
	}

      //static Buffer<real> hist_alpha(hist.xbins()), hist_delta(hist.ybins());

      if (o.i("show_each_hist"))
	{
	  palpha.replot(alpha_range(), (*hist_alpha.raw())(),hist_alpha.bins(), "alpha");
	  pdelta.replot(delta_range(), (*hist_delta.raw())(),hist_delta.bins(), "delta");	  

	  if (o.i("show_each_hist")>1)
	    {
	      hist.write_to_gnuplot_pm3d_data("hist.dat");
	      RENDER_HIST("hist.dat", "Hist", o.i("hist_pause")); 
	    }
	  else
	    if (o.i("hist_pause"))
	      wait();
	}

      ///////////////////////////////////////////////////////////

      if (RENDER >= 0)
	{
	  std::string filepath = "hist_dats/" + itosNdigits(time_block,10) + ".dat";
	  hist.write_to_gnuplot_pm3d_binary_data(filepath.c_str());
	  //system(("cp "+filepath+" tmp_dats/hist.dat && gen_movie.sh tmp_dats tmp_pngs 3D.gnut && feh tmp_pngs/hist.png").c_str());
	  //wait();	
	}
    }

	
  ///// Static Heuristic Rebuilding! ///////////////////////////////////////////////////////////////////
  puts(GREEN "Doing static-heuristic rebuilding..." NOCOLOR);
  cumulative_hist -= hist;
  cumulative_hist.write_to_gnuplot_pm3d_data("cumulative_hist.dat");
  

  heuristic_clustering2D(cumulative_hist, cumulative_clusters, DUET);

  cout << cumulative_clusters;

  bool found_clusters = (cumulative_clusters.eff_size(DUET.noise_threshold) ? 1:0);
  int N_clusters = cumulative_clusters.eff_size(DUET.noise_threshold);

  /*
  Buffer<Point2D<real> > clusters(cumulative_clusters.eff_size(DUET.noise_threshold)+(!found_clusters)); // if no clusters are found 0 size would blow the program
  clusters.copy(cumulative_clusters.values,clusters.size());
  */
  std::ofstream hist_cfg;
  hist_cfg.open("h.cfg");
  hist_cfg << hist.ybins();
  hist_cfg.close();


  // Write the clusters to the plot overlay
  std::ofstream clusters_dat;
  clusters_dat.open("s_duet.dat");
  for (idx i=0; i < N_clusters; ++i)
    clusters_dat << cumulative_clusters.values[i].x << " " << cumulative_clusters.values[i].y << " 0\n\n";
  clusters_dat.close();

  // Plot the 3D histogram with gnuplot and the simulation and DUET overlays
  // Since the "" must be passed with quotes inside the gnuplot command a triple \ is needed and  a single \ is needed for the outer command.
  RENDER_HIST("cumulative_hist.dat", "Cumulative hist", 1);
  
  static Gnuplot pM1hist;
  pM1hist.plot((*M1hist.raw())(),FFT_pN/2,"M1 histogram");      



  if (o.i("DUET.static_rebuild"))
    {

      // Build the masks and rebuild the signals
      for (idx t_block = 0; t_block < time_blocks; ++t_block)
	{
	  old_buffers = bufs.read();
	  new_buffers = bufs.next();
	  build_masks(masks, alpha(t_block), delta(t_block), X1_history(t_block), X2_history(t_block), cumulative_clusters.values, N_clusters, FFT_pN, FFT_pN/2, FFT_df, tmp_real_buffer_N_max);
      
	  apply_masks(*new_buffers, alpha(t_block), X1_history(t_block), X2_history(t_block), masks, cumulative_clusters.values, N_clusters, FFT_pN, FFT_pN/2, FFT_df, Xxo_plan, Xo);

	  write_data(wav_out, new_buffers, FFT_N, FFT_slide);	  // Explicitly use the initial region FFT_N and exclude the padding FFT_pN.
	  //      swap(bufs_ptr, bufs_ptr2);
	}		      
    }

  system("rm -f x*_rebuilt.wav");
  for (uint source = 0; source < N_clusters; ++source)
    {
      std::string wav_filepath("x"+itos(source)+"_rebuilt.wav");
      printf("%s...", wav_filepath.c_str());
      fflush(stdout);
      print_status( wav::write_mono(wav_filepath, wav_out.raw(source), samples, sample_rate_Hz) );
    }
	
  separation_stats(wav_out, original_waves_x1, N, samples);


  ///// End of Static Heuristic Rebuilding! ///////////////////////////////////////////////////////////////////

  fftw_destroy_plan(xX1_plan); 
  fftw_destroy_plan(xX2_plan);
  fftw_destroy_plan(Xxo_plan);
	


  if (RENDER > 0)
    Guarantee0( system("make render") , "Couldn't generate the movies.");
  cout << "#Clusters = " << N_clusters <<"\n";
  cumulative_clusters.print(N_clusters);
  system("cat s.dat");
  puts("\nSuccess!");

  if (WAIT)
    wait();

  return 0;
}






/////////////

