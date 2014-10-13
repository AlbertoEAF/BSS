#include "duet.h"

// Stream stats. 
int ACQUIRED_STREAMS = 0;  // How many streams were acquired during the whole program execution.
int HIGHEST_STREAM_ID = 0; // The highest stream id ever achieved.
int MERGED_STREAMS = 0;    // How many streams were merged (permanently deleted and whole information cleared -- this leaves wholes behind but stats will give nan values for the holes if left unfilled)

//#define OLD_MASK_BUILD
//#define OLD_PEAK_ASSIGN

const int N_EXPORT_DIGITS = 5; // Number of digits to output in the output serial number.



/// Returns the score for the DUET histogram based on the parameters p and q
real DUEThist_score(real x1re, real x1im, real x2re, real x2im, real omega, real p, real q)
{
  real s_re, s_im, s_abs;
  complex_multiply(x1re,x1im, x2re,x2im, &s_re,&s_im);

  s_abs = abs(s_re,s_im);

  // PERFORMANCE: The tables of the powers of omega could be reused.
  return std::pow(s_abs,p)*std::pow(omega,q); 
}

/** Taken from bessel.c, also distributed in this folder. Calculates the modified Bessel function I0. */
/*
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
*/

/**
   @param[in] K - Number of active sources: theta.size() >= K
   @param[in] x - Must have fewer than RAND_MAX elements.
*/
/*
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
*/


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
      for (idx f = DUET.Fmin; f < pN/2 - 1; ++f)
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
      for (idx f = DUET.Fmin; f < DUET.Fmax; ++f)
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
}

/*
void ransac_test(idx time_block, idx pN, idx sample_rate_Hz,
		 Buffer<real> &X1, Buffer<real> &X2,
		 Matrix<real,MatrixAlloc::Rows> &alpha, 
		 Matrix<real,MatrixAlloc::Rows> &delta, 
		 Histogram2D<real> &hist, 
		 Histogram<real> &hist_alpha, Histogram<real> &hist_delta, 
		 const DUETcfg &DUET)
{
  static real df = sample_rate_Hz/(real)pN;

  // f = 0 Hz:
  // d_Re = X1(0) / X2(0);
  // d_Im = 0;
  
  real a = X2[0] / X1[0];
  alpha(time_block, 0) = a - 1/a;
  delta(time_block, 0) = 0.0;

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
      _alpha =alpha(time_block, f) = a - 1/a;
      // Wrong

      //_delta = delta(time_block, f) = std::fmod(std::arg(F1) - std::arg(F2), M_PI);///omega;
      //_delta = delta(time_block, f) = std::arg(std::polar<real>(1,std::arg(F1) - std::arg(F2)));///omega;
      _delta = std::fmod(std::arg(F1)-std::arg(F2) + M_PI, 2*M_PI) - M_PI;

      //      _delta = std::fmod(std::arg(F1)-std::arg(F2),M_PI);
      f_axis[f] = f;
      delta_axis[f] = _delta;

      //DUET_hist_add_score(hist, hist_alpha, hist_delta, _alpha, _delta, X1[f],X1[fI], X2[f],X2[fI], omega, DUET);
    }
  

  pransac.replot(delta_axis(), f_axis(), pN/2, "Frame RANSAC");
  //wait();
}
*/


#define RELEASE(x) {}


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
void build_single_masks(Buffer<int> &masks, real *alpha, real *delta, real *X1, real *X2, const Buffer<Point2D<real> > &clusters, int N_clusters, idx FFT_N, real FFT_df, Buffer<real> &tmp, const DUETcfg &DUET)
{
  ///// ATTENTION IF YOU WANT TO MODIFY THIS CODE !! PERFORMANCE
  // This code is optimized for binwise single mask assignment. For multiple assignment use build_multi_masks instead.
   
  #ifdef OLD_MASK_BUILD
  static Buffer<int> old_masks(masks);
  idx masks_diffs = 0;
  #endif

  for (idx f = DUET.Fmin; f < DUET.Fmax; ++f)
    {
      real omega = _2Pi * f * FFT_df;
      idx f_im = FFT_N - f;
      
      for (int n=0; n < N_clusters; ++n)
	{
	  real a_n = alpha2a(clusters[n].x);
	  real delta_n = clusters[n].y;

	  tmp[n] = std::norm(a_n*std::polar<real>(1,-delta_n*omega) * std::complex<real>(X1[f],X1[f_im]) - std::complex<real>(X2[f],X2[f_im])) / (1.0 + a_n*a_n);
	}
      masks[f] = array_ops::min_index(tmp(), N_clusters);

      #ifdef OLD_MASK_BUILD
      old_masks[f] = closest_cluster(Point2D<real>(alpha[f],delta[f]), clusters);

      if (masks[f]!=old_masks[f])
	masks_diffs += 1;
      #endif
    }
#ifdef OLD_MASK_BUILD
  masks = old_masks;
  cout << "#Mask diffs = " << masks_diffs << endl;
  cout << RED << masks_diffs << NOCOLOR << endl;
#endif // OLD_MASK_BUILD
}


/// Mask bins can only belong to a source thus it is more efficient to have Buffer<int> where a negative value means the bin has no owning source and the bin value is the number of the source.
void apply_single_masks(Buffers<real> &buffers, real *X1, real *X2, const Buffer<int> &masks, Buffer<Point2D<real> > &clusters, unsigned int active_sources, idx FFT_N, real FFT_df, fftw_plan &FFTi_plan, Buffer<real> &Xo, const DUETcfg &DUET)
{
  buffers.clear();
  
  // Rebuild one source per iteration to reuse the FFT plan (only 1 needed).
  for (int source = 0; source < (int)active_sources; ++source)
    {
      Xo.clear();

      if (masks[0] == source)
	{
	  real a_k = alpha2a(clusters[source].x);
	  Xo[0] = a_k*X1[0]-X2[0];
	  Xo[0] *= Xo[0] / (1 + a_k*a_k);
	}
      
      for (int f = DUET.Fmin; f < DUET.Fmax; ++f)
	{
	  if (masks[f] == source)
	    {
	      int  f_im = FFT_N - f;
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

/// Masks bins can belong to multiple owners thus Buffers<bool> is used.
void apply_multi_masks(Buffers<real> &buffers, real *X1, real *X2, Buffers<bool> &masks, Buffer<Point2D<real> > &clusters, unsigned int active_sources, idx FFT_N, real FFT_df, fftw_plan &FFTi_plan, Buffer<real> &Xo, const DUETcfg &DUET)
{
  buffers.clear();
  
  // Rebuild one source per iteration to reuse the FFT plan (only 1 needed).
  for (int source = 0; source < (int)active_sources; ++source)
    {
      Xo.clear();

      bool *mask = masks.raw(source);

      if (mask[0])
	{
	  real a_k = alpha2a(clusters[source].x);
	  Xo[0] = a_k*X1[0]-X2[0];
	  Xo[0] *= Xo[0] / (1 + a_k*a_k);
	}
      
      for (int f = DUET.Fmin; f < DUET.Fmax; ++f)
	{
	  if (mask[f])
	    {
	      int  f_im = FFT_N - f;
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


void LDMB2C(StreamSet &Streams, IdList &active_streams, Buffers<real> *new_buffers, Buffer<Point2D<real> > &clusters_pos, int N_clusters, idx time_block, Buffer<real> &W, Buffer<int> &C, const DUETcfg &DUET)
{
  static const int MAX_CLUSTERS = DUET.max_clusters;
  static const int MAX_ACTIVE_STREAMS = DUET.max_active_streams;
  static const int FFT_N = DUET.FFT_N;
  static const int FFT_slide = DUET.FFT_slide;
  static const int FFT_overlap = FFT_N-FFT_slide;
  static const int MAX_SILENCE_BLOCKS = DUET.max_silence_blocks;
  static const int MIN_ACTIVE_BLOCKS = DUET.min_active_blocks;

  //// Solve the permutations to achieve continuity of the active streams. ///////////
  static Buffer<real> dist_k  (MAX_CLUSTERS, FLT_MAX );
  static Buffer<real> acorr_k (MAX_CLUSTERS, -FLT_MAX);

  static Matrix<real> 
    D (MAX_ACTIVE_STREAMS, MAX_CLUSTERS,  FLT_MAX), 
    A0(MAX_ACTIVE_STREAMS, MAX_CLUSTERS, -FLT_MAX);
	  
  static IdList 
    merging_streams  (MAX_ACTIVE_STREAMS), 
    assigned_clusters(MAX_CLUSTERS);
	  
  // LDB 
	  
  C.clear(); D.clear(); A0.clear();
  merging_streams.clear();
  assigned_clusters.clear();

  // Calculate the distance between old streams and new buffers in terms of D and A0. A0 requires applying the complementary window.
  for (int s=0; s < (int)active_streams.N(); ++s)
    {
      static Buffer<real> W_buf_stream(FFT_overlap), W_buf_new_stream(FFT_overlap);

      int id = active_streams[s];
      W_buf_stream.copy(Streams.last_buf_raw(id,FFT_slide), FFT_overlap);
      for (int u=0; u < FFT_overlap; ++u)
	W_buf_stream[u] *= W[u]; // Apply the complementary window of the next block.

      for (int j=0; j < N_clusters; ++j)
	{
	  // D
	  D (s,j) = Lambda_distance(Streams.pos(id),clusters_pos[j]);

	  // A0 (with complementary windows applied)
	  // Since streams haven't been assigned yet, streams assigned right at the last block have a difference of 1.
	  if (time_block - Streams.last_active_time_block(id) == 1)
	    {
	      // PERFORMANCE: We could calculate them all only once beforehand since they're reutilized for each old stream.
	      W_buf_new_stream.copy(new_buffers->raw(j), FFT_overlap); 
	      for (int u=0; u < FFT_overlap; ++u)
		W_buf_new_stream[u] *= W[u+FFT_slide]; // Apply the past complementary window.
	      // PERFORMANCE: Can do the normalization outside manually.
	      A0(s,j) = array_ops::a0(W_buf_stream(), W_buf_new_stream(), FFT_N-FFT_slide);
	    }
	}
    }
	  
  for (int s=0; s < (int)active_streams.N(); ++s)
    Streams.print(active_streams[s]);
  printf("Active_Streams=%u clusters = %u\n", active_streams.N(), N_clusters);
  puts("D:");
  D.print (active_streams.N(), N_clusters);
  puts("A0:");
  A0.print(active_streams.N(), N_clusters);

	  
  // Life
  if (active_streams.N() && N_clusters)
    {
      // Life Stage 1 : global A0 > A0MIN assignment
      printf(GREEN "Streams that live by A0 (stream@cluster): ");
      while( 1 )
	{
	  static size_t s, j; // no need to init at 0 every turn.
	  A0.max_index(s,j, active_streams.N(), N_clusters);

	  real a0(A0(s,j));
	      
	  if ( a0 > DUET.a0min ) // Life
	    {
	      int id(active_streams[s]);
	  
	      assigned_clusters.add(j);

	      C[j] = id;

	      /* Remove entries that are no longer candidates for assignment from lookup
		 (stream id=active_streams[s] and cluster j). */

	      // Eliminate the already processed stream s from lookup.
	      if (! DUET.multiple_assign)
		{
		  A0.fill_row_with(s, -FLT_MAX);
		  D .fill_row_with(s,  FLT_MAX);
		}

	      // Eliminate the already processed cluster j from lookup.
	      A0.fill_col_with(j, -FLT_MAX);
	      D .fill_col_with(j,  FLT_MAX);

	      printf("%d@%lu ", id, j);
	    }
	  else
	    break; // no more a0 > A0MIN
	}
      puts("\n" NOCOLOR);
      // Life Stage 2 : global pos assignment (Lambda_distance < threshold) 
      printf(GREEN "Streams that live by pos (stream@cluster): ");
      while( 1 )
	{	      
	  static size_t s, j; // no need to init at 0 every turn.
	  D.min_index(s,j, active_streams.N(), N_clusters);

	  real d(D(s,j));

	  if ( d < DUET.max_Lambda_distance ) // Life
	    {
	      int id(active_streams[s]);
		  
	      C[j] = id;

	      assigned_clusters.add(j);		  

	      // Eliminate the already processed stream s from lookup.
	      if (! DUET.multiple_assign)
		{
		  A0.fill_row_with(s, -FLT_MAX);
		  D .fill_row_with(s,  FLT_MAX);
		}

	      // Eliminate the already processed cluster j from lookup.
	      A0.fill_col_with(j, -FLT_MAX);
	      D .fill_col_with(j,  FLT_MAX);

	      printf("%d@%lu ", id, j);
	    }
	  else
	    break; // no more a0 > A0MIN	      
	}
      puts("\n" NOCOLOR);
    }
  // Death
  for (int s = 0; s < (int)active_streams.N(); ++s)
    {
      int id = active_streams[s];
      // The difference is zero for freshly active streams (right at the previous block).
      int stream_inactive_blocks = time_block - Streams.last_active_time_block(id) - 1;
      if (stream_inactive_blocks > MAX_SILENCE_BLOCKS)
	{
	  active_streams.del(id);

	  // Add stream to the list of streams to merge. Note we won't merge to other streams in the same condition but only streams that remain active.
	  if ( Streams.active_blocks(id) <= (const unsigned int)MIN_ACTIVE_BLOCKS )
	    merging_streams.add(id);
	  else
	    printf(RED "Stream id %d has died.\n" NOCOLOR, id);
	}
      else if (stream_inactive_blocks)
	printf(GREEN "Stream %d remains inactive by %d/%d time_blocks.\n" NOCOLOR, id, stream_inactive_blocks, MAX_SILENCE_BLOCKS);
    }

  // Merge
  // Merge to the closest active stream. Note that this is the only stage at which more than one stream can be assigned to a destination stream (merging procedure).
  if (active_streams.N())
    {
      for (long int m = 0; m < (long int)merging_streams.N(); ++m )
	{
	  int m_id = merging_streams[m];
	      
	  // Find the closest active stream to stream id=m_id.
	  real min_distance = FLT_MAX;
	  int  s_id_match   = active_streams[0];
	  for (int s = 0; s < (int)active_streams.N(); ++s)
	    {
	      int s_id = active_streams[s];

	      real distance = Lambda_distance(Streams.pos(m_id), Streams.pos(s_id));
	      if (distance < min_distance)
		{
		  min_distance = distance;
		  s_id_match = s_id;
		}
	    }

	  ++MERGED_STREAMS;

		  
	  // PERFORMANCE: This can be implemented more efficiently by adding only the active portion.
	  Streams.stream(s_id_match)->add_at(*(Streams.stream(m_id)),0);
		   
	  // Now we can free up this stream since it has been merged.
	  Streams.release_id(m_id); 

	  printf(GREEN "Stream %d merged to %d.\n" NOCOLOR, m_id, s_id_match);
	}
    }

  // Birth
  // For each cluster left unassigned a new stream is born.
  printf(GREEN "Born streams: ");
  for (int j = 0; j < N_clusters; ++j)
    {
      if (! assigned_clusters.has(j)) 
	{
	  int id = Streams.acquire_id();
	  active_streams.add(id);
	  
	  if (id > HIGHEST_STREAM_ID)
	    HIGHEST_STREAM_ID = id;
	  ++ACQUIRED_STREAMS;

	  C[j] = id;

	  printf("%d ", id);
	}
    }
  puts("\n" NOCOLOR);
} // End of LDMB



    

// Calculated to 4sigma (3 would suffice, as the generated gaussian kernels).
// WARN: it sorts the peaks.
real confidence(Histogram<real> &H, Buffer<real> &peaks_k, int K, real sigma)
{
  Buffer<real> *h = H.raw();

  // Sort the peaks to process partial peak overlap integration gracefully.
  std::sort(peaks_k(), peaks_k()+peaks_k.size());

  // Not multiplying by H.dx() since the quotient will erase it.
  real H_integral = h->sum(), peaks_integral=0; 

  size_t prev_k_max_bin=0, bin=0, max_k_bin=0;
  for (int k=0; k < K; ++k)
    {
      // Get the minimum and maximum bins for peak_k for integration (integration limits)
      H.get_bin_index(peaks_k[k]-4.0*sigma, bin      );
      H.get_bin_index(peaks_k[k]+4.0*sigma, max_k_bin);

      // If peak integration regions overlap do not integrate twice. Resume where we left peak k-1.
      if (bin <= prev_k_max_bin)
	bin = prev_k_max_bin+1;

      for (; bin <= max_k_bin; ++bin)
	peaks_integral += (*h)[bin];

      prev_k_max_bin = max_k_bin;
    }
  
  return peaks_integral / H_integral;
}

void draw_trajectories(StreamSet &streams, unsigned int time_blocks, real slide_time, Gnuplot &pTalpha, Gnuplot &pTdelta, Gnuplot &pT)
{
  pTalpha.reset(); pTdelta.reset(); pT.reset();

  pTalpha.set_labels("Localization step", "alpha");
  pTdelta.set_labels("Localization step", "delta (s)");
  pT     .set_labels("alpha"     , "delta");

  Buffer<real> times(time_blocks), alphas(time_blocks), deltas(time_blocks);
  times.fill_range(0, slide_time*(time_blocks-1));

  for (int id=1; id <= HIGHEST_STREAM_ID; ++id)
    {
      unsigned int start  = streams.first_active_time_block(id);
      unsigned int stop   = streams.last_active_time_block(id);
      unsigned int blocks = stop - start;

      if (blocks)
	{
	  Buffer<Point2D<real> > &path = *streams.trajectory(id);

	  for (unsigned int tb = start; tb < stop; ++tb)
	    {
	      alphas[tb] = path[tb].x;
	      deltas[tb] = path[tb].y;
	    }
  
	  std::string title = std::to_string(id);

	  pTalpha.plot(&times [start], &alphas[start], blocks, title.c_str());
	  pTdelta.plot(&times [start], &deltas[start], blocks, title.c_str());
	  pT     .plot(&alphas[start], &deltas[start], blocks, title.c_str());

	  printf("Stream %d: alpha=[%g,%g] delta=[%g,%g]\n", id, 
		 array_ops::min(&alphas[start],blocks),
		 array_ops::max(&alphas[start],blocks),
		 array_ops::min(&deltas[start],blocks),
		 array_ops::max(&deltas[start],blocks));
	}
    }
}

real local_confidence(Histogram2D<real> &H, Matrix<real> &kernel, Point2D<real> &pos)
{
  // Calculate the error from the Gaussian profile to the local histogram profile
  real error = 0;
  
  size_t kx_bins = kernel.rows();
  size_t ky_bins = kernel.cols();
  size_t kcx_bin = kx_bins/2;
  size_t kcy_bin = ky_bins/2;

  size_t Hcx_bin, Hcy_bin;
  H.get_bin_index(pos.x, pos.y, Hcx_bin, Hcy_bin);

  real factor = H(pos.x, pos.y) / kernel(kcx_bin, kcy_bin);

  // No need to sum the central bin as it has beem scaled to have 0 error.
  for (size_t i=1; i <= kcx_bin; ++i)
    for (size_t j=1; j <= kcy_bin; ++j)
      {
	real e1=0, e2=0, e3=0, e4=0;

	// PERFORMANCE: TESTS CAN BE MERGED 

	if (i<=Hcx_bin && j<= Hcy_bin)
	  e1 = (H.bin(Hcx_bin-i,Hcy_bin-j) - factor * kernel(kcx_bin-i, kcy_bin-j));

	if (Hcx_bin+i<H.xbins())
	  e2 = (H.bin(Hcx_bin+i,Hcy_bin-j) - factor * kernel(kcx_bin+i, kcy_bin-j));

	if (i<=Hcx_bin && Hcy_bin+j<H.ybins())
	  e3 = (H.bin(Hcx_bin-i,Hcy_bin+j) - factor * kernel(kcx_bin-i, kcy_bin+j));

	if (Hcx_bin+i<H.xbins() && Hcy_bin+j<=H.ybins())
	  e4 = (H.bin(Hcx_bin+i,Hcy_bin+j) - factor * kernel(kcx_bin+i, kcy_bin+j));

	error += e1*e1 + e2*e2 + e3*e3 + e4*e4;
      }

  return 1/error;
}

bool excess_kurtosis(real &kurtosis, Histogram<real> &H, Buffer<real> &kernel, real pos)
{
  const int kcbin = kernel.size()/2;

  size_t cbin;
  if ( ! H.get_bin_index(pos, cbin) )
    return false;

  real mu4=0, mu2=0;

  const real dx = H.dx();
  
  // Add the centrum which is not added up in the loops.
  real Hsum = H.bin(cbin);

  // The center bin won't be summed. 
  // Since the kernel is symmetric we run left (r) and right (r) terms inside the same loop iteration.
  // Starts at the smoothing kernel edges. i= distance to centre of the smoothing kernel from the left
  for (size_t i=kcbin; i; --i)
    {
      real D = dx*i;

      real Hl = ( cbin>=i         ? H.bin(cbin-i) : 0 );
      real Hr = ( cbin+i<H.bins() ? H.bin(cbin+i) : 0 );
      real Hs = Hl + Hr;

      mu4 += std::pow(D, 4) * Hs;

      mu2 += D*D * Hs;
      
      Hsum += Hs;
    }

  kurtosis = mu4/(mu2*mu2) * Hsum - 3.0;

  return true;
}

bool excess_kurtosis(real &kurtosis_x, real &kurtosis_y, Histogram2D<real> &H, Matrix<real> &kernel, real pos_x, real pos_y)
{
  int kcxbin = kernel.rows()/2;
  int kcybin = kernel.cols()/2;

  size_t cxbin, cybin;
  if ( ! H.get_bin_index(pos_x,pos_y, cxbin, cybin) )
    return false;

  real mu4x=0, mu4y=0, mu2x=0, mu2y=0;

  const real dx = H.dx(), dy = H.dy();
  
  // Add the centrum which is not added up in the loops.
  real Hxsum(H.bin(cxbin,cybin)), Hysum(Hxsum);

  // The center bin won't be summed. 
  // Since the kernel is symmetric we run left (r) and right (r) terms inside the same loop iteration.
  // Starts at the smoothing kernel edges. i= distance to centre of the smoothing kernel from the left
  for (size_t i=kcxbin; i; --i)
    {
      real D = dx*i;

      real Hl = ( cxbin>=i          ? H.bin(cxbin-i, cybin) : 0 );
      real Hr = ( cxbin+i<H.xbins() ? H.bin(cxbin+i, cybin) : 0 );
      real Hs = Hl + Hr;

      mu4x += std::pow(D, 4) * Hs;

      mu2x += D*D * Hs;
      
      Hxsum += Hs;
    }

  for (size_t i=kcybin; i; --i)
    {
      real D = dy*i;
      
      real Hl = ( cybin>=i           ? H.bin(cxbin, cybin-i) : 0 );
      real Hr = ( cybin+i<H.ybins() ? H.bin(cxbin, cybin+i) : 0 );
      real Hs = Hl + Hr;

      mu4y += std::pow(D, 4) * Hs;

      mu2y += D*D * Hs;
      
      Hysum += Hs;
    }

  kurtosis_x = mu4x/(mu2x*mu2x) * Hxsum - 3.0;
  kurtosis_y = mu4y/(mu2y*mu2y) * Hysum - 3.0;  

  return true;
}


void constrained_excess_kurtosis(real &kurtosis_x, real &kurtosis_y, Histogram2D<real> &H, Matrix<real> &kernel, real pos_x, real pos_y, const DUETcfg &DUET)
{
  int kcxbin = kernel.rows()/2;
  int kcybin = kernel.cols()/2;

  size_t cxbin, cybin;
  H.get_bin_index(pos_x,pos_y, cxbin, cybin);

  real mu4x=0, mu4y=0, mu2x=0, mu2y=0;

  const real dx = H.dx(), dy = H.dy();
  
  // Normalization constants
  // Add the centrum which is not added up in the loops.
  real Hxsum(H.bin(cxbin,cybin)), Hysum(Hxsum);

  // The center bin won't be summed. 
  // Since the kernel is symmetric we run left (r) and right (r) terms inside the same loop iteration.
  // Starts at the smoothing kernel edges. i= distance to centre of the smoothing kernel from the left
  for (size_t i=kcxbin; i; --i)
    {
      real D = dx*i;

      real Hl = ( cxbin>=i          ? H.bin(cxbin-i, cybin) : 0 );
      real Hr = ( cxbin+i<H.xbins() ? H.bin(cxbin+i, cybin) : 0 );
      real Hs = Hl + Hr;

      mu4x += std::pow(D, 4) * Hs;

      mu2x += D*D * Hs;
      
      Hxsum += Hs;
    }

  for (size_t i=kcybin; i; --i)
    {
      real D = dy*i;

      real Hl = ( cybin>=i           ? H.bin(cxbin, cybin-i) : 0 );
      real Hr = ( cybin+i<=H.ybins() ? H.bin(cxbin, cybin+i) : 0 );
      real Hs = Hl + Hr;

      mu4y += std::pow(D, 4) * Hs;

      mu2y += D*D * Hs;
      
      Hysum += Hs;
    }

  

  printf(CYAN "o%g e%g reo%g     o%g e%g reo%g\n" NOCOLOR, 
	 DUET.sigma_alpha, mu2x/Hxsum, (mu2x/Hxsum)/DUET.sigma_alpha,
	 DUET.sigma_delta, mu2y/Hysum, (mu2y/Hysum)/DUET.sigma_delta);

  kurtosis_x = mu4x/(DUET.sigma_alpha*DUET.sigma_alpha) / Hxsum - 3.0;
  kurtosis_y = mu4y/(DUET.sigma_delta*DUET.sigma_delta) / Hysum - 3.0;  
}

int main(int argc, char **argv)
{

  /* Name convention throughout this file:
  	
     i - input
     o - output
     m - magnitude

     and capital letters for the frequency domain
  */	

  OptionParser opt;

  opt.setFlag("help",'h');
  opt.setOption("x1");
  opt.setOption("x2");
  opt.setOption("FFT_N",'N');
  opt.setOption("FFT_slide",'s');
  opt.setOption("window",'w');
  

  opt.setOption("log", 'l');
  opt.setOption("ibm_log", 'i');

  opt.setFlag('p');

  int arg0 = opt.parse(argc,argv);
  if (arg0 == argc || opt.getFlag("help"))
    {
      printf("Usage:\n\tPrgm [-x1 x] [-x2 x] [-FFT_N N] [-w/--window type] file.duet\n\n");
      exit(1);
    }
 
  std::string logpath = opt.getOption("log");
  std::string ibm_logpath = opt.getOption("ibm_log");

  

  Options o(argv[arg0], Quit, 0);

  DUETcfg _DUET; // Just to initialize, then a const DUET is initialized from this one.

  // Stream behaviour
  const int  MAX_CLUSTERS       = o.i("max_clusters");
  const int  MAX_ACTIVE_STREAMS = o.i("max_active_streams");
  const int  MIN_ACTIVE_BLOCKS  = o.i("min_active_blocks");
  const real A0MIN              = o.f("a0min");

  const bool STATIC_REBUILD = o.i("DUET.static_rebuild");

  _DUET.max_clusters        = MAX_CLUSTERS;
  _DUET.max_active_streams  = MAX_ACTIVE_STREAMS;
  _DUET.min_active_blocks   = MIN_ACTIVE_BLOCKS;
  _DUET.a0min               = A0MIN;
  _DUET.max_Lambda_distance = o.f("max_Lambda_distance");

  _DUET.multiple_assign = o.i("multicluster_assign");

  int N_accum = o.i("N_accum_frames"); // how many frames should be accumulated.

  //  int WAIT = o.i("wait");

  fftw_plan xX1_plan, xX2_plan, Xxo_plan;
  int FFT_flags;

  const int N_max = o.i("N_max");
  int N;
  Buffer<real> tmp_real_buffer_N_max(N_max,0,fftw_malloc,fftw_free); // For calculations in sub-functions but we must allocate the space already

  // Choose mic input files
  std::string x1_filepath = (opt.Option("x1") ? opt.getOption("x1") : o("x1_wav"));
  std::string x2_filepath = (opt.Option("x1") ? opt.getOption("x1") : o("x2_wav"));

  // simulation (true) centroids 
  real true_alpha[N_max];
  real true_delta[N_max];

  // Read simulation parameters
  std::ifstream sim; 
  sim.open("simulation.log");
  Guarantee(sim.is_open(), "Couldn't open simulation log!");
  sim >> N;
  printf(YELLOW "N=%d" NOCOLOR, N);
  for (int i = 0; i < N; ++i)
    sim >> true_alpha[i] >> true_delta[i];
  // If N_max > N: Make the remaining true locations invisible by drawing over the same position of one of the active sources
  for (int i = N; i < N_max; ++i)
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
  const idx  samples_input  = x1_file.frames(); 
  const real Ts             = 1.0/(real)sample_rate_Hz;
  	
  printf("\nProcessing input file with %lu frames @ %u Hz.\n\n", 
	 samples_input, sample_rate_Hz);	
	
  const idx FFT_N = (opt.Option("FFT_N") ? 
		     (idx)strtol(opt.getOption("FFT_N").c_str(), NULL, 10) : o.i("FFT_N"));
  _DUET.FFT_N = FFT_N;
  Guarantee0(FFT_N % 2, "System implemented for FFTs with even size.");
 
  _DUET.FFT_slide_percentage = o.i("FFT_slide_percentage", Warn);
  if (! _DUET.FFT_slide_percentage)
    _DUET.FFT_slide_percentage = 100;

  _DUET.FFT_slide = FFT_N * (_DUET.FFT_slide_percentage/100.);
  Guarantee(_DUET.FFT_slide <= _DUET.FFT_N, "FFT_slide(%ld) > FFT_N(%ld)", _DUET.FFT_slide, _DUET.FFT_N);

  if (opt.Option("FFT_slide"))
    _DUET.FFT_slide = std::stol(opt.getOption("FFT_slide"));

  printf(YELLOW "FFT_N = %ld\n" "FFT_slide = %ld (%ld%%)\n" NOCOLOR, FFT_N, _DUET.FFT_slide, _DUET.FFT_slide_percentage);



  const idx FFT_slide = _DUET.FFT_slide;

  const unsigned int MAX_SILENCE_BLOCKS = std::ceil(o.f("stream_max_inactive_time_s")/((real)FFT_slide*Ts));
  _DUET.max_silence_blocks = MAX_SILENCE_BLOCKS;

  // This will require triple-buffering
  //  Guarantee(FFT_slide >= FFT_N/2, "FFT_slide(%ld) > FFT_N/2(%ld)", FFT_slide, FFT_N/2);

  _DUET.use_window = 1;
    
  // Frequency oversampling
  _DUET.FFT_p = o.i("FFT_oversampling_factor");
  _DUET.FFT_pN = _DUET.FFT_p * _DUET.FFT_N;
  const idx FFT_pN = _DUET.FFT_pN;

  const idx time_blocks = div_up(samples_input-FFT_N,FFT_slide) + 1;
  // If it's samples_input % block != 0, samples_all>samples_input
  const size_t samples_all = time_blocks*FFT_slide+(FFT_N-FFT_slide);

  Buffer<real> x1_wav(samples_all,0,fftw_malloc,fftw_free), x2_wav(x1_wav);
  x1_file.read(x1_wav(), samples_input);
  x2_file.read(x2_wav(), samples_input);


  // Only x1's are needed since that's the chosen channel for source separation
  Buffers<real> original_waves_x1(N, samples_all,0,fftw_malloc,fftw_free);
  for (int i = 0; i < N; ++i)
    {
      SndfileHandle wav_file("sounds/s"+std::to_string(i)+"x0.wav");
      if (! wav::ok (wav_file))
	return EXIT_FAILURE;

      wav_file.read(original_waves_x1.raw(i), samples_input);
    }

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
    wav_out(N_max, samples_all, fftw_malloc, fftw_free), 
    ibm_out(N    , samples_all, fftw_malloc, fftw_free), 
    bufs1(N_max,FFT_pN,fftw_malloc,fftw_free), bufs2(bufs1), bufs3(bufs1);
  Buffers<real> *old_buffers=NULL, *new_buffers=NULL;
  // Convenient interface to handle bufs pointers.
  DoubleLinkedList<Buffers<real>*> bufs;
  bufs.append(&bufs1); bufs.append(&bufs2); bufs.append(&bufs3);
  
  StreamSet Streams(o.i("streams"), time_blocks*FFT_slide+(FFT_N-FFT_slide), FFT_pN/2, time_blocks);
  



  const real FFT_df = sample_rate_Hz / (real) FFT_pN;

  /*
  Buffer<real> Omega(FFT_N/2);
  for (size_t k=0; k<FFT_N/2;++k)
    Omega[k] = _2Pi*k*FFT_df;
  */

  _DUET.Fmax = std::min<int>(o.f("DUET.high_cutoff_Hz")/FFT_df, 
			     int(FFT_pN/2));
  if (_DUET.Fmax <= 2)
    _DUET.Fmax = FFT_pN/2;
  _DUET.Fmin = std::min<int>(o.f("DUET.low_cutoff_Hz")/FFT_df, 
			     _DUET.Fmax-1);
  if (_DUET.Fmin <= 1)
    _DUET.Fmin = 1;


  FFT_flags = FFTW_ESTIMATE; // Use wisdom + FFTW_EXHAUSTIVE later!

  Buffer<real> f_axis(FFT_pN/2);
  f_axis.fill_range(0,(real)sample_rate_Hz/2.0);

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
    hist_delta(o.d("hist.ddelta"), o.d("delta.min"), o.d("delta.max"), hist_bound_type),
    chist_alpha(hist_alpha), chist_delta(hist_delta); // Short-time cumulative histograms
  Histogram2D<real> cumulative_hist(hist), chist(hist), old_hist(hist);
  // Buffers for the axis of alpha and delta
  Buffer<real> alpha_range(hist_alpha.bins()), delta_range(hist_delta.bins()); 
  alpha_range.fill_range(o.d("alpha.min"), o.d("alpha.max"));
  delta_range.fill_range(o.d("delta.min"), o.d("delta.max"));

  hist.print_format();

  RankList<real, Point2D<real> > 
    clusters(o.d("max_clusters"),0.0,Point2D<real>()), 
    cumulative_clusters(N_max,0.0,Point2D<real>());
  RankList<real, real> 
    delta_clusters(o.d("max_clusters"), 0.0), // Bigger than neeeded, less peaks arise in the marginals (combinations of clusters alpha,delta -> 2D clusters).
    alpha_clusters(delta_clusters);

  //// Each of the clusters should now belong to a source: create masks and separate the sources.
  Buffer<int> masks(FFT_pN/2,-1); // By default no source is assigned (-1). 

  Gnuplot palpha,pdelta;


  _DUET.p = o.f("hist.p");
  _DUET.q = o.f("hist.q");
  _DUET.sigma_alpha = o.f("hist.sigma_alpha");
  _DUET.sigma_delta = o.f("hist.sigma_delta");
  _DUET.use_smoothing    = o.i("hist.use_smoothing")   ;
  _DUET.use_smoothing_2D = o.i("hist.use_smoothing_2D");

  const int render = o.i("render");

  _DUET.aggregate_clusters = o.i("DUET.aggregate_clusters");
  _DUET.min_peak_fall   = o.d("DUET.min_peak_fall");
  _DUET.min_peak_dalpha = o.d("DUET.min_peak_dalpha");
  _DUET.min_peak_ddelta = o.d("DUET.min_peak_ddelta");
  _DUET.max_peak_scale_disparity = o.d("DUET.max_peak_scale_disparity");//10; // If smaller_peak * scale_disparity < max_peak it is rejected for being noise.

  _DUET.noise_threshold = o.d("DUET.noise_threshold");

  const DUETcfg DUET = _DUET; // Make every parameter constant to avoid mistakes
  // For the histogram smoothing.
  puts("Gen kernels");
  bool smooth_alpha, smooth_delta, smooth; // If smoothing is not possible they are 0.
  static Buffer<real> 
    conv_kernel_alpha(hist_alpha.gen_gaussian_kernel(DUET.sigma_alpha, &smooth_alpha)),
    conv_kernel_delta(hist_delta.gen_gaussian_kernel(DUET.sigma_delta, &smooth_delta)),
    conv_hist_alpha  (hist_alpha.bins()), 
    conv_hist_delta  (hist_delta.bins());
  static Matrix<real> 
    conv_kernel(hist.gen_gaussian_kernel(DUET.sigma_alpha, DUET.sigma_delta, &smooth)),
    conv_hist  (hist.xbins(),hist.ybins());
  puts("...DONE.");
  

  if ( ! (smooth_alpha && smooth_delta) )
    {
      printf(RED "Smoothing too small for histogram size. ABORTING\n" NOCOLOR);
      return 1;
    }

  int N_clusters = 0;

  
  Buffer<real> t_range(samples_all); 
  Buffer<real> block_t_range(time_blocks); 
  for (size_t t=0; t < samples_all; ++t)
    t_range[t] = Ts*t;
  for (int tb=0; tb < time_blocks; ++tb)
    block_t_range[tb] = tb*(FFT_slide*Ts);


  Buffer<real> W(FFT_N);
  if (opt.Option("window"))
    select_window(opt.getOption("window"), W);
  else
    select_window(o("window"), W);

  if (render >= 0)
    {
      puts("Calculating and writing histograms...");
      system("make cleanhists");
    }
  else
    puts("Calculating histograms...");      
    
  
  const size_t first_tb = o.f("skip_time_s") / (Ts*FFT_slide); 
  const size_t skip_samples = o.f("skip_time_s")*sample_rate_Hz;


  Timer rt_duet_timer;
  rt_duet_timer.start();
  for (idx time_block = first_tb; time_block < time_blocks; ++time_block)
    {
      idx block_offset = time_block*FFT_slide;

      for (idx i = 0; i < FFT_N; ++i)
	{
	  idx offset_i = i+block_offset;

	  if (offset_i < samples_input)
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

      hist.clear();
      hist_alpha.clear();
      hist_delta.clear();
      calc_alpha_delta(time_block, FFT_pN, sample_rate_Hz, X1, X2, alpha, delta, hist, hist_alpha, hist_delta, DUET);

      /*
	static Histogram2D<real> prod_hist(hist), diff_hist(hist);

	Gnuplot ph, po, pp, pd;
	prod_hist = hist;
	prod_hist *= old_hist;

	diff_hist = hist;
	diff_hist -= old_hist;

	old_hist.plot(po, "Old");
	prod_hist.plot(pp, "Prod");
	diff_hist.plot(pd, "Diff");

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
      */
      //old_hist = hist;
      cumulative_hist += hist;

      //ransac_test(time_block, FFT_pN, sample_rate_Hz, X1, X2, alpha, delta, hist, hist_alpha, hist_delta, DUET);

      ///////// Apply masks and rebuild current frame to audio and add it to the appropriate outputs
      if (! STATIC_REBUILD)
	{
	  static Buffer<int>  C(MAX_CLUSTERS, 0); // Classes: C[j] == id of designated stream
	  static IdList active_streams(MAX_ACTIVE_STREAMS);

	  static CyclicCounter<int> cc(N_accum);

	  chist += hist;
	  chist_alpha += hist_alpha;
	  chist_delta += hist_delta;

	  // Process the set of N_accumulation frames
	  if ((! cc.value() && (time_block >= N_accum /*|| N_accum==1*/)) ||
	      (time_block == time_blocks-1)) 
	    {
	      printf(GREEN "\t\t time_block (%lu+1)/%lu\n" NOCOLOR, time_block, time_blocks);

	      if (DUET.use_smoothing)
		{
		  chist_alpha.kernel_convolution(conv_kernel_alpha, conv_hist_alpha);
		  chist_delta.kernel_convolution(conv_kernel_delta, conv_hist_delta);

		  if (DUET.use_smoothing_2D) // WARNING: VERY SLOW OPERATION
		    chist.kernel_convolution(conv_kernel, conv_hist);
		}

	      heuristic_clustering2D(chist, clusters, DUET);
	      heuristic_clustering(chist_alpha, alpha_clusters, DUET, DUET.min_peak_dalpha);
	      heuristic_clustering(chist_delta, delta_clusters, DUET, DUET.min_peak_ddelta);

	      cout << YELLOW << clusters << NOCOLOR;
	      
	      N_clusters = clusters.eff_size(DUET.noise_threshold); 

	      puts(MAGENTA "Local confidences" NOCOLOR);
	      for (int n=0; n < N_clusters; ++n)
		{
		  real kurtosis_x, kurtosis_y;

		  Point2D<real> &pos = clusters.values[n];

		  // 1D kurtosis from 2D histogram.
		  excess_kurtosis(kurtosis_x, kurtosis_y, 
				  chist, conv_kernel, 
				  pos.x, pos.y);

		  printf(MAGENTA "%g %g      (%d %d)\n" NOCOLOR, 
			 kurtosis_x, kurtosis_y,
			 std::abs(kurtosis_x)<o.f("alpha_kurtosis"), 
			 std::abs(kurtosis_y)<o.f("delta_kurtosis"));

		}


	      static Histogram<real> alpha_level_pdf(100, HistogramBounds::Bounded);
	      static Histogram<real> delta_level_pdf(100, HistogramBounds::Bounded);

	      alpha_level_pdf.stretch(0, chist_alpha.max_value());
	      delta_level_pdf.stretch(0, chist_delta.max_value());

	      for (unsigned int bin_alpha=0; bin_alpha < chist_alpha.bins(); ++bin_alpha)
		alpha_level_pdf( chist_alpha.bin(bin_alpha) ) += 1/(real)chist_alpha.bins();
	      for (unsigned int bin_delta=0; bin_delta < chist_delta.bins(); ++bin_delta)
		delta_level_pdf( chist_delta.bin(bin_delta) ) += 1/(real)chist_delta.bins();

	      /*	      
	      static Gnuplot pLa, pLd;
	      pLa.set_labels("Relative Energy level (%)", "Fraction");
	      pLd.set_labels("Relative Energy level (%)", "Fraction");
	      alpha_level_pdf.replot(pLa,  "|alpha| pdf");
	      delta_level_pdf.replot(pLd,  "|delta| pdf");
	      */

	      printf(GREEN "Confidences 50%%AC: %f %f\n" NOCOLOR, 
		     1-array_ops::sum(&(*alpha_level_pdf.raw())[1], 50),
		     1-array_ops::sum(&(*delta_level_pdf.raw())[1], 50));
		     

	      // Calculate the confidence for test purposes for the current histogram
	      // Notice that the peak positions are sorted inside the function!
	      static Buffer<real> peaks_alpha(MAX_CLUSTERS), peaks_delta(MAX_CLUSTERS);
	      for (int n=0; n < N_clusters; ++n)
		{
		  peaks_alpha[n] = clusters.values[n].x;
		  peaks_delta[n] = clusters.values[n].y;
		}
	      printf(GREEN "Confidences: %g  %g\n" NOCOLOR, 
		     confidence(chist_alpha,peaks_alpha,N_clusters,DUET.sigma_alpha),
		     confidence(chist_delta,peaks_delta,N_clusters,DUET.sigma_delta));


	      // In the file ending we might not be able to accumulate as much as N_accum blocks.
	      if (time_block == time_blocks-1)
		N_accum = cc.value();

	      // First of the accumulated time blocks for this accumulation set.
	      unsigned int tb0 = time_block-N_accum+1; 
	      // Process all the time frames inside the current accumulated frames region.
	      for (unsigned int tr = N_accum; tr > 0; --tr)
		{
		  unsigned int tb = time_block-(tr-1);
		  
		  old_buffers = bufs.read();
		  new_buffers = bufs.next();

		  build_single_masks(masks, alpha(tb), delta(tb), X1_history(tb), X2_history(tb), clusters.values, N_clusters, FFT_pN, FFT_df, tmp_real_buffer_N_max, DUET);
		  apply_single_masks(*new_buffers, X1_history(tb), X2_history(tb), masks, clusters.values, N_clusters, FFT_pN, FFT_df, Xxo_plan, Xo, DUET);


		  if (tb == tb0) // This generates the class assignments C.
		    LDMB2C (Streams, active_streams, new_buffers, 
			    clusters.values, N_clusters, tb0, W, C, DUET);

		  // Add the separated buffers to the streams.
		  for (int j = 0; j < N_clusters; ++j)
		    {
		      static Buffer<real> 
			tmp_M(FFT_pN/2,0,fftw_malloc,fftw_free), 
			tmp_X(FFT_pN,0,fftw_malloc,fftw_free);

		      int id = C[j];

		      fftw_execute_r2r(xX1_plan, new_buffers->raw(j), tmp_X());
		      evenHC2magnitude(FFT_pN, tmp_X(), tmp_M());
		      Streams.add_buffer_at(id, j, *(*new_buffers)(j), tmp_M,
					    tb, FFT_slide, clusters.values[j]);
		    }
  		}

	      if (o.i("show_each_hist"))
		{
		  static Gnuplot px1;
		  if (o.i("show_x1W"))
		    px1.replot(x1(),FFT_N,"x1*W");
		  if (o.i("reset_plots"))
		    {
		      palpha.reset();
		      pdelta.reset();
		    }
		  palpha.plot(alpha_range(), (*chist_alpha.raw())(),hist_alpha.bins(), "alpha");
		  pdelta.plot(delta_range(), (*chist_delta.raw())(),hist_delta.bins(), "delta");	  
	  

		  static Gnuplot pa, pd;
		  static Buffer<real> f_range(FFT_pN);
		  for (int f=0; f < FFT_pN;++f)
		    f_range[f] = f*FFT_df;

		  if (o.i("show_alpha_delta_f"))
		    {
		      pa.set_labels("f (Hz)", "alpha"); pd.set_labels("f (Hz)", "delta (s)");
		      pa.replot(&f_range[DUET.Fmin],&alpha(time_block)[DUET.Fmin],DUET.Fmax-DUET.Fmin,"alpha");
		      pd.replot(&f_range[DUET.Fmin],&delta(time_block)[DUET.Fmin],DUET.Fmax-DUET.Fmin,"delta");
		    }


		  if (o.i("show_M"))      
		    {
		      static Gnuplot pM1, pM12;
		      pM1.set_xlabel("f (Hz)");
		      static Buffer<real> M1(FFT_pN/2), M2(FFT_pN/2), M12(FFT_pN/2);

		      evenHC2magnitude(X1,M1);
		      evenHC2magnitude(X2,M2);

		      pM1.replot(f_axis(),M1(),FFT_pN/2,"M1");

		      for (size_t i=0; i < M12.size();++i)
			M12[i] = M1[i]*M2[i];

		      pM12.set_labels("f (Hz)", "M1*M2");
		      pM12.replot(&f_range[DUET.Fmin],&M12[DUET.Fmin],DUET.Fmax-DUET.Fmin,"M1M2");
		    }

		  if (o.i("show_each_hist")>1)
		    {
		      // Write the clusters to the plot overlay
		      std::ofstream clusters_dat;
		      clusters_dat.open("s_duet.dat");
		      for (idx i=0; i < N_clusters; ++i)
			clusters_dat << clusters.values[i].x << " " << clusters.values[i].y << " 0\n\n";
		      clusters_dat.close();

		      chist.write_to_gnuplot_pm3d_data("chist.dat");
		      RENDER_HIST("chist.dat", "Hist", o.i("hist_pause")); 
		    }
		  else
		    if (o.i("hist_pause"))
		      wait();    
		}

	      if (render >= 0)
		{
		  std::string filepath = "hist_dats/" + itosNdigits(time_block,10) + ".dat";
		  chist.write_to_gnuplot_pm3d_binary_data(filepath.c_str());
		  //system(("cp "+filepath+" tmp_dats/hist.dat && gen_movie.sh tmp_dats tmp_pngs 3D.gnut && feh tmp_pngs/hist.png").c_str());
		}

	      chist.clear();
	      chist_alpha.clear();
	      chist_delta.clear();
	    } // end of if (! cc.value())...
	  ++cc;
	}
    } // End of blockwise processing

	
  ///// Static Heuristic Rebuilding! ///////////////////////////////////////////////////////////////////
  puts(GREEN "Doing static-heuristic rebuilding..." NOCOLOR);
  cumulative_hist -= hist;
  if (STATIC_REBUILD && DUET.use_smoothing)
    {
      hist_alpha.kernel_convolution(conv_kernel_alpha, conv_hist_alpha);
      hist_delta.kernel_convolution(conv_kernel_delta, conv_hist_delta);
      
      if (DUET.use_smoothing_2D) // WARNING: VERY SLOW OPERATION
	cumulative_hist.kernel_convolution(conv_kernel, conv_hist);
    }  
  Timer cumulative_hist_write_timer; cumulative_hist_write_timer.start();
  cumulative_hist.write_to_gnuplot_pm3d_data("cumulative_hist.dat");
  cumulative_hist_write_timer.stop();


  heuristic_clustering2D(cumulative_hist, cumulative_clusters, DUET);

  cout << cumulative_clusters;

  N_clusters = cumulative_clusters.eff_size(DUET.noise_threshold);



  if (STATIC_REBUILD)
    {
      // Build the masks and rebuild the signals
      for (idx t_block = 0; t_block < time_blocks; ++t_block)
	{
	  old_buffers = bufs.read();
	  new_buffers = bufs.next();
	  build_single_masks(masks, alpha(t_block), delta(t_block), X1_history(t_block), X2_history(t_block), cumulative_clusters.values, N_clusters, FFT_pN, FFT_df, tmp_real_buffer_N_max, DUET);
      
	  apply_single_masks(*new_buffers, X1_history(t_block), X2_history(t_block), masks, cumulative_clusters.values, N_clusters, FFT_pN, FFT_df, Xxo_plan, Xo, DUET);

	  // Explicitly use the initial region FFT_N and exclude the padding FFT_pN.
	  
	  write_data(wav_out, new_buffers, FFT_N, FFT_slide);	  
	  //      swap(bufs_ptr, bufs_ptr2);
	}		      
    }

  rt_duet_timer.stop(); 

  // Write data to disk //////////////////////////////////////////////////////////////////////////

  system("rm -f x*_rebuilt.wav ibmx*_rebuilt.wav");

  // IBM masks from mix using true sources knowledge (static and dynamic are rebuilt here)
  Buffers<real> ibm_X_bufs(original_waves_x1.buffers(),FFT_N,fftw_malloc,fftw_free); 
 
 
  Buffers<bool> ibm_masks(original_waves_x1.buffers(),FFT_N/2,false,fftw_malloc,fftw_free); // By default are not assigned (false)

  for (size_t tb=0; tb < (size_t)time_blocks; ++tb)
    {
      build_and_apply_multi_mono_ibm_masks(ibm_masks, ibm_X_bufs, original_waves_x1, tb*FFT_slide, X1_history(tb), FFT_N, xX1_plan, W, o.f("Phi_x"),DUET);

      for (int n=0; n < N; ++n)
	{
	  static Buffer<real> x_buf(FFT_N,0,fftw_malloc,fftw_free);
	  x_buf.clear();
	  fftw_execute_r2r(Xxo_plan, ibm_X_bufs.raw(n), x_buf());
	  ibm_out(n)->add_at(x_buf, tb*FFT_slide);
	}
    }

  for (int source = 0; source < N; ++source)
    {
      std::string wav_filepath("ibmx"+itosNdigits(source,N_EXPORT_DIGITS)+"_rebuilt.wav");
      printf("%s...", wav_filepath.c_str());
      fflush(stdout);
      print_status( wav::write_mono(wav_filepath, ibm_out.raw(source), samples_input, sample_rate_Hz) );
    }
  /// End of IBM masks section



  int wav_N = ( STATIC_REBUILD ? N_clusters : N_max );

  if (STATIC_REBUILD)
    {
      // Estimated sources
      for (int source = 0; source < wav_N; ++source)
	{
	  std::string wav_filepath("x"+itosNdigits(source,N_EXPORT_DIGITS)+"_rebuilt.wav");
	  printf("%s...", wav_filepath.c_str());
	  fflush(stdout);
	  print_status( wav::write_mono(wav_filepath, wav_out.raw(source), samples_input, sample_rate_Hz) );
	}
    }
  else
    {
      // Look for the global stream normalization factor
      real streams_max_abs = 0;
      for (unsigned int stream_id=1; stream_id<=Streams.latest_id();++stream_id)
	{
	  real maxabs = Streams.stream(stream_id)->max_abs();

	  if (maxabs > streams_max_abs)
	    streams_max_abs = maxabs;
	}

      for (unsigned int stream_id = 1; stream_id <= Streams.latest_id(); ++stream_id)
	{
	  if (!Streams.active_blocks(stream_id))
	    continue;

	  std::string wav_filepath("xstream"+itosNdigits(stream_id,N_EXPORT_DIGITS)+"_rebuilt.wav");
	  printf("(active_blocks=%u) %s...", Streams.active_blocks(stream_id), wav_filepath.c_str());
	  fflush(stdout);
	  print_status( wav::write_mono(wav_filepath, (*Streams.stream(stream_id))(), samples_input, sample_rate_Hz,streams_max_abs) );
	}
    }

  // Evaluate data (separation stats) ////////////////////////////////////////////////////
  puts("");
  Buffer<real> SNR0 (original_waves_x1.buffers());
  for (unsigned int n=0; n < original_waves_x1.buffers(); ++n)
    {
      real snr = SNR(&x1_wav[skip_samples],original_waves_x1.raw(n,skip_samples),samples_input-skip_samples);
      printf(GREEN "o%d : SNR = %gdB\n" NOCOLOR, n, snr);
      SNR0[n] = snr;
    }
  // Write the minimum SNR (prior to separation) and maximum achievable (using the IBM masks)
  separation_stats(ibm_out, original_waves_x1, original_waves_x1.buffers(), samples_input, skip_samples, ibm_logpath, &SNR0);
    
  int degenerate_count; // How many outputs are degenerate
  if (STATIC_REBUILD)
    {
      puts("\nStatic Separation:");
      degenerate_count = separation_stats(wav_out, original_waves_x1, wav_N, samples_input, skip_samples, logpath);
    }
  else
    {
      puts("\nDynamic separation:");
      // Even if there are streams that were merged and none took their place the "holes" will hopefully give nan stats.
      Buffers<real> streams_out(HIGHEST_STREAM_ID, samples_all, fftw_malloc, fftw_free);
      for (int s_id=1; s_id <= HIGHEST_STREAM_ID; ++s_id)
	(*streams_out(s_id-1)).copy(*Streams.stream(s_id), samples_input);
      degenerate_count=separation_stats(streams_out, original_waves_x1, HIGHEST_STREAM_ID, samples_input, skip_samples, logpath);
    }


  fftw_destroy_plan(xX1_plan); 
  fftw_destroy_plan(xX2_plan);
  fftw_destroy_plan(Xxo_plan);


  printf(GREEN "\n%u streams - %d merged streams in %lu time blocks.\n\n" NOCOLOR, ACQUIRED_STREAMS, MERGED_STREAMS, time_blocks);


  Streams.release_ids();



  // Final 2D Histogram with overlays and other plots /////////////////////////////////////////////////////////////////
  system("rm -f ecoduet.log");
  if (STATIC_REBUILD)
    {
      std::ofstream ecoduet_log("ecoduet.log", std::ios::trunc);
      for (int n=0; n < (int)cumulative_clusters.eff_size(DUET.noise_threshold); ++n)
	{
	  ecoduet_log << cumulative_clusters.values[n].x << " " 
		      << cumulative_clusters.values[n].y << "\n";
	}
      ecoduet_log.close();
    }

  if (render > 0)
    Guarantee0( system("make render") , "Couldn't generate the movies.");
  cout << "#Clusters = " << N_clusters <<"\n";
  cumulative_clusters.print(N_clusters);
  system("cat s.dat");
  puts("\nSuccess!");

  if (o.i("show_final_plots") || opt.getFlag('p'))
    {
      if (! STATIC_REBUILD)
	{
	  static Gnuplot pTalpha, pTdelta, pT;
	  draw_trajectories (Streams, time_blocks, Ts*FFT_slide, pTalpha, pTdelta, pT);
	}

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
      RENDER_HIST("cumulative_hist.dat", "Cumulative hist", o.i("hist_pause"));
    }


  cout << MAGENTA << rt_duet_timer.getElapsedTime()-cumulative_hist_write_timer.getElapsedTime() << " (s)\n" << NOCOLOR;

  if (o.i("wait"))
    wait();

  return 0;
}

