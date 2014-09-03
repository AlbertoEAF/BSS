#include "duet.h"


//#define OLD_MASK_BUILD
//#define OLD_PEAK_ASSIGN

const int MAX_PRECLUSTERS = 12;
const int MAX_MARGINAL_PEAKS = 16;

// After giving one buffer, at the end of the next (blocking) call, it can be released
// Cannot be reassigned to a different output because it stores the previous buffer and it would have conflicts, thus all the data must be passed 
size_t write_data(Matrix<real> &o, Matrix<real> *new_buffer, const size_t FFT_N, const size_t FFT_slide)
{
  static int buffers = 1; // how many buffers are in current use (state variable)
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
	  //	  RELEASE(a);
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

/// Transforms alpha back to a.
real alpha2a (real alpha)
{
  return (alpha + std::sqrt(alpha*alpha + 4.0)) * 0.5;
}

/// Fills a buffer of size FFT_N/2 // To each bin will be assigned the number of the source. values < 0 indicate that the bin won't be assigned a source (noise or intentional algorithm rejection/discard). 
/// Thus, a single buffer is required to hold all the masks
void build_masks(Buffer<int> &masks, real *alpha, real *delta, real *X1, real *X2, Buffer<Point2D<real> > &clusters, idx FFT_N, idx FFT_half_N, real FFT_df, Buffer<real> &calc_buffer)
{
      Buffer<int> old_masks(masks);
      idx masks_diffs = 0;
  int K = clusters.size();
  for (idx f = 0; f < FFT_half_N; ++f)
    {
      real omega = _2Pi * f * FFT_df;
      idx f_im = FFT_N - f;

      // Too simplistic: masks[f] = closest_cluster(Point2D<real>(alpha[f],delta[f]), clusters);
      
      for (int k=0; k < K; ++k)
	{
	  real a_k = alpha2a(clusters[k].x);
	  real delta_k = clusters[k].y;

	  calc_buffer[k] = std::norm(a_k*std::polar<real>(1,-delta_k*omega) * std::complex<real>(X1[f],X1[f_im]) - std::complex<real>(X2[f],X2[f_im])) / (1.0 + a_k*a_k);
	}
      masks[f] = array_ops::min_index(calc_buffer(), K);

      old_masks[f] = closest_cluster(Point2D<real>(alpha[f],delta[f]), clusters);

      if (masks[f]!=old_masks[f])
	masks_diffs += 1;
    }
#ifdef OLD_MASK_BUILD
  masks = old_masks;
#endif // OLD_MASK_BUILD
  //  cout << RED << masks_diffs << NOCOLOR << endl;
 }

void apply_masks(Matrix<real> &buffers, real *alpha, real *X1, real *X2, Buffer<int> &masks, Buffer<Point2D<real> > &clusters, uint active_sources, idx FFT_N, idx FFT_half_N, real FFT_df, fftw_plan &FFTi_plan, Buffer<real> &Xo)
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
      fftw_execute_r2r(FFTi_plan, Xo(), buffers(source));
    }

  buffers /= (real)FFT_N;
}


/** Dtotal metric to compare the original signal with the extracted one.

    @param[in] e - Estimated signal
    @param[in] o - Original  signal
    @param[in] samples - Number of samples

 */


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
  Matrix<real> original_waves_x1(N, samples);
  for (int i = 0; i < N; ++i)
    {
      SndfileHandle wav_file("sounds/"+std::to_string(i)+"x0.wav");
      if (! wav::ok (wav_file))
	return EXIT_FAILURE;

      wav_file.read(original_waves_x1(i), samples);
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
    delta(time_blocks, FFT_pN/2), 
    wav_out(N_max, time_blocks*FFT_slide);

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
  Buffer<real> alpha_range(hist_alpha.bins()), delta_range(hist_delta.bins()); // Buffers for the axis of alpha and delta
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

  RankList<real, Point2D<real> > preclusters(MAX_PRECLUSTERS,0.0,Point2D<real>());
  RankList<real, Point2D<real> > cumulative_clusters(N_max,0.0,Point2D<real>());

  RankList<real, real> delta_preclusters(MAX_MARGINAL_PEAKS, 0.0), alpha_preclusters(delta_preclusters);

  _DUET.p = o.f("hist.p");
  _DUET.q = o.f("hist.q");

  _DUET.sigma_alpha = o.f("hist.sigma_alpha");
  _DUET.sigma_delta = o.f("hist.sigma_delta");

  _DUET.use_smoothing = o.i("hist.use_smoothing");
  
  printf("%g %g :: %g %g\n", _DUET.sigma_alpha, hist.dx(), _DUET.sigma_delta, hist.dy());

  if (o.i("hist.assert_use_smoothing"))
    Guarantee(_DUET.use_smoothing, "Smoothing disabled! (Make sure histogram resolution is bigger than the smoothing)");

  if (_DUET.use_smoothing)
    puts(GREEN "Smoothing enabled!" NOCOLOR);
  else
    puts(RED "Smoothing Disabled" NOCOLOR);


  const int RENDER = o.i("render");


  _DUET.aggregate_clusters = o.i("DUET.aggregate_clusters");
  _DUET.min_peak_fall   = o.d("DUET.min_peak_fall");
  _DUET.min_peak_dalpha = o.d("DUET.min_peak_dalpha");
  _DUET.min_peak_ddelta = o.d("DUET.min_peak_ddelta");
  _DUET.max_peak_scale_disparity = o.d("DUET.max_peak_scale_disparity");//10; // If smaller_peak * scale_disparity < max_peak it is rejected for being noise.

  _DUET.noise_threshold = o.d("DUET.noise_threshold");

  const DUETcfg DUET = _DUET; // Make every parameter constant to avoid mistakes


  /////////////////////////////// Convolution 2D smoothing tests
  /*
  Matrix<real> conv_kernel(hist.gen_gaussian_kernel(DUET.smoothing_Delta_alpha,DUET.smoothing_Delta_delta));

  Matrix<real> conv_hist(hist.xbins(),hist.ybins());

  hist.clear();
  hist(0,0) += 1;

  Gnuplot phist;
  
  hist.kernel_convolution(conv_kernel, conv_hist);

  hist.plot(phist,"Conv");
  
  wait();
  */
  ///////////////////////////////


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

      //evenHC2magnitude(FFT_N, X1(), M1());
      //evenHC2magnitude(FFT_N, X2(), M2());
      /*
      hist.clear();
      hist_alpha.clear();
      hist_delta.clear();
      calc_alpha_delta(time_block, FFT_pN, sample_rate_Hz, X1, X2, alpha, delta, hist, hist_alpha, hist_delta, DUET);
      ransac_test(time_block, FFT_pN, sample_rate_Hz, X1, X2, alpha, delta, hist, hist_alpha, hist_delta, DUET);
      */
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

      //static Buffer<real> hist_alpha(hist.xbins()), hist_delta(hist.ybins());
      static Gnuplot palpha, pdelta;
      palpha.reset(); pdelta.reset();
      palpha.setstyle("lines");
      pdelta.setstyle("lines");

      if (o.i("show_each_hist"))
	{
	  /*
	    hist.marginal_x(hist_alpha);
	    hist.marginal_y(hist_delta);
	  */

	  palpha.plot_xy(alpha_range(), (*hist_alpha.raw())(),hist_alpha.bins(),"alpha");
	  pdelta.plot_xy(delta_range(), (*hist_delta.raw())(),hist_delta.bins(),"delta");
	  

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

	
  /* // Find the centroid (alpha,delta) of the dataset 

     real *ptr_a_t, *ptr_d_t; 
     real avg_error_alpha=0, avg_error_delta=0;
     for (idx t = 0; t < time_blocks; ++t)
     {
     static size_t elements = alpha.d();

     ptr_a_t = alpha(t);
     ptr_d_t = delta(t);

     calpha[0] = avg(ptr_a_t, elements);
     cdelta[0] = avg(ptr_d_t, elements);

     avg_error_alpha += std::abs(calpha[0]-true_alpha[0]);	
     avg_error_delta += std::abs(cdelta[0]-true_delta[0]);

     usleep(o.stoi("sleep_us", Ignore)); // use nanosleep instead - posix and resilient against interrupts
     if (WAIT)
     wait();
     }
     avg_error_alpha /= (real)time_blocks;
     avg_error_delta /= (real)time_blocks;
     printf("<e_ca> = %f     <e_cd> = %f\n", avg_error_alpha, avg_error_delta);
  */
  ///// Static Heuristic Rebuilding! ///////////////////////////////////////////////////////////////////
  
  static Gnuplot pM1hist;
  pM1hist.plot((*M1hist.raw())(),FFT_pN/2,"M1 histogram");


  puts(GREEN "Doing static-heuristic rebuilding..." NOCOLOR);
  cumulative_hist -= hist;
  cumulative_hist.write_to_gnuplot_pm3d_data("cumulative_hist.dat");
  
  /*
  cumulative_hist.clear();
  cumulative_hist.smooth_add(1, -0.05, 5e-6, 0.1, 1e-5);
  Gnuplot cumulative_hist_plot;
  cumulative_hist_plot.cmd("set xlabel 'alpha'; set ylabel 'delta (s)'");
  cumulative_hist.plot(cumulative_hist_plot, "Cumulative Histogram");
  */  

  heuristic_clustering2D(cumulative_hist, cumulative_clusters, DUET);


  cout << cumulative_clusters;

  bool found_clusters = (cumulative_clusters.eff_size(DUET.noise_threshold) ? 1:0);

  Buffer<Point2D<real> > clusters(cumulative_clusters.eff_size(DUET.noise_threshold)+(!found_clusters)); // if no clusters are found 0 size would blow the program
  clusters.copy(cumulative_clusters.values,clusters.size());

  std::ofstream hist_cfg;
  hist_cfg.open("h.cfg");
  hist_cfg << hist.ybins();
  hist_cfg.close();


  // Write the clusters to the plot overlay
  std::ofstream clusters_dat;
  clusters_dat.open("s_duet.dat");
  for (idx i=0; i < clusters.size(); ++i)
    clusters_dat << clusters[i].x << " " << clusters[i].y << " 0\n\n";
  clusters_dat.close();

  // Plot the 3D histogram with gnuplot and the simulation and DUET overlays
  // Since the "" must be passed with quotes inside the gnuplot command a triple \ is needed and  a single \ is needed for the outer command.
  RENDER_HIST("cumulative_hist.dat", "Cumulative hist", 1);
  


  //// Each of the clusters should now belong to a source: create masks and separate the sources.

  Buffer<int> masks(FFT_pN/2); 

  // 2 sets of buffers are needed to allow up to 50% overlapping.
  Matrix<real> 
    bufs1(N_max, FFT_pN), *bufs_ptr  = &bufs1,
    bufs2(N_max, FFT_pN), *bufs2_ptr = &bufs2;

  system("rm -f x*_rebuilt.wav");

  // Build the masks and rebuild the signals
  for (idx t_block = 0; t_block < time_blocks; ++t_block)
    {
      build_masks(masks, alpha(t_block), delta(t_block), X1_history(t_block), X2_history(t_block), clusters, FFT_pN, FFT_pN/2, FFT_df, tmp_real_buffer_N_max);
      
      apply_masks(*bufs_ptr, alpha(t_block), X1_history(t_block), X2_history(t_block), masks, clusters, clusters.size(), FFT_pN, FFT_pN/2, FFT_df, Xxo_plan, Xo);
      // Explicitly use the initial region FFT_N and exclude the padding FFT_pN.
      write_data(wav_out, bufs_ptr, FFT_N, FFT_slide);

      swap(bufs_ptr, bufs2_ptr);
      /*
      for(uint source = 0; source < clusters.size(); ++source)
	for (idx i = 0; i < FFT_N && i+t_block*FFT_N < samples; ++i)
	  wav_out(source, i+t_block*FFT_N) = (*bufs_ptr)(source,i);      
      */
    }		

  for (uint source = 0; source < clusters.size(); ++source)
    {
      std::string wav_filepath("x"+itos(source)+"_rebuilt.wav");
      printf("%s...", wav_filepath.c_str());
      fflush(stdout);
      print_status( wav::write_mono(wav_filepath, wav_out(source), samples, sample_rate_Hz) );
    }
	
  //  separation_stats(wav_out, original_waves_x1, N, samples);


  ///// End of Static Heuristic Rebuilding! ///////////////////////////////////////////////////////////////////


  //	write_mono_wav ("gh_fft.wav", wav_out, N_wav+h_size-1, sample_rate_Hz);
  //wait();
  //sleep(0.5);
  fftw_destroy_plan(xX1_plan); 
  fftw_destroy_plan(xX2_plan);
  fftw_destroy_plan(Xxo_plan);
	

  if (RENDER > 0)
    Guarantee0( system("make render") , "Couldn't generate the movies.");
  cout << "#Clusters = " << clusters.size()<<"\n";
  cout << clusters << "\n";
  system("cat s.dat");
  puts("\nSuccess!");

  if (WAIT)
    wait();

  return 0;
}






/////////////

