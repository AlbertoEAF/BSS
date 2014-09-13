// Comment for old platforms that do not fully support C++11
#define NOISE_SUPPORT

#include <iostream>

#include <vector>
#include <map>

#include <cmath>

#include <stdlib.h> // strtof

#include <cfloat> // FLT_EPSILON

#include "custom_assert.h"

//#include "Array.h"
#include "Buffer.h"
#include "Matrix.h"

#include <string>
#include <fstream>
#include <sstream>

#include "libs/config_parser.h"
#include "wav.h"

#include "Vector.h"

#include "array_ops.h"

#ifdef NOISE_SUPPORT
#include <chrono> // C++11: time-seed for generator
#include <random> // C++11: std::normal_distribution
#endif//NOISE_SUPPORT

//#include <float.h> // FLT_EPSILON
#include <limits> // std::numeric_limits -> epsilon()

//#include "extra.h"

//#define USE_RELATIVE_DELAY

using std::cout;
using std::endl;
using std::string;
using std::map;

typedef double real;
typedef Vector3D<real> Vec;

typedef std::vector<string> Strings;
typedef std::vector<Vec>    Vecs   ;

typedef unsigned int uint;
typedef long int idx;

Strings sources_filepaths(Options &o)
{
    map<string,string> &m = o.raw();

    Strings strings;

    for (const auto &v: m)
    {
        if (v.first.rfind(".wav") != std::string::npos)
        {
            AssertRuntime(v.first.length()>4, "You provided an invalid .wav filepath."); // can't be only the extension
            
            strings.push_back(v.first);
        }
    }

    return strings;
}

// Builds a Vec from a string in format " x y z "
Vec vec(const string &s)
{
    std::stringstream ss(s);
    real x,y,z;

    ss >> x;
    ss >> y;
    ss >> z;

    AssertRuntime (! ss.fail(), "Reading the position of one of the sources failed: %s", s.c_str());

    return Vec(x,y,z);
}

real distance(Vec &a, Vec &b)
{
    return (b-a).length();
}

// Returns the sample rate and checks if all the files have the same samplerate otherwise aborts.
int get_project_sample_rate(Strings &sources)
{
    int sample_rate_Hz = 0;
    for (auto &path: sources)
    {
        SndfileHandle wav(path);
        AssertRuntime(wav::ok(wav), "Wav file %s is corrupted or doesn't exist.", path.c_str());

        if (sample_rate_Hz)            
        {

            AssertRuntime(sample_rate_Hz == wav.samplerate(), "Wav file %s doesn't have the same samplerate.", path.c_str());
        }
        
        else
            sample_rate_Hz = wav.samplerate();
    }

    return sample_rate_Hz;
}

// Returns the maximum number of samples of a file
idx get_project_samples(Strings &sources)
{
    idx max_samples = 0;
    for (auto &path: sources)
    {
        SndfileHandle wav(path);
        if (wav.frames() > max_samples)
            max_samples = wav.frames();
    }
    return max_samples;
}

Matrix<real> read_project_files(Strings &paths, idx samples)
{
    Matrix<real> waves(paths.size(), samples);

    for (int i = 0; i < paths.size(); ++i)
    {
        SndfileHandle wav(paths[i]);
        wav.read(waves(i), wav.frames()); // We have already guaranteed that samples > wav.frames()
    }

    return waves;
}

real linear_interpolation(Matrix<real> &W, int w, real t, real dt_s)
{
    // w is the index of the wave ( source ).
    
    long int n = t/dt_s;
    real t_n = n * dt_s;
    

    if (t < 0)
        return 0;
    else if (n >= W.d()-1) // not needed if we do not extend the waves simulation
        return 0;

    return W(w,n) + (t-t_n)*(W(w,n+1)-W(w,n))/dt_s;
}

#include "gnuplot_ipp/gnuplot_ipp.h"

int main()
{
    Options o("csim.cfg", Quit, 1);

    const string sources_folder = o("sources_folder")+"/";
    const string output_folder  = o("output_folder" )+"/";

    const real c = o.f("c"); // speed of sound (m/s)

    Strings source_files = sources_filepaths(o);

    Vecs source_pos;
    for (auto &wav_path : source_files)
        source_pos.push_back(vec(o(wav_path)));

    cout << "\nSources:\n";
    for (size_t i = 0; i < source_files.size(); ++i)
        cout << " " << source_files[i] << " @ " << source_pos[i] << endl;
    cout << endl;

    // Now add the relative path to the filenames
    for (auto &path: source_files)
        path = sources_folder + path;

    const int M = 2; // DUET has only 2 mics
    const int N = source_files.size();
    printf("M = %d; N = %d\n", M, N);
    if (N>1)
      printf("Distance between sources 0 and 1: %f\n\n", distance(source_pos[1],source_pos[0]));

    const int sample_rate_Hz = get_project_sample_rate(source_files);
    const idx samples = get_project_samples(source_files);

    Matrix<real> waves = read_project_files(source_files, samples);

    printf("Processing %lu samples per wave @ %d Hz.\n", waves.d(), sample_rate_Hz);

    Vecs mics;
    real Delta;
    if (std::abs(o.f("Delta",Ignore)) > FLT_EPSILON)
      Delta = o.f("Delta");
    else
      Delta = c/(2*sample_rate_Hz);

    mics.push_back(Vec(-Delta/2.0, 0.0, 0.0));
    mics.push_back(Vec( Delta/2.0, 0.0, 0.0));

    printf("Delta = %f (cm)\n", Delta*100);

    // Relation to origins: Delay and Attenuation matrices
    Matrix<real> D(M,N), A(M,N);     // Absolute  quantities
    Buffer<real> delta(N), alpha(N); // inter-mic quantities

    Guarantee(M==2, "delta and a are only implemented for DUET.");

    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            real d_nm = distance(source_pos[n], mics[m]);
            D(m,n) = d_nm / c;
	    //           A(m,n) = 1 / (4*M_PI*d_nm*d_nm);
	    A(m,n) = 1 / d_nm;
        }
    }

    for (int n = 0; n < N; ++n)
    {
        delta[n] = D(1,n) - D(0,n);
        real a = A(1,n) / A(0,n);
        alpha[n] = a - 1/a;

	#ifdef USE_RELATIVE_DELAY
	D(1,n) -= D(0,n);
	D(0,n) = 0;
	#endif
    }

    cout << "delta: " << delta;
    cout << "alpha: " << alpha;

    const real max_A = array_ops::max(A.raw(), N*M);
    A /= max_A;

    /* Noise generator + distribution */
#ifdef NOISE_SUPPORT
    const real noise_stddev = o.d("noise.stddev", Ignore);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);

    std::normal_distribution<real> normal(0,noise_stddev);
#endif//NOISE_SUPPORT
    /* ****************************** */

    /*
    puts("");
    Buffer<real> out(samples);
    const real dt_s = 1.0/(real)sample_rate_Hz;
    for (int n = 0; n < N; ++n)
    {
        printf("Generating microphone measurements for source %d/%d.\n", n+1, N);

        for (int m = 0; m < M; ++m)
        {
            for (idx i = 0; i < samples; ++i)            
                out[i] = A(m,n) * linear_interpolation(waves,n,i*dt_s-D(m,n), dt_s);

            write_mono_wav ((output_folder+std::to_string(n)+"x"+std::to_string(m)+".wav").c_str(), out(), samples, sample_rate_Hz);
        }
    }

    puts("Generating measurements for the full mix.\n");
    for (int m = 0; m < M; ++m)
    {
        out.clear();
        for (idx i = 0; i < samples; ++i)
        {
            real t_i = i * dt_s;
            for (int n = 0; n < N; ++n)
            {
                out[i] += A(m,n) * linear_interpolation(waves,n, t_i-D(m,n), dt_s);
            }
        }

	if (noise_stddev > std::numeric_limits<real>::epsilon()) 
	  {
	    for (size_t i=0; i < samples; ++i)
	      out[i] += normal(generator);
	  }
	
        write_mono_wav ((output_folder+"x"+std::to_string(m)+".wav").c_str(), out(), samples, sample_rate_Hz, 1/(real)N );
   }
*/

    system( ("rm -rf " + output_folder + "/*").c_str() );

    puts("");
    Matrix<real> out(M,samples);
    const real dt_s = 1.0/(real)sample_rate_Hz;
    for (int n = 0; n < N; ++n)
    {
        printf("Generating microphone measurements for source %d/%d.\n", n+1, N);

        for (int m = 0; m < M; ++m)
        {
            for (idx i = 0; i < samples; ++i)            
	      out(m,i) = A(m,n) * linear_interpolation(waves,n,i*dt_s-D(m,n), dt_s);

	    //            write_mono_wav (output_folder+std::to_string(n)+"x"+std::to_string(m), out(), samples, sample_rate_Hz);
        }

	wav::write(output_folder+"s"+std::to_string(n)+"x", out, sample_rate_Hz,1);
    }
    


    puts("Generating measurements for the full mix.\n");
    out.clear();
    for (int m = 0; m < M; ++m)
    {
        for (idx i = 0; i < samples; ++i)
        {
            real t_i = i * dt_s;
            for (int n = 0; n < N; ++n)
            {
	      out(m,i) += A(m,n) * linear_interpolation(waves,n, t_i-D(m,n), dt_s);
            }
        }
#ifdef NOISE_SUPPORT
	if (noise_stddev > std::numeric_limits<real>::epsilon()) 
	  {
	    for (size_t i=0; i < samples; ++i)
	      out(m,i) += normal(generator);
	  }
#endif//NOISE_SUPPORT
	//   write_mono_wav ((output_folder+"x"+std::to_string(m)+".wav").c_str(), out(), samples, sample_rate_Hz, 1/(real)N );
    }
    wav::write(output_folder+"x", out, sample_rate_Hz, 1);


    // Record the "real" source positions to a log
    /*
    with open("simulation.log","wt") as log:
    log.write(str(N)+"\n")
    for n in range(N):
        log.write("{0} {1}\n".format(a[0][n], d[0][n]))
    */
    std::ofstream sim_log;
    sim_log.open("simulation.log");
    sim_log << std::to_string(N) << std::endl;
    for (int n = 0; n < N; ++n)
      sim_log << std::to_string(alpha[n]) << " " << std::to_string(delta[n]) << std::endl;
    sim_log.close();


    return 0;
}
