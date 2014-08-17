/**
 * @file cuda_extend.h
 * 
 * This header allows easy integration of CUDA and C and C++ code by providing macros that facilitate
 * creation of different versions.
 * 
 * 
 * */

#ifndef cuda_extend_H__
#define cuda_extend_H__

#include <assert.h>
#include "cuda_common.h"
#include <stdio.h>
#include <string.h>

#include <iostream> // SHALL BE REMOVED IN THE FUTURE FOR FULL COMPATIBILITY WITH C -> switch all cout's for printf's

/// Use this function around CUDA calls to catch any CUDA errors and terminate the program if an error arises.
#define cudaSafe(x) CUDA_SAFE_CALL(x)
/// The same purpose of cudaSafe() but useful in checking any errors. Useful for kernels. @see cudaCheckKernel()
#define cudaCheckError(error_msg) CUT_CHECK_ERROR(error_msg)
/// Use this to check kernels after synchronizing cuda. @see cudaSync()
#define cudaCheckKernel(kernel_name) cudaCheckError("kernel " # kernel_name)

#ifdef __CUDACC__
/// Same as the original CUDA functions but it is safe
#define cudaSafeEventCreate(event) cudaSafe(cudaEventCreate(event))
#define cudaSafeEventRecord(event,...) cudaSafe(cudaEventRecord(event,##__VA_ARGS__))
#define cudaSafeEventDestroy(event) cudaSafe(cudaEventDestroy(event))

#define cudaSafeStreamCreate(stream) cudaSafe(cudaStreamCreate(stream))
#define cudaStreamSync(stream) cudaSafe(cudaStreamSynchronize(stream))
#define cudaSafeStreamDestroy(stream) cudaSafe(cudaStreamDestroy(stream))

#define cudaSafeSetDevice(device) cudaSafe(cudaSetDevice(device))
#define cudaSafeGetDevice() cudaSafe(cudaSafeGetDevice())
#endif //__CUDACC__

//#include "p99/p99.h" // For more advanced macros

 typedef long double ldouble;

#ifdef __CUDACC__



/// Internal implementation: Through compilation fetches the otherwise inaccessible __CUDA_ARCH__ that is only available through device functions and makes it accessible globally through the name CUDA_ARCH   
void __device__ fetch_cuda_arch___()
{
  /// Only required this name: CUDA_ARCH_ for proper documentation in doxygen
#define CUDA_ARCH_ __CUDA_ARCH__
}

/// Globally accessible alias to @b _\_CUDA_ARCH\_\_ even outside device functions. @warn Can only be used for #if CUDA_ARCH < ..  comparisons or #ifdef's, not in standard code like cout << CUDA_ARCH;
#define CUDA_ARCH CUDA_ARCH_

#endif // __CUDACC__

#ifdef __CUDACC__ // CUDA IS AVAILABLE
#define _cudaboth_       __host__   __device__
#define _cudahost_       __host__
#else         
/* It's safe to reduce these to nothing since they're intended to run on the host */
#define _cudaboth_
#define _cudahost_ 
#endif // __CUDACC__


#define _cudaglobal_ __global__

// Avoids name mangling which makes debug easier
#define _cudaglobalC_ extern "C" __global__

#define _cudadevice_ __device__



/* 
   Always defining these guarantees that 
  executing device-specific code throws 
  error since it's not available. 

   You should always protect 
  cuda-specific code with CUDA(...)! 

Why? 
     For instance, if one declares specific methods for 
  the host and for the device and you don't have CUDA,
  you don't want that code to be available to the host
  even if it could be compiled without errors.
*/



/// @def CUDA Always use this to compile __device__ specific code AND you DON'T have to return an l-value
#ifdef __CUDACC__
//#define CUDA(x) do {x} while(0);
#define CUDA(x) x
#else
//#define CUDA(x) do { } while(0);
#define CUDA(x) 
#endif // __CUDACC__

// Compile different code for CUDA && C && get a return value (only for 1 expression) (no ; allowed inside)
#ifdef __CUDACC__
#define get_ifCUDA(then_,else_) (then_)
#else 
#define get_ifCUDA(then_,else_) (else_)
#endif // __CUDACC__

// Compile different code for CUDA and C
#ifdef __CUDACC__
//#define ifCUDA(then_,else_) do {then_} while(0);
#define ifCUDA(then_,else_) then_
#else 
//#define ifCUDA(then_,else_) do {else_} while(0);
#define ifCUDA(then_,else_) else_
#endif // __CUDACC__

// Inside _cudaboth_ functions only!! Execute code only on the device or host sides
#ifndef __CUDA__ARCH__
#define HOST  (x) do{x} while(0);
#define DEVICE(x) 
#else
#define HOST  (x) 
#define DEVICE(x) do{x} while(0);
#endif // __CUDACC__





typedef enum cudaLoc { Host, Device, Both } cudaLoc;


// Smaller aliases for the same qualifiers.
#define _cub_  _cudaboth_
#define _cudh_ _cudaboth_
#define _cuhd_ _cudaboth_
#define _cuh_  _cudahost_
#define _cud_  _cudadevice_
#define _cug_  _cudaglobal_

#define _cudab_  _cudaboth_
#define _cudadh_ _cudaboth_
#define _cudahd_ _cudaboth_
#define _cudah_  _cudahost_
#define _cudad_  _cudadevice_
#define _cudag_  _cudaglobal_

#define _cudagC_ _cudaglobalC_



/* Aliases for thread access */
#define tId threadIdx
#define bId blockIdx
#define bDim blockDim
#define gDim gridDim

/// Synchronizes the CUDA device safely
#define cudaSync() cudaSafe(cudaDeviceSynchronize())
/// Same as cudaSync() but doesn't check for errors. Useful in high-performance or debugging kernels, since you can let the error pass to the CudaCheckError() or CudaCheckKernel()
#define cudaUnsafeSync() cudaDeviceSynchronize()
/// Requires C99 or greater but allows execution of commands in a omp master section and syncs all CUDA devices in the end. NVCC isn't great: nvcc fails to compile this macro although it works properly and was tested with the gcc-compiler. DON'T USE UNTIL A NEWER NVCC COMES OUT
#define OMPcudaSync(commands) { _Pragma("omp barrier") _Pragma("omp master")  { commands; cudaSync(); }  _Pragm\
a("omp barrier") }

/**
 * @def cudaMetrics 
 * @brief Measure performance of operations     (USAGE: CALL THE FIRST ARGUMENT WITHIN PARENTHESIS!)
 */
#define cudaMetrics(command_,rw_size_,flops_,iterations_)		\
  {									\
    cout << "cudaMetrics::" << #command_ << endl;			\
      cudaEvent_t a,b;							\
      cudaSafe(cudaEventCreate(&a));					\
      cudaSafe(cudaEventCreate(&b));					\
      cudaSafe(cudaEventRecord(a));					\
      for (int i=0; i < iterations_; ++i) { command_; cudaSync(); }	\
      cudaCheckError("Failed to launch kernel in cudaMetrics");		\
      cudaSafe(cudaEventRecord(b));					\
      cudaSafe(cudaEventSynchronize(b));				\
      float dt_ms;							\
      cudaSafe(cudaEventElapsedTime(&dt_ms, a, b));			\
      cudaSafe(cudaEventDestroy(a));					\
      cudaSafe(cudaEventDestroy(b));					\
      cout << "\tSamples: " << iterations_ << "\n"			\
	   << "\tTime: " << dt_ms/(double)iterations_ << " (ms)\n"	\
	   << "\tGFLOP/s: "<< (double)flops_*(double)iterations_/((double)dt_ms*0.001)/1e9 << " \n" \
	   << "\tBandwidth: " << ((double)rw_size_*(double)iterations_/(double)1073741824)/((double)dt_ms*0.001) << " (GB/s)\n" << endl; /*G=1073741824*/ \
  }

#define cudaMetricsShot(command_,rw_size_,flops_)			\
  {									\
    cout << "cudaMetrics::" << #command_ << endl;			\
      cudaEvent_t a,b;							\
      cudaSafe(cudaEventCreate(&a));					\
      cudaSafe(cudaEventCreate(&b));					\
      cudaSafe(cudaEventRecord(a));					\
      command_; 						\
      cudaSafe(cudaEventRecord(b));					\
      cudaSafe(cudaEventSynchronize(b));				\
      cudaCheckError("Failed to launch kernel in cudaMetrics");		\
      float dt_ms;							\
      cudaSafe(cudaEventElapsedTime(&dt_ms, a, b));			\
      cudaSafe(cudaEventDestroy(a));					\
      cudaSafe(cudaEventDestroy(b));					\
      cout << "\tTime: " << dt_ms << " (ms)\n"				\
	   << "\tGFLOP/s: "<< (double)flops_/((double)dt_ms*0.001)/1e9 << " \n" \
	   << "\tBandwidth: " << ((double)rw_size_/(double)1073741824)/((double)dt_ms*0.001) << " (GB/s)\n" << endl; /*G=1073741824*/ \
  }




/// Warning: Name limited in size! 
typedef struct cuMetrics {
  char name[100];
  int gpu;
  float dt; // (s)
  double BW; //(GB/s)
  double GFLOP; // GFLOP/s
} cuMetrics;

/** WARNING: It calls cudaSetDevice() on each call so you must call it back afterwards to set the GPU in multigpu programs. @see cuMetrics
 * @param[in] e_Start The starting event ( no need to call cudaEventSynchronize())
 * @param[in] e_Stop  The ending event (no need to call cudaEventSynchronize())
 * @param[in] operations How many operations were executed between the events
 * @param[in] op_rw_size The sum of the reads and writes in bytes by operation
 * @param[in] op_flops The count of flops per operation
 * @param[in] print Beyond returning a cuMetrics structure it prints the metrics onto the screen.
 */
#ifdef __CUDACC__
cuMetrics cudaEventMetrics(const std::string &name, int gpu_id, cudaEvent_t e_start, cudaEvent_t e_stop, long double operations, long double op_rw_size, long double op_flops, int print = 0)
{
  float dt_ms;
  cudaSafeSetDevice(gpu_id);
  cudaSafe(cudaEventSynchronize(e_stop));
  cudaSafe(cudaEventElapsedTime(&dt_ms, e_start, e_stop));

  long double rw_size = operations * op_rw_size;
  long double flops   = operations * op_flops;

  long double Gb = 1<<30;
  long double time = (long double)dt_ms*(long double)0.001; // in secs

  cuMetrics metrics;
  
  strncpy(metrics.name, name.c_str(), 100);
  metrics.gpu = gpu_id;
  metrics.BW = rw_size/Gb / time;
  metrics.GFLOP = flops/Gb / time;
  metrics.dt = time;

  if (print)
    std::cout << "CudaEventMetrics::" << name 
	      << "\nTime = " << dt_ms << "(ms)"
	      << "\nBandwidth = " << rw_size/Gb /time << " (GB/s)"
	      << "\nGFLOP/s = "   << flops/Gb   / time  << " (GFLOP/s)\n" << std::endl; 
  return metrics;
}
#endif // __CUDACC__

// __CONSTANT__ AND SYMBOLS MANIPULATION

/// Creates an aligned symbol in constant memory
#define cudaCreateSymbol(type, name) __constant__ __align__(sizeof(type)) type name

/// Copies the variable to constant memory checking if the sizes match. To disable the check: #define NDEBUG
#define cudaSetSymbol(d_symbol, h_symbol) assert(sizeof(d_symbol)==sizeof(h_symbol)); cudaSafe(cudaMemcpyToSymbol(d_symbol, &h_symbol, sizeof(d_symbol)))




/// @def ALIGN Provides compiler-agnostic alignment in NVCC,GCC and MSVC
#if defined(__CUDACC__) // NVCC
   #define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for ALIGN macro for your host compiler!"
#endif


#ifndef NDEBUG
#define Debug(x) std::cout << x << std::endl;
#else
#define Debug(x)
#endif // NDEBUG


#ifdef __CUDACC__
/** Get properties of the GPU
 * @param[in] gpu_id The GPU id of which properties will be fetched. */
struct cudaDeviceProp cudaGetGPUinfo(int gpu_id)
{
  struct cudaDeviceProp GPU;

  cudaSafe(cudaGetDeviceProperties(&GPU, gpu_id));

  printf("\nGPU: %s\n Capability: %d.%d\n\n Regs/SM       = %d\n Threads/Block = %d\n Threads/SM    = %d\n\n", GPU.name, GPU.major, GPU.minor, GPU.regsPerBlock, GPU.maxThreadsPerBlock, GPU.maxThreadsPerMultiProcessor);

  return GPU;
}

/// Get the properties of the active GPU
struct cudaDeviceProp cudaGetGPUinfo()
{
  int gpu_id;
  cudaSafe(cudaGetDevice(&gpu_id));
  
  return cudaGetGPUinfo(gpu_id);
}

/// Fetches the information of all installed graphic cards
void cudaGetListGPUinfo()
{
 int n;
 cudaSafe(cudaGetDeviceCount(&n));
 for (int i = 0; i < n; ++i)
   cudaGetGPUinfo(i);
}


/// Sets the grid size of the kernel call knowing the simulation size and the block dimension. Works in all cases.
dim3 cudaBlocks(dim3 simSize, dim3 blockDim)
{
  // Using ceil() is wrong!
  return make_uint3(simSize.x/blockDim.x + (simSize.x%blockDim.x?1:0),
		    simSize.y/blockDim.y + (simSize.y%blockDim.y?1:0),
		    simSize.z/blockDim.z + (simSize.z%blockDim.z?1:0));
  
}
/// Sets the grid size of the kernel call knowing the simulation size and the block dimension. Works in all cases.
dim3 cudaBlocks(uint simSize, uint blockDim)
{
  // Using ceil() is wrong!
  return make_uint3(simSize/blockDim + (simSize%blockDim?1:0), 1, 1);
}

#endif // __CUDACC__






//http://cplusplus.co.il/2010/07/17/variadic-macro-to-count-number-of-arguments/
//https://gustedt.wordpress.com/2010/06/08/detect-empty-macro-arguments/

// Counts the number of parameters in a variadic macro
//#define VA_NUM_NARGS(...)  (sizeof((int[]){__VA_ARGS__})/sizeof(int))

#ifndef NDEBUG
// Needed for use outside main
template <class T, class T2> // Can be easiliy switched by a C macro
extern void assert_msg_(T cond, T2 msg)
{
  if (! (cond)) 
  { 
    puts(msg); 
    exit(EXIT_FAILURE); 
  }
}
#define ASSERT_MSG(cond, msg) assert_msg_(cond,msg);
#else
#define ASSERT_MSG(cond, msg)
#endif // NDEBUG

#ifdef __CUDACC__

/** @struct cudaKernel_config 
 * 
 * @brief Stores the configuration of a kernel's launch parameters.
 * 
 * After using cudaConfigKernel(), this struct is declared and is used in cuda() to call the kernel without
 * explicitly calling the number of blocks, etc. at each call. Useful if you want to call the kernel in several 
 * places with the same arguments.
 * 
 * */
typedef struct cudaKernel_config{
  
  dim3         blockDim;
  dim3         blocks;
  size_t       smsize;
  int          initialized;
  
} cudaKernel_config;


void cudaKernelConfigPrint(cudaKernel_config k) { printf("gridDim(%d,%d,%d) blockDim(%d,%d,%d) sharedmemsize(%lu)\n", 
			                          k.blocks.x,k.blocks.y,k.blocks.z, k.blockDim.x,k.blockDim.y,k.blockDim.z, k.smsize); }


#define cudaTPrint(kernel,Template,...) printf(#__VA_ARGS__ " Kernel config for " #kernel "<" #Template ">: ");cudaKernelConfigPrint(cuConf_##kernel##_cuKLC_##__VA_ARGS__##template##Template)

/// Catches the configuration struct for the desired kernel
#define getCuKernelConfig(kernel,...) cuConf_##kernel##_cuKLC_##__VA_ARGS__




/** @brief Register CUDA kernels Launch Configurations that can be called through cuda()
 * 
 * @param[in] kernel Kernel name.
 * @param[in] ... Optional UID: Unique ID of the launch configuration for the specific kernel. Useful for different call setups with the same kernel.
 * 
 * Register a Kernel Launch configuration to guarantee that you always launch kernels with the same grid configurations, shared mem and stream id. You can register several configurations for the same kernel with different unique identifiers (UID).  
 * 
 * After registering the kernel configuration you need to configure it with cudaConfigKernel() or cudaConfigKernelUID() and then you just need to call cuda() anywhere.
 * 
 * It is also easier read code this way and you guarantee that you always call the kernels correctly in different parts of the program.
 * 
 * Instead of @code kernel<<< grid_cfg,block_cfg,shared,stream>>>(...)@endcode
 *            you use   @code    cuda (kernel)(...)  @endcode
 *           anywhere.
 * 
 * @see cudaConfigKernel()
 * @see cudaConfigKernelUID()
 * @see cuda()
 */
#define cudaKernel(kernel,...) cudaKernel_config cuConf_##kernel##_cuKLC_##__VA_ARGS__
                  
/// Same as cudaKernel() but allows the kernel to have one template parameter. Configure it with cudaConfigureKernelT() or cudaConfigureKernelTUID().              
#define cudaKernelT(kernel,Template,...) cudaKernel_config cuConf_##kernel##_cuKLC_##__VA_ARGS__##template##Template
                  
                  
/// Configure a kernel launch. You must declare the configuration previously with cudaKernel().
#define cudaConfigKernel(kernel,problem_size ,dim,shmemsize)   \
cuConf_##kernel##_cuKLC_.blockDim = make_uint3(dim) ;		\
cuConf_##kernel##_cuKLC_.blocks   = cudaBlocks(problem_size,dim) ; \
cuConf_##kernel##_cuKLC_.smsize   = shmemsize ;             \
cuConf_##kernel##_cuKLC_.initialized = 1;

/// Same as cudaConfigKernel but when it was declared with a specific UID launch setup (second argument in cudaKernel()).
#define cudaConfigKernelUID(kernel,UID,problem_size ,dim,shmemsize)   \
cuConf_##kernel##_cuKLC_##UID.blockDim = make_uint3(dim) ;		\
cuConf_##kernel##_cuKLC_##UID.blocks   = cudaBlocks(problem_size,dim) ; \
cuConf_##kernel##_cuKLC_##UID.smsize   = shmemsize ;             \
cuConf_##kernel##_cuKLC_##UID.initialized = 1;

/// Same as cudaConfigKernel() but when it was declared with a template. @see cudaKernelT().
#define cudaConfigKernelT(kernel,Template,problem_size,dim,shmemsize) \
cuConf_##kernel##_cuKLC_template##Template.blockDim = make_uint3(dim) ;		\
cuConf_##kernel##_cuKLC_template##Template.blocks   = cudaBlocks(problem_size,dim) ; \
cuConf_##kernel##_cuKLC_template##Template.smsize   = shmemsize ;             \
cuConf_##kernel##_cuKLC_template##Template.initialized = 1;

/// Same as cudaConfigKernelT but when it was declared with a specific UID launch setup (third argument in cudaKernelT()).
#define cudaConfigKernelTUID(kernel,UID,Template,problem_size,dim,shmemsize)\
cuConf_##kernel##_cuKLC_##UID##template##Template.blockDim = make_uint3(dim) ;\
cuConf_##kernel##_cuKLC_##UID##template##Template.blocks   = cudaBlocks(problem_size,dim) ;\
cuConf_##kernel##_cuKLC_##UID##template##Template.smsize   = shmemsize ;\
cuConf_##kernel##_cuKLC_##UID##template##Template.initialized = 1;


/// Additional make_uint3 constructor
inline _cub_ uint3 make_uint3(uint u)
{
  return  make_uint3(u,1,1);
}
inline _cub_ uint3 make_uint3(dim3 d3)
{
 return d3; 
}


/// Same as cudaConfigKernel but declares the kernel launch configuration directly in the local scope. No need to call cudaKernel()
#define cudaLocalConfigKernel(kernel,problem_size ,dim,shmemsize)   cudaKernel(kernel); cudaConfigKernel(kernel,problem_size,dim,shmemsize)


/** Call a kernel with a registered Launch configuration ( you can use cudaConfigKernel() )
 * @param[in] kernel Kernel name.
 * @param[in] ... UID Optional: Unique ID of the kernel launch configuration.
 * 
 * It makes sure you have configured previously the kernel launch. There is an overhead in this check that goes away if you @b #define @b NDEBUG.
 *
 * If you use an UID then you must use cudaConfigKernelUID() instead of cudaConfigKernel().
 * 
 * @see cudaKernel()
 * @see cudaConfigKernel()
 * @see cudaConfigKernelUID()
 * 
 */
#define cuda(kernel, ...) \
ASSERT_MSG(cuConf_##kernel##_cuKLC_##__VA_ARGS__.initialized, "ERROR: No kernel configuration available for kernel " #kernel ", use cudaConfigKernel().");\
kernel<<< cuConf_##kernel##_cuKLC_##__VA_ARGS__.blocks, cuConf_##kernel##_cuKLC_##__VA_ARGS__.blockDim, cuConf_##kernel##_cuKLC_##__VA_ARGS__.smsize>>>

#define streamcuda(stream,kernel,...)\
ASSERT_MSG(cuConf_##kernel##_cuKLC_##__VA_ARGS__.initialized, "ERROR: No kernel configuration available for kernel " #kernel ", use cudaConfigKernel().");\
kernel<<< cuConf_##kernel##_cuKLC_##__VA_ARGS__.blocks, cuConf_##kernel##_cuKLC_##__VA_ARGS__.blockDim, cuConf_##kernel##_cuKLC_##__VA_ARGS__.smsize, stream>>>


/// Same as cuda() but for 1-parameter template kernels. @see cudaKernelT() @see cudaConfigKernelT() @see cudaConfigKernelTUID()
#define cudaT(kernel, Template,...) \
ASSERT_MSG(cuConf_##kernel##_cuKLC_##__VA_ARGS__##template##Template.initialized, "ERROR: No kernel configuration available for kernel " #kernel "<"#Template">, use cudaConfigKernelT().");\
kernel<Template><<<cuConf_##kernel##_cuKLC_##__VA_ARGS__##template##Template.blocks, cuConf_##kernel##_cuKLC_##__VA_ARGS__##template##Template.blockDim, cuConf_##kernel##_cuKLC_##__VA_ARGS__##template##Template.smsize>>>

#define streamcudaT(stream,kernel,Template,...)\
ASSERT_MSG(cuConf_##kernel##_cuKLC_##__VA_ARGS__##template##Template.initialized, "ERROR: No kernel configuration available for kernel " #kernel "<"#Template">, use cudaConfigKernelT().");\
kernel<Template><<<cuConf_##kernel##_cuKLC_##__VA_ARGS__##template##Template.blocks, cuConf_##kernel##_cuKLC_##__VA_ARGS__##template##Template.blockDim, cuConf_##kernel##_cuKLC_##__VA_ARGS__##template##Template.smsize, stream>>>


#endif // __CUDACC__

#ifdef __CUDACC__
/// Resets all devices with cudaDeviceReset() and settings contexts safely
void cudaDeviceResetAll()
{
  int nGPUs;
  cudaGetDeviceCount(&nGPUs);
  
  int i;
  for (i = 0; i < nGPUs; ++i)
  {
    cudaSafe(cudaSetDevice (i));
    cudaSafe(cudaDeviceReset());
  }
}
#endif // __CUDACC__



#ifdef __CUDACC__
/** Safely enables Peer Access between all device pairs that support it. Notice it might not be bidirectional. @warning Make sure you call cudaDeviceResetAll() in the end or cudaDeviceReset() for all the GPUs on which PeerAccess is enabled or deactivate peer access. It is unsafe in parallel regions!
 *
 * @param[in] default_gpu_id Since it is unsafe in context, you can pass the gpu_id to which you want to return at the end of the execution. By default it is 0.
 *
 * @see cudaDeviceEnablePeerAccess() from NVIDIA docs
 */
void cudaDeviceEnablePeerAccessAll(int default_gpu_id = 0)
{
  int nGPU;
  cudaSafe(cudaGetDeviceCount(&nGPU));
  int i,j;
  int canAccessPeer;

  for (i = 0; i < nGPU; ++i)
    {
      cudaSafe(cudaSetDevice(i));
      for (j = 0; j < nGPU; ++j)
	{
	  if (j==i)
	    continue; // same card

	  cudaSafe(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
	  if (canAccessPeer)
	    cudaSafe(cudaDeviceEnablePeerAccess(j, 0));
	}
    }
  cudaSafe(cudaSetDevice(default_gpu_id));
}
#endif // __CUDACC__

int min_int(int a, int b) // We could write it as a macro but that doesn't have type safety without __typeof__ operators that bring subtle bugs
{
  if (a < b)
    return a;
  return b;
}
int max_int(int a, int b) // We could write it as a macro but that doesn't have type safety without __typeof__ operators that bring subtle bugs
{
  if (a > b)
    return a;
  return b;
}
#ifdef __CUDACC__
/// Choose smaller or bigger block sizes that have the 
typedef enum SearchBlockSizeStrategy { ChooseSmallest, ChooseLargest } SearchBlockSizeStrategy; 

// We still want to mantain compatibility with C so no #include'ing <algorithm> // std::min and max()
//#include <algorithm> // std::min and std:max

/** Finds the best block size for the selected GPU // Use cudaFuncGetAttributes(&A,kernel_name) and A.numRegs to know your kernel reg usage
 *
 * @param[in] total_threads If total_threads is specified the block size will be also limited by that dimension (otherwise it is limited only byt the GPU's max block size)
 */
uint cudaBestBlockSize1D(int regs_per_thread, SearchBlockSizeStrategy strategy = ChooseLargest, int total_threads = 0)
{
  int warpSize = 32;
  int SM_BLOCKS = 8; // Cuda's magic number: you can't get more than 8 blocks running at the same time (when searching for maximum occupancy it seems to be impossible to surpass this value anyways.)

  int gpu_id;
  cudaSafe(cudaGetDevice(&gpu_id));
  struct cudaDeviceProp GPU;
  cudaSafe(cudaGetDeviceProperties(&GPU, gpu_id));
  
  // GPU characteristics
  int SM_REGS = GPU.regsPerBlock; // NVIDIA's misnomer: it's really the number of regs per SM
  int SM_THREADS = GPU.maxThreadsPerMultiProcessor; 
  int BLOCK_THREADS = GPU.maxThreadsPerBlock; 

  int max_threads_per_SM = min_int(SM_THREADS, SM_REGS/regs_per_thread); // This is what limits the occupation (each architecture has it's limit of regs/thread below which we can run at 100% occupancy)

  // Start searching for the best size combination
  int best_blockDimX = warpSize; // initial guess
  int best_threads_per_SM = 1;
  int best_blocks_per_SM = 1;
  for (int blockDimX = warpSize; blockDimX <= BLOCK_THREADS; blockDimX += warpSize)
  {
    int blocks_per_SM = min_int(SM_BLOCKS,max_threads_per_SM/blockDimX);
    int threads_per_SM = blocks_per_SM * blockDimX;
    printf("threads/sm=%5d      blockdim=%5d      blocks/sm=%d      occupancy=%5.1f%%\n",max_threads_per_SM, blockDimX, blocks_per_SM, threads_per_SM/float(SM_THREADS)*100.0);

    // If the total number of threads is specified we assume there's a strict block size to follow and do not create extra threads to increase occupancy.
    if (!total_threads || blockDimX <= total_threads) // We could put this in the for condition itself but that way we couldn't print all the hypothesis.
    {
      if (strategy == ChooseSmallest)
      {
        if (threads_per_SM > best_threads_per_SM) // strategy only conditions the "if"
        {
          best_blockDimX = blockDimX;
          best_threads_per_SM = threads_per_SM;

          // curiosity
          best_blocks_per_SM = blocks_per_SM;
        }
      }
      else // ChooseLargest
      {
        if (threads_per_SM >= best_threads_per_SM) // this condition changes
        {
          best_blockDimX = blockDimX;
          best_threads_per_SM = threads_per_SM;

          // curiosity
          best_blocks_per_SM = blocks_per_SM;
        } 
      }  
    }
  }

  // Registos   (threads/sm)(regs)    threads/sm    dim.y   Dim. Bloco    blocks/sm   Blocos    Ocupação (\%)

  printf("cudaBestBlock ::: threads/sm=%d  blockdim=%d  blocks/sm=%d  occupancy=%.1f%%\n",max_threads_per_SM, best_blockDimX, best_blocks_per_SM, best_threads_per_SM/(float)SM_THREADS*100.0);
  return best_blockDimX;
}

/// If you use this use always!! boundary checks
dim3 cudaBestBlockSize2D(int regs_per_thread, SearchBlockSizeStrategy strategy = ChooseLargest)
{
  int warpSize = 32;
  int SM_BLOCKS = 8; // Cuda's magic number: you can't get more than 8 blocks running at the same time (when searching for maximum occupancy it seems to be impossible to surpass this value anyways.)

  int gpu_id;
  cudaSafe(cudaGetDevice(&gpu_id));
  struct cudaDeviceProp GPU;
  cudaSafe(cudaGetDeviceProperties(&GPU, gpu_id));
  
  // GPU characteristics
  int SM_REGS = GPU.regsPerBlock; // NVIDIA's misnomer: it's really the number of regs per SM
  int SM_THREADS = GPU.maxThreadsPerMultiProcessor; 
  int BLOCK_THREADS = GPU.maxThreadsPerBlock; 

  int max_threads_per_SM = min_int(SM_THREADS, SM_REGS/regs_per_thread); // This is what limits the occupation (each architecture has it's limit of regs/thread below which we can run at 100% occupancy)

  // Start searching for the best size combination
  int best_blockDimY = 1; // initial guess
  int best_threads_per_SM = 1;
  int best_blocks_per_SM = 1;

  int blockDimX = warpSize;
  for (int blockDimY = 1; blockDimX*blockDimY <= BLOCK_THREADS; ++blockDimY)
  {
    int blocks_per_SM = min_int(SM_BLOCKS,max_threads_per_SM/(blockDimX*blockDimY));
    int threads_per_SM = blocks_per_SM * (blockDimX*blockDimY);
    printf("threads/sm=%5d   blockdimY=%5d   blockdim=%5d      blocks/sm=%d      occupancy=%5.1f%%\n",max_threads_per_SM, blockDimY, blockDimX*blockDimY, blocks_per_SM, threads_per_SM/float(SM_THREADS)*100.0);

    if (strategy == ChooseSmallest)
    {
      if (threads_per_SM > best_threads_per_SM) // strategy only conditions the "if"
      {
        best_blockDimY = blockDimY;
        best_threads_per_SM = threads_per_SM;

        // curiosity
        best_blocks_per_SM = blocks_per_SM;
      }
    }
    else // ChooseLargest
    {
      if (threads_per_SM >= best_threads_per_SM) // this condition changes
      {
        best_blockDimY = blockDimY;
        best_threads_per_SM = threads_per_SM;

        // curiosity
        best_blocks_per_SM = blocks_per_SM;
      } 
    }  
  
  }

  // Registos   (threads/sm)(regs)    threads/sm    dim.y   Dim. Bloco    blocks/sm   Blocos    Ocupação (\%)

  printf("cudaBestBlock ::: threads/sm=%d  blockdimY=%d blockdim=%d  blocks/sm=%d  occupancy=%.1f%%\n",max_threads_per_SM, best_blockDimY, blockDimX*best_blockDimY, best_blocks_per_SM, best_threads_per_SM/(float)SM_THREADS*100.0);
  return dim3(blockDimX,best_blockDimY);
}
#endif // __CUDACC__




#endif  // cuda_extend_H__

