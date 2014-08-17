/*
 * This is a special version of Nuno Cardoso's version of CUDA's SDK reduction,
 * plus some missing files from CUDA's SDK reduction
 * packaged into a single header by Alberto Ferreira so it can be used with a 
 * simple #include in architectures < 2.x (using a separate .o isn't possible).
 *
 * The date of modification was 5/08/2013.
 *
 * Template specializations were removed in the header to improve the compilation time.
 * 
 *  Usage: 
 *     * reduction(array_ptr, array_size);
 *     * reduction_wstream(array_ptr, array_size, cuda_stream); // using streams
 *
 * Minimal version: Doesn't support int2, etc., only standard floating-point (float, double, long double, ...)
 */ 

#ifndef MONOLITHIC_SDK_REDUCTION_H_

#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cassert>

#include <vector_types.h>
 
#include "cuda_common.h"

// reduction.h @ SDK 
template <class T>
void reduce(int size, int threads, int blocks,
            int whichKernel, T *d_idata, T *d_odata);


// reduction_kernel.cuh
template <class T>
void call_reduce(int size, int threads, int blocks, 
                 int whichKernel, T *d_idata, T *d_odata);
template <class T>
void call_reduce_wstream(int size, int threads, int blocks, 
                 int whichKernel, T *d_idata, T *d_odata, cudaStream_t stream);

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */







#define EMUSYNC __syncthreads()

inline void reduce_check_error_and_sync()
{
    CUT_CHECK_ERROR("Reduce: Kernel execution failed");
    CUDA_SAFE_CALL(cudaThreadSynchronize());
}

template<class T>
inline __device__ T zzero(){
/*    T res;
    return res;
    */
    return 0;
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};


/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory. 
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.  
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum = zzero<T>();

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        mySum += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            mySum += g_idata[i+blockSize];  
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}


extern "C"
bool isPow2(unsigned int x){
    return ((x&(x-1))==0);
}


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void 
call_reduce(int size, int threads, int blocks, 
       int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
            switch (threads)
            {
            case 512:
                reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 256:
                reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 128:
                reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 64:
                reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 32:
                reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 16:
                reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  8:
                reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  4:
                reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  2:
                reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  1:
                reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            }
        }
        else
        {
            switch (threads)
            {
            case 512:
                reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 256:
                reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 128:
                reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 64:
                reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 32:
                reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 16:
                reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  8:
                reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  4:
                reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  2:
                reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  1:
                reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            }       
    }
}

            
              
              
////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void 
call_reduce_wstream(int size, int threads, int blocks, 
       int whichKernel, T *d_idata, T *d_odata, cudaStream_t stream)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
            switch (threads)
            {
            case 512:
                reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 256:
                reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 128:
                reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 64:
                reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 32:
                reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 16:
                reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  8:
                reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  4:
                reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  2:
                reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  1:
                reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            }
        }
        else
        {
            switch (threads)
            {
            case 512:
                reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 256:
                reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 128:
                reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 64:
                reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 32:
                reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case 16:
                reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  8:
                reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  4:
                reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  2:
                reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            case  1:
                reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size); 
                    reduce_check_error_and_sync(); break;
            }       
    }
}







#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads){ 
    if (whichKernel < 3){
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else{
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if (whichKernel == 6)
        blocks = MIN(maxBlocks, blocks);
}

///////////////////// Call Sum reduction ////////////////////////////////////////////////////////////////////
template <class T> 
T reduction(T *array_d, int size){

	T result;	
	int kernel = 6;
	int s = size;
	int maxThreads = 256;
	int maxBlocks = 64;
	int blocks = 0;
	int threads = 0;

	getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);
	call_reduce<T>(s, threads, blocks, kernel, array_d, array_d);

	// sum partial block sums on GPU
    	s = blocks;
    	while( s > 1 ) {
        	int threads = 0;
		int blocks = 0;
       	 	getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);        
        	call_reduce<T>(s, threads, blocks, kernel, array_d, array_d);

		if (kernel < 3)
                    s = (s + threads - 1) / threads;
                else
                    s = (s + (threads * 2 - 1)) / (threads * 2);
    	}
	//Copy SUM From GPU to CPU
	CUDA_SAFE_CALL( cudaMemcpy( &result, array_d, sizeof(T), cudaMemcpyDeviceToHost) );
	return result;
}




template <class T> 
T reduction_wstream(T *array_d, int size, cudaStream_t stream ){

	T result;	
	int kernel = 6;
	int s = size;
	int maxThreads = 256;
	int maxBlocks = 64;
	int blocks = 0;
	int threads = 0;

	getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);
	call_reduce_wstream<T>(s, threads, blocks, kernel, array_d, array_d, stream);

	// sum partial block sums on GPU
    	s = blocks;
    	while( s > 1 ) {
        	int threads = 0;
		int blocks = 0;
       	 	getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);        
        	call_reduce_wstream<T>(s, threads, blocks, kernel, array_d, array_d, stream);

		if (kernel < 3)
                    s = (s + threads - 1) / threads;
                else
                    s = (s + (threads * 2 - 1)) / (threads * 2);
    	}
	//Copy SUM From GPU to CPU
	CUDA_SAFE_CALL( cudaMemcpyAsync( &result, array_d, sizeof(T), cudaMemcpyDeviceToHost, stream) );
	return result;
}



#endif // MONOLITHIC_SDK_REDUCTION_H_
