/**

  *****************************
    Programmer: Alberto Ferreira
    Project start-Date: 22/11/2012

    License: GPL2

    Disclaimer: You may use the code as long as you give credit to the 
              copyright holder, Alberto Ferreira.
  *****************************
  

  Note: This library provides two modes. It is compatible with CUDA and OpenMP.
    The library transparently compiles in different forms if the CUDA compiler
    is used, which means that you get no extra baggage if you use it only for 
    CPU purposes.
        Automatically, if OpenMP is available, the library will use it in 
    parallelizable operations.

        There is automatic memory management for CUDA too, in fact you can 
    initialize the matrices in the Host, Device, or Both. If using the CUDA
    compiler, you get access to new functions that allow you to synchronize
    automatically memory from the host to the device and vice-versa.
        If you're using multi-gpu make sure you call the initializer after
    choosing the desired GPU with cudaSetDevice(gpu) or the device memory
    will be allocated elsewhere.

        The matrix is stored unidimensionally with malloc to avoid calling the 
    constructors automatically so that the default value initialization is faster.
    Changing to new is therefore considered a regression, only placement new would
    be accepted.
*/

#ifndef MATRIX_H
#define MATRIX_H

#include "cuda_common.h"
#include "cuda_extend.h"

#include "safe.h"

#include <iostream>
#include <cassert>
#include <omp.h>
#include <stdlib.h>



// Templates das funcoes friend
template <class T> class Matrix;
template <class T> std::ostream &operator << (std::ostream &, const Matrix<T> &);
template <class T> std::istream &operator >> (std::istream &, Matrix<T> &);

template <class T> Matrix<float> abs_matrix (Matrix<T> &);

template <class T>
class Matrix
{
  // returns a matrix which is the abs of this one
  friend  Matrix<float> abs_matrix <>(Matrix<T> &); 
  
  friend std::ostream &operator << <>(std::ostream &, const Matrix<T> &);
  friend std::istream &operator >> <>(std::istream &, Matrix<T> &);
 public:

 Matrix() : h_m(NULL),m_size(0),m_cols(0),m_rows(0) {CUDA(d_m=NULL;cudaloc=Host;)}
  Matrix (unsigned int cols, unsigned int rows, T default_values, cudaLoc = Host, bool SkipDeviceInitialize = 0);
  //  Matrix (const Matrix<T> & copia, cudaSyncType cudasync = SyncAll);
  Matrix (const Matrix<T> & copia);
  ~Matrix ();

  void syncToHost   (bool avoid_sync = 0);
  void syncToDevice (bool avoid_sync = 0);


  inline T & operator () (uint row, uint col);

  inline unsigned int rows() const { return m_rows; }
  inline unsigned int cols() const { return m_cols; }
  inline unsigned int size() const { return m_size; }

  Matrix       & operator  = (const Matrix<T> &);
  const Matrix & operator += (const Matrix<T> &);
  const Matrix   operator +  (const Matrix<T> &);
  const Matrix   operator *  (const Matrix<T> &);
    
  inline T * RAW()   
  { 
    //    ifCUDA(
#ifdef __CUDACC__
    if (cudaloc == Host) 
      return h_m; 
    else if (cudaloc == Device)
      return d_m;
    
    std::cerr << "Matrix::RAW(): Matrix was declared in cudaLoc:Both\n"
	      << "Use h_RAW() OR d_RAW() " << std::endl;
    return NULL;
#else	   
    return h_m;
#endif
  }      
  
  inline T * h_RAW() { return h_m; }
#ifdef __CUDACC__
  inline T * d_RAW() { return d_m; }
#endif
    
 private:

  unsigned int m_rows, m_cols;
  T *h_m; // standard matrix behaviour if CUDA is not available
  uint m_size;

  CUDA(cudaLoc cudaloc; T * d_m;)
};

       
       
CUDA(
     template <class T> // call me with 1 thread only!
     _cudaglobal_ void _init_matrix_on_device (T *d_m, unsigned int cols, uint rows, T default_value)
     {
       uint i = tId.x + bId.x*bDim.x;
       uint j = tId.y + bId.y*bDim.y;

       if (i < cols && j < rows)
	 d_m[i+cols*j] = default_value;
     }
     )

template <class T>
Matrix<T>::Matrix (unsigned int cols, unsigned int rows, T default_value, cudaLoc loc, bool SkipDeviceInitialize) 
: h_m(NULL), m_rows(rows), m_cols(cols), m_size(rows*cols)
{
  if (! m_size)
    {
      std::cerr << "Matrix::Matrix error: You initialized a matrix with no size at all!" << std::endl;
      exit(1);
    }

#ifdef __CUDACC__
  /* CUDA-enabled code */
	d_m = NULL;
	cudaloc = loc;

	 if (cudaloc != Host)   // Device
	   {
	     cudaSafe ( cudaMalloc((void**)&d_m, m_size*sizeof(T)) );
	     if (! SkipDeviceInitialize)
	       {
		 dim3 matrixSize(cols,rows), blockSize(32,4);
		 _init_matrix_on_device<<<cudaBlocks(matrixSize,blockSize),blockSize>>>(d_m, cols, rows, default_value);
		 cudaUnsafeSync();	     
		 cudaCheckError ("Matrix::Matrix(): Device allocation error!");
	       }
	   }
	       
	 if (cudaloc != Device) // Host
	   {
	     h_m = (T*)malloc(m_size*sizeof(T));

	     if (h_m == NULL)
	       {
		 std::cerr << "RAM Full. Allocation Failed!" << std::endl;
		 exit(1);
	       }

	     for (uint i = 0; i < m_size; ++i)
	       h_m[i] = default_value;
	   }

#else
	 /* No CUDA available, 
	    only host is available. 
	    Must allocate there! */

	 h_m = (T*)malloc(m_size*sizeof(T));
 
	 if (h_m == NULL)
	       {
		 std::cerr << "RAM Full. Allocation Failed!" << std::endl;
		 exit(1);
	       }

	 for (uint i = 0; i < m_size; ++i)
	   h_m[i] = default_value;
#endif
}

#ifdef __CUDACC__
template <class T>
void Matrix<T>::syncToDevice(bool avoid_sync)
{
  if (cudaloc != Both)
    {
      std::cerr << "Matrix::syncToDevice(): Matrix is not allocated on Both!" 
		<< std::endl;
      exit(2);
    }
  if (! avoid_sync)
    cudaSync();
  cudaSafe(cudaMemcpy(d_m, h_m, m_size*sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
void Matrix<T>::syncToHost(bool avoid_sync)
{
  if (cudaloc != Both)
    {
      std::cerr << "Matrix::syncToHost(): Matrix is not allocated on Both!" 
		<< std::endl;
      exit(2);
    }
  if (! avoid_sync)
    cudaSync();
  cudaSafe(cudaMemcpy(h_m, d_m, m_size*sizeof(T), cudaMemcpyDeviceToHost));
}
#endif // __CUDACC__



template <class T> 
Matrix<T>::Matrix (const Matrix<T> & copia) // USES MALLOC TO AVOID CALLING THE CONSTRUCTORS RIGHT AWAY
:m_size(copia.size()), m_rows(copia.rows()), m_cols(copia.cols()), h_m(NULL)
{
  if (! copia.size())
    {
      std::cerr << "Matrix::Matrix(&copy): You are trying to initialize a matrix from an unitialized matrix!" << std::endl;
      exit(1);
    }


  #ifdef __CUDACC__
  // CUDA-specific code

  cudaloc = copia.cudaloc;

  // Device copy
  if ( cudaloc != Host)
    {
      cudaSafe ( cudaMalloc((void**)&d_m, m_size*sizeof(T)) );
      cudaSafe(cudaMemcpy(d_m, copia.d_m, m_size*sizeof(T), cudaMemcpyDeviceToDevice));
    }
  // Host copy
  if ( cudaloc != Device )
    {
      //  h_m = new T[copia.size()]; // Either we start to use placement new or we will mantain malloc
      h_m = (T*) malloc(m_size*sizeof(T));
      if (h_m == NULL)
	{
	  std::cerr << "RAM Full. Allocation Failed!" << std::endl;
	  exit(1);
	}

      #pragma parallel for
      for (uint i = 0; i < m_size; ++i)
	h_m[i] = copia.h_m[i];
    }


  #else
  // No CUDA available
  //  h_m = new T[copia.size()]; // Either we start to use placement new or we will mantain malloc
  h_m = (T*) malloc(m_size*sizeof(T));
  if (h_m == NULL)
    {
      std::cerr << "RAM Full. Allocation Failed!" << std::endl;
      exit(1);
    }
  // No point in not syncing anything
  #pragma parallel for
  for (uint i = 0; i < m_size; ++i)
    h_m[i] = copia.h_m[i];
  
  #endif

}


template <class T> 
Matrix<T>::~Matrix ()
{
  ifCUDA(
	 /* CUDA-enabled code */
	 if (cudaloc != Host)
	   cudaSafe(cudaFree(d_m));
	 if (cudaloc != Device)
	   free(h_m);
	 ,
	 /* Regular code */
	 free(h_m);
	 );
}


template <class T> 
T & Matrix<T>::operator () (uint row, uint col)
{
  // assegura que nao acedemos a memoria fora da matriz
  assert (row < m_rows && col < m_cols);

  return h_m[row + m_rows*col];
}

template <class T>
std::ostream &operator << (std::ostream &output, const Matrix<T> &matrix)
{
  CUDA(
    /*CUDA-enabled code */
    if (matrix.cudaloc == Device) 
    {
      std::cerr << "ERROR! Matrix::operator<< used in a matrix which is not on the Host!" << std::endl;
      return output;
    }  
  )
    
  for (uint row=0; row < matrix.m_rows ; ++row)
    {
      for (uint col=0; col < matrix.m_cols; ++col)
        output << matrix.h_m[col+matrix.m_cols*row] << "\t"; 
      output << "\n";
    }

  output << std::endl;
        

  return output; // allows cout << matrix1 << matrix2
}
/*
template <class T>
std::istream &operator >> (std::istream &input, Matrix<T> &matrix)
{
  unsigned int row, col;

  for (row=0; row < matrix.m_rows ; ++row)
    {
      for (col=0; col < matrix.m_cols; ++col)
        input >> matrix.matrix[row][col];
    }

  return input; // permite cin >> matrix1 >> matrix2
}
*/

 
template <class T>
Matrix<T> & Matrix<T>::operator = (const Matrix<T> &m)
{
  

  // Safety checks

  if (this == &m)
    return *this;


  if (m_size && m_size != m.m_size || ! m.m_size CUDA(|| cudaloc != m.cudaloc)); 
  {
    std::cerr << "Matrix::operator= : Invalid operation. Matrices size OR cudaloc don't match OR you're using unitialized matrices." << std::endl;
    exit(1);
    //      return *this;
  }
  
  m_size = m.m_size;

  #ifdef __CUDACC__
  /* CUDA-specific code */
  
  // Device copy
  if ( cudaloc != Host)
    {
      if (! m_size)
	cudaSafe ( cudaMalloc((void**)&d_m, m_size*sizeof(T)) );
      cudaSafe(cudaMemcpy(d_m, m.d_m, m_size*sizeof(T), cudaMemcpyDeviceToDevice));
    }
  // Host copy
  if ( cudaloc != Device )
    {
      //  h_m = new T[copia.size()]; // Either we start to use placement new or we will mantain malloc
      if (! m_size)
	{
	  h_m = (T*) malloc(m_size*sizeof(T));
	  if (h_m == NULL)
	    {
	      std::cerr << "RAM Full. Allocation Failed!" << std::endl;
	      exit(1);
	    }
	}
      #pragma parallel for
      for (uint i = 0; i < m_size; ++i)
	h_m[i] = m.h_m[i];
    }
#else
  /* No CUDA available */
  if (! m_size)
    {
      h_m = (T*) malloc(m_size*sizeof(T));
      if (h_m == NULL)
	{
	  std::cerr << "RAM Full. Allocation Failed!" << std::endl;
	  exit(1);
	}
    }
  #pragma omp parallel for 
  for (uint i = 0; i < m_size; ++i)
    h_m[i] = m.h_m[i];
#endif
	 
  return *this;
}
 
/*
template <class T>
const Matrix<T> & Matrix<T>::operator += (const Matrix<T> &a)
  {
    unsigned int i,j;

    for (i=0; i < rows() ; i++)
      for (j=0; j < cols(); j++)
	matrix[i][j] += a.matrix[i][j];

    return *this;
  }


template <class T>
const Matrix<T> Matrix<T>::operator + (const Matrix<T> &a)
{
  unsigned int i,j;
  Matrix<T> b(rows(),cols());

  for (i=0; i<rows()  ; ++i)
    for (j=0; j<cols() ; ++j)
      b.matrix[i][j] = (T) ( matrix[i][j] + a.matrix[i][j] );

  return b;
}




template <class T>
const Matrix<T> Matrix<T>::operator *(const Matrix<T> &t)
{
  assert( m_cols == t.m_rows);

  unsigned int i,j,k;
  Matrix<T> b(m_rows,t.m_cols);

  for ( i=0; i < rows(); i++)
    for ( j=0; j <t.m_cols; j++)
      {
        b.matrix[i][j]=0;
        for (k=0; k<cols();k++)
          b.matrix[i][j] += matrix[i][k]*t.matrix[k][j];
      }

  return b;
}
*/

template <class T> Matrix<float> abs_matrix(Matrix<T> &m)
{
  Matrix<float> modulo(m.m_rows,m.m_cols,0);
  

#pragma omp parallel for
  for (uint i=0; i < m.m_rows * m.m_cols; ++i)
      modulo.h_m[i] = (m.h_m[i]).abs();

  return modulo;
}





#endif
