/**

*****************************
Programmer: Alberto Ferreira
Project start-Date: 22/11/2012

License: GPL2

Disclaimer: You may use the code as long as you give credit to the 
copyright holder, Alberto Ferreira.
*****************************

The matrix is stored unidimensionally with malloc to avoid calling the 
constructors automatically so that the default value initialization is faster.
Changing to new is therefore considered a regression, only placement new would
be accepted.

*/

#ifndef MATRIX_DECLARATION_H__
#define MATRIX_DECLARATION_H__

#ifndef MATRIX_H__
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#else
#include <iosfwd> // You must still include iostream afterwards.
#endif

#include <string.h> // memcpy

#include "custom_assert.h"

namespace MatrixAlloc
{
  enum Mode { Rows, Cols };
}

// Templates for friend functions
template <class T, MatrixAlloc::Mode alloc_mode = MatrixAlloc::Rows> class Matrix;
template <class T, MatrixAlloc::Mode alloc_mode> std::ostream &operator << (std::ostream &, Matrix<T,alloc_mode> &);
template <class T, MatrixAlloc::Mode alloc_mode> std::istream &operator >> (std::istream &, Matrix<T,alloc_mode> &);



//template <class T> Matrix<float> abs_matrix (Matrix<T> &);

template <class T, MatrixAlloc::Mode alloc_mode>
class Matrix
{
  // returns a matrix which is the abs of this one
  //friend  Matrix<float> abs_matrix <>(Matrix<T> &); 
  
  friend std::ostream &operator << <>(std::ostream &, Matrix<T,alloc_mode> &);
  friend std::istream &operator >> <>(std::istream &, Matrix<T,alloc_mode> &);
 public:

  Matrix (size_t rows, size_t cols, T default_values = 0);
  Matrix (const Matrix<T,alloc_mode> & copy);
  Matrix (const Matrix<T,alloc_mode> * copy);
  ~Matrix ();

  T & operator () (size_t row, size_t col);
  T operator ()  (size_t row, size_t col) const;
  inline T * operator ()  () { return m; }
  inline T * raw() const {return m; }
  inline T * row(size_t row);
  inline T * col(size_t row);

  inline size_t rows() const { return m_rows; }
  inline size_t cols() const { return m_cols; }
  inline size_t size() const { return m_size; }

  inline void clear() { for (size_t i=0;i<m_size;++i)m[i]=0; }

  // returns a row or column according to the matrix alloc_mode
  inline T * operator () (size_t main_dimension_pos); 
  // Length of the main dimension ( number of elements after accessing Matrix(i) )
  inline size_t d() const { if (alloc_mode == MatrixAlloc::Rows) return m_cols; else return m_rows; }
  // Number of main entries available (rows in MatrixAlloc::Rows) ( Matrix(i) is valid for all i < Matrix.n() )
  inline size_t n() const { if (alloc_mode == MatrixAlloc::Rows) return m_rows; else return m_cols; }

  Matrix       & operator  = (const Matrix<T,alloc_mode> &);
  const Matrix & operator += (const Matrix<T,alloc_mode> &);
  const Matrix   operator +  (const Matrix<T,alloc_mode> &);
  const Matrix   operator *  (const Matrix<T,alloc_mode> &);

  const Matrix & operator *= (const T factor);
  const Matrix & operator /= (const T factor);

  void max_index(size_t &row, size_t &col, size_t max_row_index, size_t max_col_index);
  void min_index(size_t &row, size_t &col, size_t max_row_index, size_t max_col_index);

  void fill_row_with(size_t row, T value);
  void fill_col_with(size_t col, T value);
    
  void print(size_t rows, size_t cols);

 private:
  const size_t m_rows, m_cols, m_size;
  T *m;
  T _default_value;
};

 
#endif // MATRIX_H
