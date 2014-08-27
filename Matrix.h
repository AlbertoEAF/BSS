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

#ifndef MATRIX_H
#define MATRIX_H


#include <iosfwd> // You must still include iostream afterwards.
#include <string.h> // memcpy

#include "safemem.h"
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

  inline T & operator () (size_t row, size_t col);
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

  void max_index(size_t &row, size_t &col);
  void min_index(size_t &row, size_t &col);
    
  void print(size_t rows, size_t cols);

 private:
  const size_t m_rows, m_cols, m_size;
  T *m;
  T _default_value;
};

       
       
template <class T, MatrixAlloc::Mode alloc_mode>
Matrix<T,alloc_mode>::Matrix (size_t rows, size_t cols, T default_value) 
  : m_rows(rows), m_cols(cols), m_size(rows*cols), m(NULL), _default_value(default_value)
{
  Guarantee(m_size, "Initializing matrix with size 0 in at least one of the dimensions.");

  m = (T*)safe_malloc(m_size*sizeof(T));

  for (size_t i = 0; i < m_size; ++i)
    m[i] = default_value;
}


template <class T, MatrixAlloc::Mode alloc_mode> 
Matrix<T,alloc_mode>::Matrix (const Matrix<T,alloc_mode> & copy) 
  : m_rows(copy.rows()), m_cols(copy.cols()), m_size(copy.size()), m(NULL), _default_value(copy._default_value)
{
  Assert(copy.size(), "Copying matrix with size 0 in at least one of the dimensions!");

  // USES MALLOC TO AVOID CALLING THE CONSTRUCTORS OF EACH ENTRY RIGHT AWAY
  m = (T*) safe_malloc(m_size*sizeof(T));
  memcpy((void*)m, (void*)copy.m, m_size*sizeof(T));
}


template <class T, MatrixAlloc::Mode alloc_mode> 
Matrix<T,alloc_mode>::Matrix (const Matrix<T,alloc_mode> * copy) 
  : m_rows(copy->rows()), m_cols(copy->cols()), m_size(copy->size()), m(NULL), _default_value(copy->_default_value)
{
  Assert(copy->size(), "Copying matrix with size 0 in at least one of the dimensions!");

  // USES MALLOC TO AVOID CALLING THE CONSTRUCTORS OF EACH ENTRY RIGHT AWAY
  m = (T*) safe_malloc(m_size*sizeof(T));
  memcpy((void*)m, (void*)copy->m, m_size*sizeof(T));
}

template <class T, MatrixAlloc::Mode alloc_mode> 
Matrix<T,alloc_mode>::~Matrix ()
{
  free(m);
}

template <class T, MatrixAlloc::Mode alloc_mode> 
T & Matrix<T,alloc_mode>::operator () (size_t row, size_t col)
{
  Assert (row < m_rows && col < m_cols, "Out of bounds access (%lu,%lu) for Matrix(%lu,%lu) !",row,col,m_rows,m_cols);
  
  if (alloc_mode == MatrixAlloc::Rows)
    return m[col + m_cols*row];
  else
    return m[row + m_rows*col];
}

template <class T, MatrixAlloc::Mode alloc_mode> 
T * Matrix<T,alloc_mode>::operator () (size_t main_dimension_pos)
{
  if (alloc_mode == MatrixAlloc::Rows)
  {
    Assert (main_dimension_pos < m_rows, "Out of bounds row access for row %lu in Matrix(%lu,%lu) !", main_dimension_pos, m_rows,m_cols);
    return m + m_cols*main_dimension_pos;
  }
  else
  {
    Assert (main_dimension_pos < m_rows, "Out of bounds row access for column %lu in Matrix(%lu,%lu) !", main_dimension_pos, m_rows,m_cols);
    return m + m_rows*main_dimension_pos;
  }
}

template <class T, MatrixAlloc::Mode alloc_mode>
std::ostream &operator << (std::ostream &output, Matrix<T,alloc_mode> &matrix)
{
  for (size_t row=0; row < matrix.m_rows ; ++row)
  {
    for (size_t col=0; col < matrix.m_cols; ++col)
      output << matrix(row,col) << "\t";  
      
    output << "\n";
  }
  output << std::endl;

  return output;
}


template <class T, MatrixAlloc::Mode alloc_mode>
std::istream &operator >> (std::istream &input, Matrix<T,alloc_mode> &matrix)
{
  for (size_t row = 0; row < matrix.m_rows ; ++row)
  {
    for (size_t col = 0; col < matrix.m_cols; ++col)
      input >> matrix(row,col);
  }

  return input;
}


 
template <class T, MatrixAlloc::Mode alloc_mode>
Matrix<T,alloc_mode> & Matrix<T,alloc_mode>::operator = (const Matrix<T,alloc_mode> &o)
{
  if (this == &o)
    return *this;

  Assert ( m_rows == o.m_rows && m_cols == o.m_cols , "Invalid assignment operation. Matrices sizes do not match!");

  memcpy((void*)m, (void*)o.m, sizeof(T)*m_size);

  return *this;
}

template <class T, MatrixAlloc::Mode alloc_mode>
const Matrix<T,alloc_mode> & Matrix<T,alloc_mode>::operator += (const Matrix<T,alloc_mode> &a)
{
  Assert (m_rows == a.m_rows && m_cols && a.m_cols, "Matrix sizes do not match!");

  for (size_t i=0; i < rows() ; i++)
    for (size_t j=0; j < cols(); j++)
      (*this)(i,j) += a(i,j);

  return *this;
}

/*
  template <class T>
  const Matrix<T> Matrix<T>::operator + (const Matrix<T> &a)
  {
  size_t i,j;
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

  size_t i,j,k;
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

/*
template <class T> Matrix<float> abs_matrix(Matrix<T> &m)
{
  Matrix<float> modulo(m.m_rows,m.m_cols,0);

  #pragma omp parallel for
  for (size_t i=0; i < m.m_rows * m.m_cols; ++i)
    modulo.h_m[i] = (m.h_m[i]).abs();

  return modulo;
}
*/

template <class T, MatrixAlloc::Mode alloc_mode>
T * Matrix<T,alloc_mode>::row(size_t r)
{
  Assert (alloc_mode == MatrixAlloc::Rows, "This matrix is allocated in Cols, use col() instead!");
  Assert (r < m_rows, "Out of bounds access! Row %lu (max = %lu).", r, m_rows-1);
  
  return m + r*m_cols;
}

template <class T, MatrixAlloc::Mode alloc_mode>
T * Matrix<T,alloc_mode>::col(size_t c)
{
  Assert (alloc_mode == MatrixAlloc::Cols, "This matrix is allocated in Rows, use row() instead!");
  Assert (c < m_cols, "Out of bounds access! Col %lu (max = %lu).", c, m_cols-1);
  
  return m + c*m_rows;
}

template <class T, MatrixAlloc::Mode alloc_mode>
const Matrix<T,alloc_mode> & Matrix<T,alloc_mode>::operator *= (const T factor)
{
  for (size_t i = 0; i < m_size ; ++i)
    m[i] *= factor;

  return *this;
}

template <class T, MatrixAlloc::Mode alloc_mode>
const Matrix<T,alloc_mode> & Matrix<T,alloc_mode>::operator /= (const T factor)
{
  (*this) *= 1/factor; // Multiplication is faster

  return *this;
}

template <class T, MatrixAlloc::Mode alloc_mode>
void Matrix<T,alloc_mode>::max_index(size_t &row, size_t &col)
{
  T m(_default_value), value;

  row = col = 0;
  for (size_t r=0; r < m_rows; ++r)
    for (size_t c=0; c < m_cols; ++c)
      {
	value = (*this)(r,c);
	if (value > m)
	  {
	    m = value;
	    row = r;
	    col = c;
	  }
      }
}

template <class T, MatrixAlloc::Mode alloc_mode>
void Matrix<T,alloc_mode>::min_index(size_t &row, size_t &col)
{
  T m(_default_value), value;

  row = col = 0;
  for (size_t r=0; r < m_rows; ++r)
    for (size_t c=0; c < m_cols; ++c)
      {
	value = (*this)(r,c);
	if (value > m)
	  {
	    m = value;
	    row = r;
	    col = c;
	  }
      }
}

template <class T, MatrixAlloc::Mode alloc_mode>
void Matrix<T,alloc_mode>::print(size_t rows, size_t cols)
{
  for (size_t i=0; i < rows; ++i)
    {
      for (size_t j=0; j < cols; ++j)
	std::cout << (*this)(i,j) << " ";
      std::cout << std::endl;
    }
}


#endif // MATRIX_H
