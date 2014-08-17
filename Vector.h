#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <memory>
#include <cmath>

/////// NEEDED FOR FRIEND FUNCTIONS TO BE TEMPLATED ///////
template <class T> class Vector3D;
template <class T> std::ostream & operator<< (std::ostream &, const Vector3D<T>&);
template <class T> Vector3D<T> CrossProduct  (const Vector3D<T> &a, const Vector3D<T> &b);
template <class T> T dot (Vector3D<T> &, Vector3D<T> &);
/////////////////////////////////////////////////////////////////

template <class T>
class Vector3D
{
  friend Vector3D CrossProduct <>(const Vector3D &a, const Vector3D &b);
  friend T dot <>(Vector3D &, Vector3D &);
  friend std::ostream & operator << <>(std::ostream &, const Vector3D <T>&);

public:
  Vector3D (T x=0, T y=0, T z=0)
  {
    x_ = x;
    y_ = y;
    z_ = z;
  }
  Vector3D (const Vector3D<T> &v)
  {
    x_ = v.x_;
    y_ = v.y_;
    z_ = v.z_;
  }

  const T x() { return x_; }
  const T y() { return y_; }
  const T z() { return z_; }

  Vector3D & operator+= (const Vector3D &vector);
  Vector3D & operator-= (const Vector3D &vector);
  Vector3D & operator*= (const T scale       );
  const Vector3D operator + (const Vector3D &vector) const;
  const Vector3D operator - (const Vector3D &vector) const;
  const Vector3D operator * (const T scale       ) const;
  Vector3D & operator = (const Vector3D &vector);

  void operator () (T x, T y, T z)
  {
    x_ = x;
    y_ = y;
    z_ = z;
  }

  T length() const;
  Vector3D<T> & normalize();

protected:
  T x_, y_, z_;
};

template <class T>
std::ostream & operator << (std::ostream &out, const Vector3D <T>&vec)
{
  out << "(" << vec.x_ << "," << vec.y_ << "," << vec.z_ << ")";

  return out;
}


template <class T>
Vector3D<T> & Vector3D<T>::operator+= (const Vector3D <T> &v)
{
  x_ += v.x_;
  y_ += v.y_;
  z_ += v.z_;

  return *this;
}

template <class T>
Vector3D<T> & Vector3D<T>::operator-= (const Vector3D <T> &v)
{
  x_ -= v.x_;
  y_ -= v.y_;
  z_ -= v.z_;

  return *this;
}

template <class T>
Vector3D<T> & Vector3D<T>::operator*= (const T scale)
{
  x_ *= scale;
  y_ *= scale;
  z_ *= scale;

  return *this;
}

template <class T>
Vector3D<T> & Vector3D<T>::operator= (const Vector3D <T> &v)
{
  if (&v == this) return *this;

  x_ = v.x_;
  y_ = v.y_;
  z_ = v.z_;

  return *this;
}

template <class T>
const Vector3D<T>  Vector3D<T>::operator+ (const Vector3D <T> &v) const
{
  Vector3D<T> vec(*this);
  vec += v;

  return vec;
}

template <class T>
const Vector3D<T>  Vector3D<T>::operator- (const Vector3D <T> &v) const
{
  Vector3D<T> vec(*this);
  vec -= v;

  return vec;
}

template <class T>
const Vector3D<T> Vector3D<T>::operator* (const T scale) const
{
  Vector3D<T> vec(*this);

  vec *= scale;

  return vec;
}

template <class T>
T Vector3D<T>::length() const
{
  return std::sqrt( (x_*x_)+(y_*y_)+(z_*z_) );
}

template <class T>
Vector3D<T> & Vector3D<T>::normalize()
{
  T len = this->length();

  AssertRuntime(std::abs(len) > 0.00001, 
		"Vector(%f,%f,%f) is too small!",x_,y_,z_);

  x_ /= len;
  y_ /= len;
  z_ /= len;

  return *this;
}


template <class T>
Vector3D<T> CrossProduct (const Vector3D<T> &a, const Vector3D<T> &b)
{
  return Vector3D<T> (a.y_ * b.z_ - b.y_ * a.z_,
		      a.x_ * b.z_ - b.x_ * a.z_,
		      a.x_ * b.y_ - b.x_ * a.y_);
}

template <class T>
T dot (Vector3D<T> &v1, Vector3D<T> &v2)
{
  return v1.x_*v2.x_  +  v1.y_*v2.y_  +  v1.z_*v2.z_;
}

template <class T>
Vector3D<T> project (Vector3D<T> ya, Vector3D<T> xa)
{
  T l0 = xa.length();
  T l = l0 * l0;
  
  return Vector3D<T> (dot(ya,xa)*xa.x_ / l,
                    dot(ya,xa)*xa.y_ / l,
                    dot(ya,xa)*xa.z_ / l);
}

template <class T>
T nproject (Vector3D<T> ya, Vector3D<T> xa)
{
  return dot(ya,xa) / xa.length();
}

#endif
