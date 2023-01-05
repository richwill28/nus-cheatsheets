#ifndef _VECTOR3D_H_
#define _VECTOR3D_H_

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

// For 3D vectors and 3D points.

class Vector3d {
public:
  // Constructors

  Vector3d() = default;
  explicit Vector3d(const double v[3]) {
    data[0] = v[0];
    data[1] = v[1];
    data[2] = v[2];
  }
  explicit Vector3d(const float v[3]) {
    data[0] = v[0];
    data[1] = v[1];
    data[2] = v[2];
  }
  Vector3d(double x, double y, double z) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
  }

  // Data setting and reading.

  Vector3d &setX(double a) {
    data[0] = a;
    return (*this);
  }
  Vector3d &setY(double a) {
    data[1] = a;
    return (*this);
  }
  Vector3d &setZ(double a) {
    data[2] = a;
    return (*this);
  }

  Vector3d &setXYZ(const double v[3]) {
    data[0] = v[0];
    data[1] = v[1];
    data[2] = v[2];
    return (*this);
  }
  Vector3d &setXYZ(const float v[3]) {
    data[0] = v[0];
    data[1] = v[1];
    data[2] = v[2];
    return (*this);
  }
  Vector3d &setXYZ(double x, double y, double z) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
    return (*this);
  }
  Vector3d &setToZeros() {
    data[0] = data[1] = data[2] = 0.0;
    return (*this);
  }

  double &x() { return data[0]; }
  double &y() { return data[1]; }
  double &z() { return data[2]; }

  [[nodiscard]] double x() const { return data[0]; }
  [[nodiscard]] double y() const { return data[1]; }
  [[nodiscard]] double z() const { return data[2]; }

  void getXYZ(double v[3]) const {
    v[0] = data[0];
    v[1] = data[1];
    v[2] = data[2];
  }
  void getXYZ(float v[3]) const {
    v[0] = (float)data[0];
    v[1] = (float)data[1];
    v[2] = (float)data[2];
  }

  // Operators.

  double &operator[](int i) {
    assert(i >= 0 && i < 3);
    return data[i];
  }

  double operator[](int i) const {
    assert(i >= 0 && i < 3);
    return data[i];
  }

  Vector3d operator+() const { return (*this); }

  Vector3d operator-() const { return {-data[0], -data[1], -data[2]}; }

  Vector3d &operator+=(const Vector3d &v) {
    data[0] += v.data[0];
    data[1] += v.data[1];
    data[2] += v.data[2];
    return (*this);
  }

  Vector3d &operator-=(const Vector3d &v) {
    data[0] -= v.data[0];
    data[1] -= v.data[1];
    data[2] -= v.data[2];
    return (*this);
  }

  Vector3d &operator*=(const Vector3d &v) {
    data[0] *= v.data[0];
    data[1] *= v.data[1];
    data[2] *= v.data[2];
    return (*this);
  }

  Vector3d &operator/=(const Vector3d &v) {
    data[0] /= v.data[0];
    data[1] /= v.data[1];
    data[2] /= v.data[2];
    return (*this);
  }

  Vector3d &operator*=(double a) {
    data[0] *= a;
    data[1] *= a;
    data[2] *= a;
    return (*this);
  }

  Vector3d &operator/=(double a) {
    data[0] /= a;
    data[1] /= a;
    data[2] /= a;
    return (*this);
  }

  friend Vector3d operator+(const Vector3d &v1, const Vector3d &v2);
  friend Vector3d operator-(const Vector3d &v1, const Vector3d &v2);
  friend Vector3d operator*(const Vector3d &v1, const Vector3d &v2);
  friend Vector3d operator/(const Vector3d &v1, const Vector3d &v2);
  friend Vector3d operator*(double a, const Vector3d &v);
  friend Vector3d operator*(const Vector3d &v, double a);
  friend Vector3d operator/(const Vector3d &v, double a);
  friend bool operator==(const Vector3d &v1, const Vector3d &v2);
  friend bool operator!=(const Vector3d &v1, const Vector3d &v2);
  friend double dot(const Vector3d &v1, const Vector3d &v2);
  friend Vector3d cross(const Vector3d &v1, const Vector3d &v2);
  friend Vector3d triNormal(const Vector3d &v1, const Vector3d &v2,
                            const Vector3d &v3);

  // Other functions.

  [[nodiscard]] double length() const {
    return sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
  }

  [[nodiscard]] double sqrLength() const {
    return (data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
  }

  [[nodiscard]] Vector3d unitVector() const {
    double invLen = 1.0 / length();
    return {data[0] * invLen, data[1] * invLen, data[2] * invLen};
  }

  Vector3d &makeUnitVector() {
    double invLen = 1.0 / length();
    data[0] *= invLen;
    data[1] *= invLen;
    data[2] *= invLen;
    return (*this);
  }

private:
  // The 3D vector data.
  std::array<double, 3> data{};

}; // Vector3d

// More unary and binary vector operators.

inline Vector3d operator+(const Vector3d &v1, const Vector3d &v2) {
  return {v1.data[0] + v2.data[0], v1.data[1] + v2.data[1],
          v1.data[2] + v2.data[2]};
}

inline Vector3d operator-(const Vector3d &v1, const Vector3d &v2) {
  return {v1.data[0] - v2.data[0], v1.data[1] - v2.data[1],
          v1.data[2] - v2.data[2]};
}

inline Vector3d operator*(const Vector3d &v1, const Vector3d &v2) {
  return {v1.data[0] * v2.data[0], v1.data[1] * v2.data[1],
          v1.data[2] * v2.data[2]};
}

inline Vector3d operator/(const Vector3d &v1, const Vector3d &v2) {
  return {v1.data[0] / v2.data[0], v1.data[1] / v2.data[1],
          v1.data[2] / v2.data[2]};
}

inline Vector3d operator*(double a, const Vector3d &v) {
  return {a * v.data[0], a * v.data[1], a * v.data[2]};
}

inline Vector3d operator*(const Vector3d &v, double a) {
  return {a * v.data[0], a * v.data[1], a * v.data[2]};
}

inline Vector3d operator/(const Vector3d &v, double a) {
  return {v.data[0] / a, v.data[1] / a, v.data[2] / a};
}

inline bool operator==(const Vector3d &v1, const Vector3d &v2) {
  return ((v1.data[0] == v2.data[0]) && (v1.data[1] == v2.data[1]) &&
          (v1.data[2] == v2.data[2]));
}

inline bool operator!=(const Vector3d &v1, const Vector3d &v2) {
  return ((v1.data[0] != v2.data[0]) || (v1.data[1] != v2.data[1]) ||
          (v1.data[2] != v2.data[2]));
}

inline double dot(const Vector3d &v1, const Vector3d &v2) {
  return (v1.data[0] * v2.data[0]) + (v1.data[1] * v2.data[1]) +
         (v1.data[2] * v2.data[2]);
}

inline Vector3d cross(const Vector3d &v1, const Vector3d &v2) {
  return {v1.data[1] * v2.data[2] - v1.data[2] * v2.data[1],
          v1.data[2] * v2.data[0] - v1.data[0] * v2.data[2],
          v1.data[0] * v2.data[1] - v1.data[1] * v2.data[0]};
}

// Returns the normal vector of the triangle.
inline Vector3d triNormal(const Vector3d &v1, const Vector3d &v2,
                          const Vector3d &v3) {
  return cross(v2 - v1, v3 - v1);
}

inline std::istream &operator>>(std::istream &is, Vector3d &v) {
  return (is >> v.x() >> v.y() >> v.z());
}

inline std::ostream &operator<<(std::ostream &os, Vector3d v) {
  return (os << v.x() << " " << v.y() << " " << v.z());
}

#endif // _VECTOR3D_H_
