#ifndef _COLOR_H_
#define _COLOR_H_

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>


// For RGB color. Each component value ranges from 0.0 to 1.0.

class Color {
public:
  // Constructors

  Color() { data = {0, 0, 0}; }
  explicit Color(const float c[3]) {
    data[0] = c[0];
    data[1] = c[1];
    data[2] = c[2];
  }
  explicit Color(const double c[3]) {
    data[0] = (float)c[0];
    data[1] = (float)c[1];
    data[2] = (float)c[2];
  }
  Color(float r, float g, float b) {
    data[0] = r;
    data[1] = g;
    data[2] = b;
  }

  // Data setting and reading.

  Color &setR(float a) {
    data[0] = a;
    return (*this);
  }
  Color &setG(float a) {
    data[1] = a;
    return (*this);
  }
  Color &setB(float a) {
    data[2] = a;
    return (*this);
  }

  Color &setRGB(const float c[3]) {
    data[0] = c[0];
    data[1] = c[1];
    data[2] = c[2];
    return (*this);
  }
  Color &setRGB(const double c[3]) {
    data[0] = (float)c[0];
    data[1] = (float)c[1];
    data[2] = (float)c[2];
    return (*this);
  }
  Color &setRGB(float r, float g, float b) {
    data[0] = r;
    data[1] = g;
    data[2] = b;
    return (*this);
  }

  float &r() { return data[0]; }
  float &g() { return data[1]; }
  float &b() { return data[2]; }

  [[nodiscard]] float r() const { return data[0]; }
  [[nodiscard]] float g() const { return data[1]; }
  [[nodiscard]] float b() const { return data[2]; }

  void getRGB(double c[3]) const {
    c[0] = data[0];
    c[1] = data[1];
    c[2] = data[2];
  }
  void getRGB(float c[3]) const {
    c[0] = data[0];
    c[1] = data[1];
    c[2] = data[2];
  }

  // Operators.

  float &operator[](int i) {
    assert(i >= 0 && i < 3);
    return data[i];
  }

  float operator[](int i) const {
    assert(i >= 0 && i < 3);
    return data[i];
  }

  Color operator+() const { return (*this); }

  Color operator-() const { return {-data[0], -data[1], -data[2]}; }

  Color operator+=(const Color &c) {
    data[0] += c.data[0];
    data[1] += c.data[1];
    data[2] += c.data[2];
    return (*this);
  }

  Color operator-=(const Color &c) {
    data[0] -= c.data[0];
    data[1] -= c.data[1];
    data[2] -= c.data[2];
    return (*this);
  }

  Color operator*=(const Color &c) {
    data[0] *= c.data[0];
    data[1] *= c.data[1];
    data[2] *= c.data[2];
    return (*this);
  }

  Color operator/=(const Color &c) {
    data[0] /= c.data[0];
    data[1] /= c.data[1];
    data[2] /= c.data[2];
    return (*this);
  }

  Color operator*=(float a) {
    data[0] *= a;
    data[1] *= a;
    data[2] *= a;
    return (*this);
  }

  Color operator/=(float a) {
    data[0] /= a;
    data[1] /= a;
    data[2] /= a;
    return (*this);
  }

  friend Color operator+(const Color &c1, const Color &c2);
  friend Color operator-(const Color &c1, const Color &c2);
  friend Color operator*(const Color &c1, const Color &c2);
  friend Color operator/(const Color &c1, const Color &c2);
  friend Color operator*(float a, const Color &c);
  friend Color operator*(const Color &c, float a);
  friend Color operator/(const Color &c, float a);
  friend bool operator==(const Color &c1, const Color &c2);
  friend bool operator!=(const Color &c1, const Color &c2);

  // Other functions.

  Color &clamp(float low = 0.0f, float high = 1.0f) {
    if (data[0] > high)
      data[0] = high;
    else if (data[0] < low)
      data[0] = low;
    if (data[1] > high)
      data[1] = high;
    else if (data[1] < low)
      data[1] = low;
    if (data[2] > high)
      data[2] = high;
    else if (data[2] < low)
      data[2] = low;
    return (*this);
  }

  Color &gammaCorrect(float gamma = 2.2f) {
    float power = 1.0f / gamma;
    data[0] = pow(data[0], power);
    data[1] = pow(data[1], power);
    data[2] = pow(data[2], power);
    return (*this);
  }

private:
  // The RGB data.
  std::array<float, 3> data{};

}; // Color

// More unary and binary operators.

inline Color operator+(const Color &c1, const Color &c2) {
  return {c1.data[0] + c2.data[0], c1.data[1] + c2.data[1],
          c1.data[2] + c2.data[2]};
}

inline Color operator-(const Color &c1, const Color &c2) {
  return {c1.data[0] - c2.data[0], c1.data[1] - c2.data[1],
          c1.data[2] - c2.data[2]};
}

inline Color operator*(const Color &c1, const Color &c2) {
  return {c1.data[0] * c2.data[0], c1.data[1] * c2.data[1],
          c1.data[2] * c2.data[2]};
}

inline Color operator/(const Color &c1, const Color &c2) {
  return {c1.data[0] / c2.data[0], c1.data[1] / c2.data[1],
          c1.data[2] / c2.data[2]};
}

inline Color operator*(float a, const Color &c) {
  return {a * c.data[0], a * c.data[1], a * c.data[2]};
}

inline Color operator*(const Color &c, float a) {
  return {a * c.data[0], a * c.data[1], a * c.data[2]};
}

inline Color operator/(const Color &c, float a) {
  return {c.data[0] / a, c.data[1] / a, c.data[2] / a};
}

inline bool operator==(const Color &c1, const Color &c2) {
  return ((c1.data[0] == c2.data[0]) && (c1.data[1] == c2.data[1]) &&
          (c1.data[2] == c2.data[2]));
}

inline bool operator!=(const Color &c1, const Color &c2) {
  return ((c1.data[0] != c2.data[0]) || (c1.data[1] != c2.data[1]) ||
          (c1.data[2] != c2.data[2]));
}

inline std::istream &operator>>(std::istream &is, Color &c) {
  return (is >> c.r() >> c.g() >> c.b());
}

inline std::ostream &operator<<(std::ostream &os, Color c) {
  return (os << c.r() << " " << c.g() << " " << c.b());
}

#endif // _COLOR_H_
