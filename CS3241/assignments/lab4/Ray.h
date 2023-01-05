#ifndef _RAY_H_
#define _RAY_H_

#include "Vector3d.h"
#include <iostream>

class Ray {
public:
  // Constructors

  Ray() = default;

  Ray(const Vector3d &origin, const Vector3d &direction) {
    data[0] = origin;
    data[1] = direction;
  }

  // Data setting and reading.

  Ray &setRay(const Vector3d &origin, const Vector3d &direction) {
    data[0] = origin;
    data[1] = direction;
    return (*this);
  }

  Ray &setOrigin(const Vector3d &origin) {
    data[0] = origin;
    return (*this);
  }
  Ray &setDirection(const Vector3d &direction) {
    data[1] = direction;
    return (*this);
  }

  [[nodiscard]] Vector3d origin() const { return data[0]; }
  [[nodiscard]] Vector3d direction() const { return data[1]; }

  // Other functions.

  [[nodiscard]] Vector3d pointAtParam(double t) const {
    return data[0] + t * data[1];
  }

  Ray &makeUnitDirection() {
    data[1].makeUnitVector();
    return (*this);
  }

  Ray &moveOriginForward(double delta_t) {
    data[0] += delta_t * data[1];
    return (*this);
  }

private:
  std::array<Vector3d, 2> data;

}; // Ray

inline std::ostream &operator<<(std::ostream &os, const Ray &r) {
  return (os << "(" << r.origin() << ") + t(" << r.direction() << ")");
}

#endif // _RAY_H_
