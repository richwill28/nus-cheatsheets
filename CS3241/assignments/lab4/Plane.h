#ifndef _PLANE_H_
#define _PLANE_H_

#include "Surface.h"

class Plane : public Surface {
public:
  // The plane equation is Ax + By + Cz + D = 0.
  double A, B, C, D;

  Plane(double A_, double B_, double C_, double D_,
        const Material &theMaterial) {
    A = A_;
    B = B_;
    C = C_;
    D = D_;
    material = theMaterial;
  }

  Plane(const Vector3d &normal, const Vector3d &pointOnPlane,
        const Material &theMaterial) {
    A = normal.x();
    B = normal.y();
    C = normal.z();
    D = -dot(pointOnPlane, normal);
    material = theMaterial;
  }

  bool hit(const Ray &r, // Ray being sent.
           double tmin,  // Minimum hit parameter to be searched for.
           double tmax,  // Maximum hit parameter to be searched for.
           SurfaceHitRecord &rec) const override;

  [[nodiscard]] bool
  shadowHit(const Ray &r, // Ray being sent.
            double tmin,  // Minimum hit parameter to be searched for.
            double tmax   // Maximum hit parameter to be searched for.
  ) const override;
};

#endif // _PLANE_H_
