#ifndef _SPHERE_H_
#define _SPHERE_H_

#include "Surface.h"

class Sphere : public Surface {
public:
  Vector3d center;
  double radius;

  Sphere(const Vector3d &theCenter, double theRadius,
         const Material &theMaterial) {
    center = theCenter;
    radius = theRadius;
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

#endif // _SPHERE_H_
