#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "Surface.h"

class Triangle : public Surface {
public:
  Vector3d v0, v1, v2; // Vertices.
  Vector3d n0, n1, n2; // Vertex normals.

  Triangle(const Vector3d &v0_, const Vector3d &v1_, const Vector3d &v2_,
           const Material &theMaterial) {
    v0 = v0_;
    v1 = v1_;
    v2 = v2_;
    n0 = n1 = n2 = triNormal(v0, v1, v2);
    material = theMaterial;
  }

  Triangle(const Vector3d &v0_, const Vector3d &v1_, const Vector3d &v2_,
           const Vector3d &n0_, const Vector3d &n1_, const Vector3d &n2_,
           const Material &theMaterial) {
    v0 = v0_;
    v1 = v1_;
    v2 = v2_;
    n0 = n0_;
    n1 = n1_;
    n2 = n2_;
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

#endif // _TRIANGLE_H_
