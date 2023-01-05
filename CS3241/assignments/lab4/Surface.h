#ifndef _SURFACE_H_
#define _SURFACE_H_

//  Abstract class Surface may be subclassed to a particular type of
//  Surface such as a Plane, Sphere, Triangle, and triangle mesh.

#include "Color.h"
#include "Material.h"
#include "Ray.h"
#include "Vector3d.h"

struct SurfaceHitRecord {
  double t{};        // Ray hits at p = Ray.origin() + t * Ray.direction().
  Vector3d p;        // The point of intersection.
  Vector3d normal;   // Surface normal at p. May not be unit vector.
  Material material; // Surface material.
};

class Surface {
public:
  Material material; // Material of the surface.

  // Does a Ray hit the Surface?
  virtual bool hit(const Ray &r, // Ray being sent.
                   double tmin,  // Minimum hit parameter to be searched for.
                   double tmax,  // Maximum hit parameter to be searched for.
                   SurfaceHitRecord &rec) const = 0;

  // Does a Ray hit any Surface?  Allows early termination.
  [[nodiscard]] virtual bool
  shadowHit(const Ray &r, // Ray being sent.
            double tmin,  // Minimum hit parameter to be searched for.
            double tmax   // Maximum hit parameter to be searched for.
  ) const = 0;

  virtual ~Surface() = default;

}; // Surface

#endif // _SURFACE_H_
