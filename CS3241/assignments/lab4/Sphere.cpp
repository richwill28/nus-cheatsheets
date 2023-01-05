//============================================================
// STUDENT NAME: Richard Willie
// NUS User ID.: E0550368
// COMMENTS TO GRADER:
//   - Developed on Manjaro
//   - Tested with GCC 12.2.0, GLUT 3.2.2, GLEW 2.2.0,
//     and OpenMP 4.5
//   - How to compile:
//       - cd <path to CMakeLists.txt>
//       - cmake -S ./ -B ./build
//       - cmake --build ./build
// ============================================================

#include "Sphere.h"
#include <cmath>

using namespace std;

bool Sphere::hit(const Ray &r, double tmin, double tmax,
                 SurfaceHitRecord &rec) const {
  //***********************************************
  //*********** WRITE YOUR CODE HERE **************
  //***********************************************

  // Find the ray-sphere intersection
  Vector3d R_o = r.origin() - center;
  Vector3d R_d = r.direction();
  double a = dot(R_d, R_d);
  double b = 2.0 * dot(R_d, R_o);
  double c = dot(R_o, R_o) - radius * radius;
  double d = b * b - 4 * a * c;

  // No real solutions
  if (d < 0.0) {
    return false;
  }

  double t1 = (-b - sqrt(d)) / (2.0 * a);
  double t2 = (-b + sqrt(d)) / (2.0 * a);

  // Both negatives, which means that the ray is pointing away from the sphere
  if (t1 < 0.0 && t2 < 0.0) {
    return false;
  }

  double t0;
  if (t1 < 0.0) {
    t0 = t2;
  } else if (t2 < 0.0) {
    t0 = t1;
  } else {
    t0 = min(t1, t2);
  }

  if (t0 < tmin || t0 > tmax) {
    return false;
  }

  rec.t = t0;
  rec.p = r.pointAtParam(t0);
  rec.normal = (rec.p - center).unitVector();
  rec.material = material;
  return true;
}

bool Sphere::shadowHit(const Ray &r, double tmin, double tmax) const {
  //***********************************************
  //*********** WRITE YOUR CODE HERE **************
  //***********************************************

  // Find the ray-sphere intersection
  Vector3d R_o = r.origin() - center;
  Vector3d R_d = r.direction();
  double a = dot(R_d, R_d);
  double b = 2.0 * dot(R_d, R_o);
  double c = dot(R_o, R_o) - radius * radius;
  double d = b * b - 4 * a * c;

  // No real solutions
  if (d < 0.0) {
    return false;
  }

  double t1 = (-b - sqrt(d)) / (2.0 * a);
  double t2 = (-b + sqrt(d)) / (2.0 * a);

  // Both negatives, which means that the ray is pointing away from the sphere
  if (t1 < 0.0 && t2 < 0.0) {
    return false;
  }

  double t0;
  if (t1 < 0.0) {
    t0 = t2;
  } else if (t2 < 0.0) {
    t0 = t1;
  } else {
    t0 = min(t1, t2);
  }

  return (t0 >= tmin && t0 < tmax);
}
