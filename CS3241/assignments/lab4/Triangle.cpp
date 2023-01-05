#include "Triangle.h"
#include <cmath>

using namespace std;

bool Triangle::hit(const Ray &r, double tmin, double tmax,
                   SurfaceHitRecord &rec) const {
  Vector3d e1 = v1 - v0;
  Vector3d e2 = v2 - v0;
  Vector3d p = cross(r.direction(), e2);
  double a = dot(e1, p);
  // if ( a == 0.0 ) return false;
  double f = 1.0 / a;
  Vector3d s = r.origin() - v0;
  double beta = f * dot(s, p);
  if (beta < 0.0 || beta > 1.0)
    return false;

  Vector3d q = cross(s, e1);
  double gamma = f * dot(r.direction(), q);
  if (gamma < 0.0 || beta + gamma > 1.0)
    return false;

  double t = f * dot(e2, q);

  if (t >= tmin && t <= tmax) {
    // We have a hit -- populat hit record.
    rec.t = t;
    rec.p = r.pointAtParam(t);
    double alpha = 1.0 - beta - gamma;
    rec.normal = alpha * n0 + beta * n1 + gamma * n2;
    rec.material = material;
    return true;
  }
  return false;
}

bool Triangle::shadowHit(const Ray &r, double tmin, double tmax) const {
  Vector3d e1 = v1 - v0;
  Vector3d e2 = v2 - v0;
  Vector3d p = cross(r.direction(), e2);
  double a = dot(e1, p);
  // if ( a == 0.0 ) return false;
  double f = 1.0 / a;
  Vector3d s = r.origin() - v0;
  double beta = f * dot(s, p);
  if (beta < 0.0 || beta > 1.0)
    return false;

  Vector3d q = cross(s, e1);
  double gamma = f * dot(r.direction(), q);
  if (gamma < 0.0 || beta + gamma > 1.0)
    return false;

  double t = f * dot(e2, q);
  return (t >= tmin && t <= tmax);
}

/*
// Below is a more straightforward implementation, which is closer to that
described in lecture.


bool Triangle::hit( const Ray &r, double tmin, double tmax, SurfaceHitRecord
&rec ) const
{
    double A = v0.x() - v1.x();
    double B = v0.y() - v1.y();
    double C = v0.z() - v1.z();

    double D = v0.x() - v2.x();
    double E = v0.y() - v2.y();
    double F = v0.z() - v2.z();

    double G = r.direction().x();
    double H = r.direction().y();
    double I = r.direction().z();

    double J = v0.x() - r.origin().x();
    double K = v0.y() - r.origin().y();
    double L = v0.z() - r.origin().z();

    double EIHF = E*I - H*F;
    double GFDI = G*F - D*I;
    double DHEG = D*H - E*G;

    double denom = (A*EIHF + B*GFDI + C*DHEG);

    double beta = (J*EIHF + K*GFDI + L*DHEG) / denom;

    if ( beta < 0.0 || beta > 1.0 ) return false;

    double AKJB = A*K - J*B;
    double JCAL = J*C - A*L;
    double BLKC = B*L - K*C;

    double gamma = (I*AKJB + H*JCAL + G*BLKC) / denom;

    if ( gamma < 0.0 || beta + gamma > 1.0 ) return false;

    double t = -(F*AKJB + E*JCAL + D*BLKC) / denom;

    if ( t >= tmin && t <= tmax )
    {
        // We have a hit -- populat hit record.
        rec.t = t;
        rec.p = r.pointAtParam(t);
        double alpha = 1.0 - beta - gamma;
        rec.normal = alpha * n0 + beta * n1 + gamma * n2;
        rec.mat_ptr = matp;
        return true;
    }
    return false;
}




bool Triangle::shadowHit( const Ray &r, double tmin, double tmax ) const
{
    double A = v0.x() - v1.x();
    double B = v0.y() - v1.y();
    double C = v0.z() - v1.z();

    double D = v0.x() - v2.x();
    double E = v0.y() - v2.y();
    double F = v0.z() - v2.z();

    double G = r.direction().x();
    double H = r.direction().y();
    double I = r.direction().z();

    double J = v0.x() - r.origin().x();
    double K = v0.y() - r.origin().y();
    double L = v0.z() - r.origin().z();

    double EIHF = E*I - H*F;
    double GFDI = G*F - D*I;
    double DHEG = D*H - E*G;

    double denom = (A*EIHF + B*GFDI + C*DHEG);

    double beta = (J*EIHF + K*GFDI + L*DHEG) / denom;

    if ( beta < 0.0 || beta > 1.0 ) return false;

    double AKJB = A*K - J*B;
    double JCAL = J*C - A*L;
    double BLKC = B*L - K*C;

    double gamma = (I*AKJB + H*JCAL + G*BLKC) / denom;

    if ( gamma < 0.0 || beta + gamma > 1.0 ) return false;

    double t = -(F*AKJB + E*JCAL + D*BLKC) / denom;

    return ( t >= tmin && t <= tmax );
}
*/
