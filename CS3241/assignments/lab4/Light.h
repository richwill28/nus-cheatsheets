#ifndef _LIGHT_H_
#define _LIGHT_H_

#include "Color.h"
#include "Vector3d.h"

//////////////////////////////////////////////////////////////////////////////
//
// The lighting model used here is a slightly modified version of that
// seen in the lecture on ray tracing.
//
// Here it is computed as
//
//     I_local = I_a * k_a  +
//               SUM_OVER_ALL_LIGHTS ( I_source * [ k_d * (N.L) + k_r * (R.V)^n
//               ] )
//
// and
//
//     I = I_local  +  k_rg * I_reflected
//
// Note that light sources only illuminate the scene, and they do not
// appear in the rendered image.
//
//////////////////////////////////////////////////////////////////////////////

struct PointLightSource {
  Vector3d position;
  Color I_source;
};

// There should just be one single AmbientLightSource object in each scene.

struct AmbientLightSource {
  Color I_a;
};

#endif // _LIGHT_H_
