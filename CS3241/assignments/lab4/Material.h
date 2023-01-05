#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include "Color.h"

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

struct Material {
  Color k_a;
  Color k_d;
  Color k_r;

  float n{}; // The specular reflection exponent. It ranges from 0.0 to 128.0.

  Color k_rg;
};

#endif // _MATERIAL_H_
