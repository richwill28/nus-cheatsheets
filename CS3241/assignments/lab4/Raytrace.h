#ifndef _RAYTRACE_H_
#define _RAYTRACE_H_

#include "Color.h"
#include "Ray.h"
#include "Scene.h"

class Raytrace {
public:
  //////////////////////////////////////////////////////////////////////////////
  // Traces a ray into the scene.
  // reflectLevel: specifies number of levels of reflections (0 for no
  // reflection). hasShadow: specifies whether to generate shadows.
  //////////////////////////////////////////////////////////////////////////////

  static Color TraceRay(const Ray &ray, const Scene &scene, int reflectLevels,
                        bool hasShadow);
};

#endif // _RAYTRACE_H_
