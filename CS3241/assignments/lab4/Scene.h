#ifndef _SCENE_H_
#define _SCENE_H_

#include "Camera.h"
#include "Image.h"
#include "Light.h"
#include "Material.h"
#include "Surface.h"
#include <vector>

struct Scene {
  std::vector<Surface *> surfaces; // Array of surface primitives.

  std::vector<Material> materials; // Vector of materials

  std::vector<PointLightSource> ptLights; // Vector of point light sources

  AmbientLightSource amLight; // The global ambient light source.

  Color backgroundColor; // Use this color if ray hits nothing.

  Camera camera; // The camera.
};

#endif // _SCENE_H_
