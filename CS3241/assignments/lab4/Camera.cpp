#include "Camera.h"
#include "Vector3d.h"
#include <cassert>

Camera &Camera::setCamera(const Vector3d &eye, const Vector3d &lookAt,
                          const Vector3d &upVector, double left, double right,
                          double bottom, double top, double near,
                          int image_width, int image_height) {
  assert(image_width > 0 && image_height > 0);
  mImageWidth = image_width;
  mImageHeight = image_height;

  mCOP = eye;
  Vector3d cop_n = (eye - lookAt).unitVector();
  Vector3d cop_u = cross(upVector.unitVector(), cop_n);
  Vector3d cop_v = cross(cop_n, cop_u);

  mImageOrigin = mCOP + (left * cop_u) + (bottom * cop_v) + (-near * cop_n);

  mImageU = (right - left) * cop_u;
  mImageV = (top - bottom) * cop_v;
  return (*this);
}
