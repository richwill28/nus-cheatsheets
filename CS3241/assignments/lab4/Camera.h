#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "Ray.h"
#include "Vector3d.h"
#include <cassert>

// A Camera object is used for obtaining a ray through each pixel of the image.

class Camera {
public:
  //////////////////////////////////////////////////////////////////////////////////////
  // The default camera sits at the origin looking in the -z direction and with
  // the y-axis as the up-vector. The image plane is 1 unit away and the camera
  // has a symmetric square FOV of 90 degrees. The default image resolution is
  // 256x256.
  //////////////////////////////////////////////////////////////////////////////////////

  Camera() {
    setCamera(Vector3d(0, 0, 0), Vector3d(0, 0, -1), Vector3d(0, 1, 0), -1, 1,
              -1, 1, 1, 256, 256);
  }

  //////////////////////////////////////////////////////////////////////////////////////
  // A camera can be set up just like using gluLookAt() and glFrustum()
  // together in OpenGL. image_width and image_height are just like the
  // viewport size in pixels.
  //////////////////////////////////////////////////////////////////////////////////////

  Camera(const Vector3d &eye, const Vector3d &lookAt, const Vector3d &upVector,
         double left, double right, double bottom, double top, double near,
         int image_width, int image_height) {
    setCamera(eye, lookAt, upVector, left, right, bottom, top, near,
              image_width, image_height);
  }

  Camera &setCamera(const Vector3d &eye, const Vector3d &lookAt,
                    const Vector3d &upVector, double left, double right,
                    double bottom, double top, double near, int image_width,
                    int image_height);

  Camera &setImageSize(int image_width, int image_height) {
    assert(image_width > 0 && image_height > 0);
    mImageWidth = image_width;
    mImageHeight = image_height;
    return (*this);
  }

  [[nodiscard]] int getImageWidth() const { return mImageWidth; }

  [[nodiscard]] int getImageHeight() const { return mImageHeight; }

  //////////////////////////////////////////////////////////////////////////////////////
  // Returns a ray that goes from the camera's origin through the
  // pixel location (pixelPosX, pixelPosY) of the camera.
  // Note that pixelPosX and pixelPosY can be non-integer.
  // The image origin is at the bottom-leftmost corner, that means:
  // * The bottom-leftmost corner of the image is (0, 0).
  // * The top-rightmost corner of the image is (imageWidth, imageHeight).
  // * The center of the bottom-leftmost pixel is (0.5, 0.5).
  // * The center of the top-rightmost pixel is (imageWidth-0.5,
  // imageHeight-0.5).
  //
  // Note that the ray returned may not have unit direction vector.
  //////////////////////////////////////////////////////////////////////////////////////

  [[nodiscard]] Ray getRay(double pixelPosX, double pixelPosY) const {
    Vector3d imgPos = mImageOrigin + (pixelPosX / mImageWidth) * mImageU +
                      (pixelPosY / mImageHeight) * mImageV;
    return {mCOP, imgPos - mCOP};
  }

private:
  Vector3d mCOP; // The center of projection or the camera viewpoint.
  Vector3d mImageOrigin;
  Vector3d mImageU, mImageV;
  int mImageWidth{}, mImageHeight{}; // In number of pixels.

}; // Camera

#endif // _CAMERA_H_
