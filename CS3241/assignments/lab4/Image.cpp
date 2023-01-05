#include "Image.h"
#include "ImageIO.h"
#include <cassert>
#include <cmath>

Image &Image::setImage(int width, int height) {
  assert(width > 0 && height > 0);
  mWidth = width;
  mHeight = height;
  delete[] mData;
  mData = new Color[width * height];
  return (*this);
}

Image &Image::setImage(int width, int height, Color initColor) {
  setImage(width, height);
  for (int i = 0; i < width * height; i++)
    mData[i] = initColor;
  return (*this);
}

Image &Image::gammaCorrect(float gamma) {
  for (int i = 0; i < mWidth * mHeight; i++) {
    mData[i].clamp(0.0f, 1.0f);
    mData[i].gammaCorrect(gamma);
  }
  return (*this);
}

bool Image::writeToFile(const std::string &filename) const {
  assert(mWidth > 0 && mHeight > 0);
  auto *bytes = new uchar[3 * mWidth * mHeight];

  for (int i = 0; i < mWidth * mHeight; i++) {
    int r = (int)(256.0 * mData[i].r());
    if (r > 255)
      r = 255;
    int g = (int)(256.0 * mData[i].g());
    if (g > 255)
      g = 255;
    int b = (int)(256.0 * mData[i].b());
    if (b > 255)
      b = 255;

    bytes[3 * i + 0] = (uchar)r;
    bytes[3 * i + 1] = (uchar)g;
    bytes[3 * i + 2] = (uchar)b;
  }

  int status = ImageIO::SaveImageToFilePNG(filename, bytes, mWidth, mHeight, 3);

  delete[] bytes;
  return (status == 1);
}
