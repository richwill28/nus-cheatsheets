#ifndef _IMAGE_H_
#define _IMAGE_H_

#include "Color.h"
#include <cassert>
#include <cmath>
#include <cstdlib>


class Image {
public:
  Image() : mWidth(0), mHeight(0), mData(nullptr){};

  Image(int width, int height) : mWidth(width), mHeight(height) {
    assert(width > 0 && height > 0);
    mData = new Color[width * height];
  }

  ~Image() { delete[] mData; }

  Image &setImage(int width, int height);

  Image &setImage(int width, int height, Color initColor);

  Image &setPixel(int x, int y, Color c) {
    assert(x >= 0 && x < mWidth && y >= 0 && y < mHeight);
    mData[y * mWidth + x] = c;
    return (*this);
  }

  [[nodiscard]] Color getPixel(int x, int y) const {
    assert(x >= 0 && x < mWidth && y >= 0 && y < mHeight);
    return mData[y * mWidth + x];
  }

  [[nodiscard]] int width() const { return mWidth; }

  [[nodiscard]] int height() const { return mHeight; }

  Image &gammaCorrect(float gamma = 2.2f);

  // Write image to a file. Returns true iff successful.
  [[nodiscard]] bool writeToFile(const std::string &filename) const;

private:
  int mWidth{}, mHeight{};
  Color *mData{};

  // Disallow the use of copy constructor and assignment operator.
  Image(const Image &image) = delete;
  Image &operator=(const Image &image) = delete;

}; // Image

#endif // _IMAGE_H_
