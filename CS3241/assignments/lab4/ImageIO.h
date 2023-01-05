#ifndef _IMAGEIO_H_
#define _IMAGEIO_H_

typedef unsigned char uchar;

class ImageIO {
public:
  /////////////////////////////////////////////////////////////////////////////
  // Deallocate the memory allocated to (*imageData) returned by
  // the function ReadImageFile().
  // (*imageData) will be set to NULL.
  /////////////////////////////////////////////////////////////////////////////

  static void DeallocateImageData(uchar **imageData);

  /////////////////////////////////////////////////////////////////////////////
  // Read an image from the input filename.
  // Returns 1 if successful or 0 if unsuccessful.
  // The returned image data will be pointed to by (*imageData).
  // The image width, image height, and number of components (color channels)
  // per pixel will be returned in (*imageWidth), (*imageHeight),
  // and (*numComponents).
  // The value of (*numComponents) can be 1, 2, 3 or 4.
  // The returned image data is always packed tightly with red, green, blue,
  // and alpha arranged from lower to higher memory addresses.
  // Each color channel take one byte.
  // The first pixel (origin of the image) is at the bottom-left of the image.
  /////////////////////////////////////////////////////////////////////////////

  static int ReadImageFile(const std::string &filename, uchar **imageData,
                           int *imageWidth, int *imageHeight,
                           int *numComponents);

  /////////////////////////////////////////////////////////////////////////////
  // Save an image to the output filename in PNG format.
  // Returns 1 if successful or 0 if unsuccessful.
  // The input image data is pointed to by imageData.
  // The image width, image height, and number of components (color channels)
  // per pixel are provided in imageWidth, imageHeight, numComponents.
  // The value of numComponents can be 1, 2, 3 or 4.
  // Note that some numComponents cannot be supported by some image file
  // formats. The input image data is assumed packed tightly with red, green,
  // blue, and alpha arranged from lower to higher memory addresses. Each color
  // channel take one byte. The first pixel (origin of the image) is at the
  // bottom-left of the image.
  /////////////////////////////////////////////////////////////////////////////

  static int SaveImageToFilePNG(const std::string &filename,
                                const uchar *imageData, int imageWidth,
                                int imageHeight, int numComponents);

  /////////////////////////////////////////////////////////////////////////////
  // Save an image to the output filename in JPEG format.
  // Returns 1 if successful or 0 if unsuccessful.
  // The input image data is pointed to by imageData.
  // The image width, image height, and number of components (color channels)
  // per pixel are provided in imageWidth, imageHeight, numComponents.
  // The value of numComponents can be 1, 2, 3 or 4.
  // Note that some numComponents cannot be supported by some image file
  // formats. The input image data is assumed packed tightly with red, green,
  // blue, and alpha arranged from lower to higher memory addresses. Each color
  // channel take one byte. The first pixel (origin of the image) is at the
  // bottom-left of the image. The quality value ranges from 1 to 100; default
  // is 90.
  /////////////////////////////////////////////////////////////////////////////

  static int SaveImageToFileJPEG(const std::string &filename,
                                 const uchar *imageData, int imageWidth,
                                 int imageHeight, int numComponents,
                                 int quality = 90);
};

#endif // _IMAGEIO_H_
