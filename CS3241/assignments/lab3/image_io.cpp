#include <stdio.h>
#include <stdlib.h>


#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "image_io.h"

/////////////////////////////////////////////////////////////////////////////
// Deallocate the memory allocated to (*imageData) returned by
// the function ReadImageFile().
// (*imageData) will be set to NULL.
/////////////////////////////////////////////////////////////////////////////

void DeallocateImageData(uchar **imageData) {
  stbi_image_free(*imageData);
  (*imageData) = NULL;
}

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

int ReadImageFile(const char *filename, uchar **imageData, int *imageWidth,
                  int *imageHeight, int *numComponents) {
  // Enable flipping of images vertically when read in.
  // This is to follow OpenGL's image coordinate system, i.e. bottom-leftmost is
  // (0, 0).
  stbi_set_flip_vertically_on_load(true);

  int w, h, n;
  unsigned char *data = stbi_load(filename, &w, &h, &n, 0);

  if (data == NULL) {
    fprintf(stderr, "Error: Cannot read image file %s.\n", filename);
    return 0;
  } else {
    *imageData = data;
    *imageWidth = w;
    *imageHeight = h;
    *numComponents = n;
    return 1;
  }
}

/////////////////////////////////////////////////////////////////////////////
// Save an image to the output filename in PNG format.
// Returns 1 if successful or 0 if unsuccessful.
// The input image data is pointed to by imageData.
// The image width, image height, and number of components (color channels)
// per pixel are provided in imageWidth, imageHeight, numComponents.
// The value of numComponents can be 1, 2, 3 or 4.
// Note that some numComponents cannot be supported by some image file formats.
// The input image data is assumed packed tightly with red, green, blue,
// and alpha arranged from lower to higher memory addresses.
// Each color channel take one byte.
// The first pixel (origin of the image) is at the bottom-left of the image.
/////////////////////////////////////////////////////////////////////////////

int SaveImageToFilePNG(const char *filename, const uchar *imageData,
                       int imageWidth, int imageHeight, int numComponents) {
  stbi_flip_vertically_on_write(true);

  int write_status = stbi_write_png(filename, imageWidth, imageHeight,
                                    numComponents, imageData, 0);

  if (write_status == 0) {
    fprintf(stderr, "Error: Cannot write image file %s.\n", filename);
    return 0;
  } else {
    return 1;
  }
}

/////////////////////////////////////////////////////////////////////////////
// Save an image to the output filename in JPEG format.
// Returns 1 if successful or 0 if unsuccessful.
// The input image data is pointed to by imageData.
// The image width, image height, and number of components (color channels)
// per pixel are provided in imageWidth, imageHeight, numComponents.
// The value of numComponents can be 1, 2, 3 or 4.
// Note that some numComponents cannot be supported by some image file formats.
// The input image data is assumed packed tightly with red, green, blue,
// and alpha arranged from lower to higher memory addresses.
// Each color channel take one byte.
// The first pixel (origin of the image) is at the bottom-left of the image.
// The quality value ranges from 1 to 100; default is 90.
/////////////////////////////////////////////////////////////////////////////

int SaveImageToFileJPEG(const char *filename, const uchar *imageData,
                        int imageWidth, int imageHeight, int numComponents,
                        int quality) {
  stbi_flip_vertically_on_write(true);

  int write_status = stbi_write_jpg(filename, imageWidth, imageHeight,
                                    numComponents, imageData, quality);

  if (write_status == 0) {
    fprintf(stderr, "Error: Cannot write image file %s.\n", filename);
    return 0;
  } else {
    return 1;
  }
}
