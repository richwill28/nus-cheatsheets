#ifndef _UTIL_H_
#define _UTIL_H_

#include <cmath>
#include <cstdlib>


typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned short ushort;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923 // PI/2
#endif

//============================================================================

#define CMalloc(mem_size) Util::_CheckedMalloc((mem_size), __FILE__, __LINE__)

class Util {
public:
  static void ErrorExit(const char *format, ...);
  // Outputs an error message to the stderr and exits program.

  static void ErrorExitLoc(const char *srcfile, int lineNum, const char *format,
                           ...);
  // Outputs an error message to the stderr and exits program.
  // Needs source file name and line number.

  static void ShowWarning(const char *format, ...);
  // Outputs a warning message to the stderr.

  static void ShowWarningLoc(const char *srcfile, int lineNum,
                             const char *format, ...);
  // Outputs a warning message to the stderr.
  // Needs source file name and line number.

  static double GetCurrRealTime();
  // Returns time in seconds (plus fraction of a second) since midnight
  // (00:00:00), January 1, 1970, coordinated universal time (UTC).

  static double GetCurrCPUTime();
  // Returns cpu time in seconds (plus fraction of a second) since the
  // start of the current process.

  static void *_CheckedMalloc(size_t size, const char *srcfile, int lineNum)
  // Same as malloc(), but checks for out-of-memory.
  {
    void *p = malloc(size);
    if (p == nullptr)
      ErrorExitLoc(srcfile, lineNum, "Cannot allocate memory.");
    return p;
  }

  //============================================================================

  static double fsqr(double f)
  // Returns the square of f.
  {
    return (f * f);
  }

  static float fsqr(float f)
  // Returns the square of f.
  {
    return (f * f);
  }

  static int sqr(int i)
  // Returns the square of i.
  {
    return (i * i);
  }

  static double fcube(double f)
  // Returns the cube of f.
  {
    return (f * f * f);
  }

  static float fcube(float f)
  // Returns the cube of f.
  {
    return (f * f * f);
  }

  static int cube(int i)
  // Returns the cube of i.
  {
    return (i * i * i);
  }

  template <typename Type> static Type Min2(Type a, Type b) {
    return ((a < b) ? a : b);
  }

  template <typename Type> static Type Max2(Type a, Type b) {
    return ((a > b) ? a : b);
  }

  template <typename Type> static Type Min3(Type a, Type b, Type c) {
    Type t = a;
    if (b < t)
      t = b;
    return ((c < t) ? c : t);
  }

  template <typename Type> static Type Max3(Type a, Type b, Type c) {
    Type t = a;
    if (b > t)
      t = b;
    return ((c > t) ? c : t);
  }

  template <typename Type> static Type Clamp(Type x, Type low, Type hi) {
    if (x < low)
      return low;
    else if (x > hi)
      return hi;
    return x;
  }

  static int ClampToNearestInt(double x, int low, int hi) {
    int i = (int)floor(x + 0.5);
    if (i < low)
      i = low;
    else if (i > hi)
      i = hi;
    return i;
  }

  template <typename Type>
  static void CopyArrayN(Type dest[], const Type src[], size_t size) {
    memcpy(dest, src, size * sizeof(Type));
  }

  template <typename Type>
  static void CopyArray4(Type dest[4], const Type src[4]) {
    // memcpy( dest, src, 4 * sizeof(Type) );
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[3] = src[3];
  }

  template <typename Type>
  static void CopyArray3(Type dest[3], const Type src[3]) {
    // memcpy( dest, src, 3 * sizeof(Type) );
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
  }

  template <typename Type>
  static void CopyArray2(Type dest[2], const Type src[2]) {
    // memcpy( dest, src, 2 * sizeof(Type) );
    dest[0] = src[0];
    dest[1] = src[1];
  }

}; // Util

#endif // _UTIL_H_
