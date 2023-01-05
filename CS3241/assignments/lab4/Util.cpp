// To disable deprecation warnings for using vsprintf() and _ftime64().
#define _CRT_SECURE_NO_WARNINGS

#include "Util.h"
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/timeb.h>
#include <sys/types.h>


#define MSG_BUF_SIZE 1024

void Util::ErrorExit(const char *format, ...)
// Outputs an error message to the stderr and exits program.
{
  va_list args;
  static char buffer[MSG_BUF_SIZE];
  va_start(args, format);
  vsprintf(buffer, format, args);
  va_end(args);
  fprintf(stderr, "ERROR: %s\n\n", buffer);
  exit(1);
}

void Util::ErrorExitLoc(const char *srcfile, int lineNum, const char *format,
                        ...)
// Outputs an error message to the stderr and exits program.
// Needs source file name and line number.
{
  va_list args;
  static char buffer[MSG_BUF_SIZE];
  va_start(args, format);
  vsprintf(buffer, format, args);
  va_end(args);
  fprintf(stderr, "ERROR at \"%s\" (line %d):\n%s\n\n", srcfile, lineNum,
          buffer);
  exit(1);
}

void Util::ShowWarning(const char *format, ...)
// Outputs a warning message to the stderr.
{
  va_list args;
  static char buffer[MSG_BUF_SIZE];
  va_start(args, format);
  vsprintf(buffer, format, args);
  va_end(args);
  fprintf(stderr, "WARNING: %s\n\n", buffer);
}

void Util::ShowWarningLoc(const char *srcfile, int lineNum, const char *format,
                          ...)
// Outputs a warning message to the stderr.
// Needs source file name and line number.
{
  va_list args;
  static char buffer[MSG_BUF_SIZE];
  va_start(args, format);
  vsprintf(buffer, format, args);
  va_end(args);
  fprintf(stderr, "WARNING at \"%s\" (line %d):\n%s\n\n", srcfile, lineNum,
          buffer);
}

//============================================================================

double Util::GetCurrRealTime()
// Returns time in seconds (plus fraction of a second) since midnight
// (00:00:00), January 1, 1970, coordinated universal time (UTC).
{
#ifdef _WIN32
  struct _timeb timebuffer {};
  _ftime(&timebuffer);
#else
  struct timeb timebuffer;
  ftime(&timebuffer);
#endif
  return ((double)timebuffer.time + ((double)timebuffer.millitm / 1000.0));
}

double Util::GetCurrCPUTime()
// Returns cpu time in seconds (plus fraction of a second) since the
// start of the current process.
{
  return ((double)clock()) / CLOCKS_PER_SEC;
}
