//============================================================
// STUDENT NAME: Richard Willie
// NUS User ID.: A0219710L
// COMMENTS TO GRADER:
//   - Developed on Manjaro
//   - Tested with GCC 12.1.1 and GLUT 3.2.2
//   - How to compile:
//       - cd <path to CMakeLists.txt>
//       - cmake -S ./ -B ./build
//       - cmake --build ./build
//============================================================

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

/////////////////////////////////////////////////////////////////////////////
// CONSTANTS
/////////////////////////////////////////////////////////////////////////////

#define PI 3.1415926535897932384626433832795

#define MAX_NUM_OF_DISCS 200 // Limit the number of discs.
#define MIN_RADIUS 10.0      // Minimum radius of disc.
#define MAX_RADIUS 50.0      // Maximum radius of disc.
#define NUM_OF_SIDES 18      // Number of polygon sides to approximate a disc.

#define MIN_X_SPEED 1.0  // Minimum speed of disc in X direction.
#define MAX_X_SPEED 20.0 // Maximum speed of disc in X direction.
#define MIN_Y_SPEED 1.0  // Minimum speed of disc in Y direction.
#define MAX_Y_SPEED 20.0 // Maximum speed of disc in Y direction.

#define DESIRED_FPS 30 // Desired number of frames per second.
#define MSEC_BETWEEN_FRAMES (unsigned int)1000 / DESIRED_FPS

/////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES
/////////////////////////////////////////////////////////////////////////////

typedef struct discType {
  double pos[2];   // The X and Y coordinates of the center of the disc.
  double speed[2]; // The velocity of the disc in X and Y directions. Can be
                   // negative.
  double radius;   // Radius of the disc.
  unsigned char color[3]; // RGB color of the disc. Each value is 0 to 255.
} discType;

/*
The 2D space in which the discs are located spans from the coordinates [0, 0],
which corresponds to the bottom-leftmost corner of the display window, to the
coordinates [winWidth, winHeight], which corresponds to the top-rightmost
corner of the display window.

The speed is measured as the distance moved in the above 2D space per
render frame time.
*/

int numDiscs = 0; // Number of discs that have been added.

discType disc[MAX_NUM_OF_DISCS]; // Array for storing discs.

bool drawWireframe = false; // Draw polygons in wireframe if true,
// otherwise polygons are filled.

int winWidth = 800;  // Window width in pixels.
int winHeight = 600; // Window height in pixels.

std::random_device rd;  // Obtain a seed for the random number engine.
std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd.

double randRadius(double minRadius, double maxRadius) {
  std::uniform_real_distribution<double> rdRadius(minRadius, maxRadius);
  return rdRadius(gen);
}

double randSpeed(double minSpeed, double maxSpeed) {
  std::uniform_real_distribution<double> rdSpeed(minSpeed, maxSpeed);
  std::uniform_int_distribution<int> rdDirection(0, 1);
  return rdSpeed(gen) * (rdDirection(gen) ? 1 : -1);
}

unsigned char randColor() {
  std::uniform_int_distribution<unsigned char> rdColor(0, 255);
  return rdColor(gen);
}

/////////////////////////////////////////////////////////////////////////////
// Draw the disc in its color using GL_TRIANGLE_FAN.
/////////////////////////////////////////////////////////////////////////////

void DrawDisc(const discType *d) {
  static bool firstTime = true;
  static double unitDiscVertex[NUM_OF_SIDES + 1][2];

  if (firstTime) {
    firstTime = false;

    // Pre-compute and store the vertices' positions of a unit-radius disc.
    for (size_t i = 0; i < NUM_OF_SIDES + 1; i++) {
      unitDiscVertex[i][0] = cos(i * 2 * PI / NUM_OF_SIDES);
      unitDiscVertex[i][1] = sin(i * 2 * PI / NUM_OF_SIDES);
    }
  }

  // Draw the disc in its color as a GL_TRIANGLE_FAN.
  glColor3ubv(d->color);
  glBegin(GL_TRIANGLE_FAN);

  // Draw the center of the disc.
  glVertex2d(d->pos[0], d->pos[1]);

  // Draw all the other vertices.
  for (size_t i = 0; i < NUM_OF_SIDES + 1; i++) {
    double x = d->pos[0] + d->radius * unitDiscVertex[i][0];
    double y = d->pos[1] + d->radius * unitDiscVertex[i][1];
    glVertex2d(x, y);
  }

  glEnd();
}

/////////////////////////////////////////////////////////////////////////////
// The display callback function.
/////////////////////////////////////////////////////////////////////////////

void MyDisplay(void) {
  glClear(GL_COLOR_BUFFER_BIT);

  if (drawWireframe) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  } else {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }

  for (size_t i = 0; i < numDiscs; i++) {
    DrawDisc(&disc[i]);
  }

  glutSwapBuffers();
}

/////////////////////////////////////////////////////////////////////////////
// The mouse callback function.
// If mouse left button is pressed, a new disc is created with its center
// at the mouse cursor position. The new disc is assigned the followings:
// (1) a random radius between MIN_RADIUS and MAX_RADIUS,
// (2) a random speed in X direction in the range
//     from -MAX_X_SPEED to -MIN_X_SPEED, and from MIN_X_SPEED to MAX_X_SPEED.
// (3) a random speed in Y direction in the range
//     from -MAX_Y_SPEED to -MIN_Y_SPEED, and from MIN_Y_SPEED to MAX_Y_SPEED.
// (4) R, G, B color, each ranges from 0 to 255.
/////////////////////////////////////////////////////////////////////////////

void MyMouse(int btn, int state, int x, int y) {
  if (btn == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
    return;
  }

  if (numDiscs >= MAX_NUM_OF_DISCS) {
    printf("Already reached maximum number of discs.\n");
    return;
  }

  disc[numDiscs].pos[0] = x;
  disc[numDiscs].pos[1] = winHeight - 1 - y;

  disc[numDiscs].radius = randRadius(MIN_RADIUS, MAX_RADIUS);

  disc[numDiscs].speed[0] = randSpeed(MIN_X_SPEED, MAX_X_SPEED);
  disc[numDiscs].speed[1] = randSpeed(MIN_Y_SPEED, MAX_Y_SPEED);

  disc[numDiscs].color[0] = randColor();
  disc[numDiscs].color[1] = randColor();
  disc[numDiscs].color[2] = randColor();

  numDiscs++;

  glutPostRedisplay();
}

/////////////////////////////////////////////////////////////////////////////
// The reshape callback function.
// It also sets up the viewing.
/////////////////////////////////////////////////////////////////////////////

void MyReshape(int w, int h) {
  // Update window width and height.
  winWidth = w;
  winHeight = h;

  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, w, 0, h);
  glMatrixMode(GL_MODELVIEW);
  glutPostRedisplay();
}

/////////////////////////////////////////////////////////////////////////////
// The keyboard callback function.
/////////////////////////////////////////////////////////////////////////////

void MyKeyboard(unsigned char key, int x, int y) {
  switch (key) {
  // Quit program.
  case 'q':
  case 'Q':
    exit(0);
    break;

  // Toggle wireframe or filled polygons.
  case 'w':
  case 'W':
    drawWireframe = !drawWireframe;
    glutPostRedisplay();
    break;

  // Reset and erase all discs.
  case 'r':
  case 'R':
    numDiscs = 0;
    glutPostRedisplay();
    break;
  }
}

/////////////////////////////////////////////////////////////////////////////
// Updates the positions of all the discs.
//
// Increments the position of each disc by its speed in each of the
// X and Y directions. Note that the speeds can be negative.
// At its new position, if the disc is entirely or partially outside the
// left window boundary, then shift it right so that it is inside the
// window and just touches the left window boundary. Its speed in the X
// direction must now be reversed (negated). Similar approach is
// applied for the cases of the right, top, and bottom window boundaries.
/////////////////////////////////////////////////////////////////////////////

void UpdateAllDiscPos(void) {
  for (int i = 0; i < numDiscs; i++) {
    // Update disc position.
    disc[i].pos[0] += disc[i].speed[0];
    disc[i].pos[1] += disc[i].speed[1];

    if (disc[i].pos[0] <= disc[i].radius) {
      // Left window boundary.
      disc[i].pos[0] = disc[i].radius;
      disc[i].speed[0] *= -1;
    } else if (disc[i].pos[0] + disc[i].radius >= winWidth) {
      // Right window boundary.
      disc[i].pos[0] = winWidth - disc[i].radius;
      disc[i].speed[0] *= -1;
    }

    if (disc[i].pos[1] <= disc[i].radius) {
      // Bottom window boundary.
      disc[i].pos[1] = disc[i].radius;
      disc[i].speed[1] *= -1;
    } else if (disc[i].pos[1] + disc[i].radius >= winHeight) {
      // Top window boundary.
      disc[i].pos[1] = winHeight - disc[i].radius;
      disc[i].speed[1] *= -1;
    }
  }

  glutPostRedisplay();
}

/////////////////////////////////////////////////////////////////////////////
// The timer callback function.
/////////////////////////////////////////////////////////////////////////////

void MyTimer(int v) {
  UpdateAllDiscPos();
  glutTimerFunc(MSEC_BETWEEN_FRAMES, MyTimer, 0);
}

/////////////////////////////////////////////////////////////////////////////
// The init function. It initializes some OpenGL states.
/////////////////////////////////////////////////////////////////////////////

void MyInit(void) {
  glClearColor(0.0, 0.0, 0.0, 1.0); // Black background color.
  glShadeModel(GL_FLAT);
}

static void WaitForEnterKeyBeforeExit(void) {
  printf("Press Enter to exit.\n");
  fflush(stdin);
  getchar();
}

/////////////////////////////////////////////////////////////////////////////
// The main function.
/////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  atexit(WaitForEnterKeyBeforeExit); // atexit() is declared in stdlib.h

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowSize(winWidth, winHeight);
  glutCreateWindow("main");

  MyInit();

  // Register the callback functions.
  glutDisplayFunc(MyDisplay);
  glutReshapeFunc(MyReshape);
  glutMouseFunc(MyMouse);
  glutKeyboardFunc(MyKeyboard);
  glutTimerFunc(0, MyTimer, 0);

  // Display user instructions in console window.
  printf("Click LEFT MOUSE BUTTON in window to add new disc.\n");
  printf("Press 'w' to toggle wireframe.\n");
  printf("Press 'r' to erase all discs.\n");
  printf("Press 'q' to quit.\n\n");

  // Enter GLUT event loop.
  glutMainLoop();
  return 0;
}
