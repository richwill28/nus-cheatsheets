
/* Two-Dimensional Sierpinski Gasket          */
/* Generated Using Randomly Selected Vertices */
/* And Bisection                              */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

void myinit() {

  /* attributes */

  glClearColor(1.0, 1.0, 1.0, 1.0); /* white background */
  glColor3f(1.0, 0.0, 0.0);         /* draw in red */

  /* set up viewing */
  /* 500 x 500 window with origin lower left */

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, 50.0, 0.0, 50.0);
  glMatrixMode(GL_MODELVIEW);
}

void display(void) {

  GLfloat vertices[3][2] = {
      {0.0, 0.0}, {25.0, 50.0}, {50.0, 0.0}}; /* A triangle */

  int j, k;
  GLfloat p[2] = {7.5, 5.0}; /* An arbitrary initial point inside traingle */

  glClear(GL_COLOR_BUFFER_BIT); /*clear the window */

  /* compute and plots 5000 new points */

  glBegin(GL_POINTS);

  for (k = 0; k < 5000; k++) {
    j = rand() % 3; /* pick a vertex at random */

    /* Compute point halfway between selected vertex and old point */

    p[0] = (p[0] + vertices[j][0]) / 2.0;
    p[1] = (p[1] + vertices[j][1]) / 2.0;

    /* plot new point */

    glVertex2fv(p);
  }
  glEnd();
  glFlush(); /* clear buffers */
}

int main(int argc, char **argv) {

  /* Standard GLUT initialization */

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB); /* default, not needed */
  glutInitWindowSize(500, 500);                /* 500 x 500 pixel window */
  glutInitWindowPosition(0, 0);          /* place window top left on display */
  glutCreateWindow("Sierpinski Gasket"); /* window title */
  glutDisplayFunc(display); /* display callback invoked when window opened */

  myinit(); /* set attributes */

  glutMainLoop(); /* enter event loop */
}
