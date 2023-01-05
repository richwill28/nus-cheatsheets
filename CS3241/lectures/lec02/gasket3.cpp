/* recursive subdivision of a tetrahedron to form 3D Sierpinski gasket */
/* number of recursive steps given on command line */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

/* initial tetrahedron */

GLfloat v[4][3] = {{0.0, 0.0, 1.0},
                   {0.0, 0.942809, -0.33333},
                   {-0.816497, -0.471405, -0.333333},
                   {0.816497, -0.471405, -0.333333}};

GLfloat colors[4][3] = {
    {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}};

int n;

void triangle(GLfloat *va, GLfloat *vb, GLfloat *vc) {
  glVertex3fv(va);
  glVertex3fv(vb);
  glVertex3fv(vc);
}

void tetra(GLfloat *a, GLfloat *b, GLfloat *c, GLfloat *d) {
  glColor3fv(colors[0]);
  triangle(a, b, c);
  glColor3fv(colors[1]);
  triangle(a, c, d);
  glColor3fv(colors[2]);
  triangle(a, d, b);
  glColor3fv(colors[3]);
  triangle(b, d, c);
}

void divide_tetra(GLfloat *a, GLfloat *b, GLfloat *c, GLfloat *d, int m) {

  GLfloat mid[6][3];
  int j;
  if (m > 0) {
    /* compute six midpoints */

    for (j = 0; j < 3; j++)
      mid[0][j] = (a[j] + b[j]) / 2;
    for (j = 0; j < 3; j++)
      mid[1][j] = (a[j] + c[j]) / 2;
    for (j = 0; j < 3; j++)
      mid[2][j] = (a[j] + d[j]) / 2;
    for (j = 0; j < 3; j++)
      mid[3][j] = (b[j] + c[j]) / 2;
    for (j = 0; j < 3; j++)
      mid[4][j] = (c[j] + d[j]) / 2;
    for (j = 0; j < 3; j++)
      mid[5][j] = (b[j] + d[j]) / 2;

    /* create 4 tetrahedrons by subdivision */

    divide_tetra(a, mid[0], mid[1], mid[2], m - 1);
    divide_tetra(mid[0], b, mid[3], mid[5], m - 1);
    divide_tetra(mid[1], mid[3], c, mid[4], m - 1);
    divide_tetra(mid[2], mid[4], d, mid[5], m - 1);

  } else
    (tetra(a, b, c, d)); /* draw tetrahedron at end of recursion */
}

void display() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glBegin(GL_TRIANGLES);
  divide_tetra(v[0], v[1], v[2], v[3], n);
  glEnd();
  glFlush();
}

void myReshape(int w, int h) {
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  if (w <= h)
    glOrtho(-2.0, 2.0, -2.0 * (GLfloat)h / (GLfloat)w,
            2.0 * (GLfloat)h / (GLfloat)w, -10.0, 10.0);
  else
    glOrtho(-2.0 * (GLfloat)w / (GLfloat)h, 2.0 * (GLfloat)w / (GLfloat)h, -2.0,
            2.0, -10.0, 10.0);
  glMatrixMode(GL_MODELVIEW);
  glutPostRedisplay();
}

int main(int argc, char **argv) {
  // n = atoi(argv[1]); /* or set number of subdivision steps here */
  n = 4; /* number of subdivision steps */
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(500, 500);
  glutCreateWindow("3D Gasket");
  glutReshapeFunc(myReshape);
  glutDisplayFunc(display);
  glEnable(GL_DEPTH_TEST);
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glutMainLoop();
}
