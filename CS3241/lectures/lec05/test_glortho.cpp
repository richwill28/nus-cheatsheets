#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

void display() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glBegin(GL_TRIANGLES);
  glColor3f(1.0, 0.0, 0.0);
  glVertex3f(0.9, -0.9, 0.2);
  glVertex3f(-0.9, 0.9, 0.2);
  glVertex3f(-0.9, -0.9, 0.2);
  glColor3f(0.0, 1.0, 0.0);
  glVertex3f(0.9, -0.9, -0.5);
  glVertex3f(0.9, 0.9, -0.5);
  glVertex3f(-0.9, -0.9, -0.5);
  glEnd();

  glutSwapBuffers();
}

void myReshape(int w, int h) {
  glViewport(0, 0, w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(600, 600);
  glutCreateWindow("Testing glOrtho");

  glutDisplayFunc(display);
  glutReshapeFunc(myReshape);

  glClearColor(0.0, 0.0, 0.0, 1.0); // Set black background color.
  glEnable(GL_DEPTH_TEST); // Use depth-buffer for hidden surface removal.
  glShadeModel(GL_SMOOTH);

  glutMainLoop();
}
