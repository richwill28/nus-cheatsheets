
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

void display() {
  glClear(GL_COLOR_BUFFER_BIT);
  glFlush();
}

void mouse(int btn, int state, int x, int y) {}

void motion(int x, int y) { printf("x = %d, y = %d\n", x, y); }

void myReshape(int w, int h) {
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  if (w <= h)
    glOrtho(-50.0, 50.0, -50.0 * (GLfloat)h / (GLfloat)w,
            50.0 * (GLfloat)h / (GLfloat)w, -1.0, 1.0);
  else
    glOrtho(-50.0 * (GLfloat)w / (GLfloat)h, 50.0 * (GLfloat)w / (GLfloat)h,
            -50.0, 50.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutCreateWindow("test_mouse");
  glutDisplayFunc(display);
  glutReshapeFunc(myReshape);
  glutMouseFunc(mouse);
  glutPassiveMotionFunc(motion);
  glutMainLoop();
}
