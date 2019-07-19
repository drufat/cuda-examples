#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

extern "C" {
#include "gears.h"
}

static Gears g = {0, 0, 0, 0, 0, 0, 0};

static void key(unsigned char k, int x, int y) {
  switch (k) {
  case 27: /* Escape */
    exit(0);
  default:
    return;
  }
}

void __display(void) {
  gears_paint(&g);
  glutSwapBuffers();
}

void idle(void) {
  gears_advance(&g);
  glutPostRedisplay();
}

void visible(int vis) {
  if (vis == GLUT_VISIBLE) {
    glutIdleFunc(idle);
  } else {
    glutIdleFunc(nullptr);
  }
}

int main(int argc, char *argv[]) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

  glutInitWindowPosition(100, 100);
  glutInitWindowSize(300, 300);
  glutCreateWindow("Gears GLUT");

  gears_initialize(&g);
  gears_resize(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));

  glutDisplayFunc(__display);
  glutReshapeFunc(gears_resize);
  glutKeyboardFunc(key);
  glutVisibilityFunc(visible);

  glutMainLoop();
  return 0;
}
