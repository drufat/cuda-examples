#include <math.h>
#include <stdlib.h>
#include <stdio.h>
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

static void
key(unsigned char k, int x, int y)
{
    switch (k) {
    case 27:  /* Escape */
        exit(0);
        break;
    default:
        return;
    }
    glutPostRedisplay();
}

static void
special(int k, int x, int y)
{
    switch (k) {
    //  case GLUT_KEY_UP:
    //    view_rotx += 5.0;
    //    break;
    //  case GLUT_KEY_DOWN:
    //    view_rotx -= 5.0;
    //    break;
    //  case GLUT_KEY_LEFT:
    //    view_roty += 5.0;
    //    break;
    //  case GLUT_KEY_RIGHT:
    //    view_roty -= 5.0;
    //    break;
    default:
        return;
    }
    glutPostRedisplay();
}

void __display(void)
{
    gears_paint(&g);
    glutSwapBuffers();
}

void idle(void)
{
    gears_advance(&g);
    glutPostRedisplay();
}

void visible (int vis)
{
    if (vis == GLUT_VISIBLE)
        glutIdleFunc(idle);
    else
        glutIdleFunc(NULL);
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

    glutInitWindowPosition(100, 100);
    glutInitWindowSize(300, 300);
    glutCreateWindow("Gears GLUT");

    gears_initialize(&g);
    gears_resize(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));

    glutDisplayFunc( __display );
    glutReshapeFunc( gears_resize );
    glutKeyboardFunc(key);
    glutSpecialFunc(special);
    glutVisibilityFunc( visible );

    glutMainLoop();
    return 0;
}
