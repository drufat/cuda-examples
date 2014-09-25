#ifndef GEARS_H
#define GEARS_H

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

typedef struct {
    GLuint gear1;
    GLuint gear2;
    GLuint gear3;
    int xRot;
    int yRot;
    int zRot;
    int gear1Rot;
} Gears;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void gears_initialize(Gears *g);

void gears_paint(const Gears *g);

void gears_resize(int width, int height);

void gears_advance(Gears *g);

GLuint gears_make(const GLfloat *reflectance, GLdouble innerRadius,
                  GLdouble outerRadius, GLdouble thickness,
                  GLdouble toothSize, GLint toothCount);

void gears_draw(GLuint gear, GLdouble dx, GLdouble dy, GLdouble dz,
                GLdouble angle);

void gears_normalize_angle(int *angle);

#endif // GEARS_H
