#ifndef GLFLUIDS_H
#define GLFLUIDS_H

// Qt
#include <QGLWidget>
#include <QGLFunctions>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>

#include "defines.h"
#include "fluidsGL_kernels.h"

class GLFluids : public QGLWidget, protected QGLFunctions
{
    Q_OBJECT

public:
    GLFluids(QWidget *parent = 0);
    ~GLFluids();
    void reset();

protected:

    void initializeGL();
    void paintGL();
    void resizeGL(int x, int y);

    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);

private:
    void simulateFluids();

    float2 *vxfield;
    float2 *vyfield;

    float2 *hvfield;
    float2 *dvfield;

    int wWidth;
    int wHeight;
    int lastx = 0, lasty = 0;

    // Particle data
    GLuint vbo;                 // OpenGL vertex buffer object
    struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
    float2 *particles; // particle positions in host memory

    // CUFFT plan handle
    cufftHandle planr2c;
    cufftHandle planc2r;
};


#endif // GLFLUIDS_H
