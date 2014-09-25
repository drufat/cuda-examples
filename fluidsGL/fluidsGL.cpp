/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
// OpenGL Graphics includes
#include <GL/glew.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include "helper_timer.h"

#include "defines.h"
#include "fluidsGL_kernels.h"

const char *sSDKname = "fluidsGL";
// CUDA example code that implements the frequency space version of
// Jos Stam's paper 'Stable Fluids' in 2D. This application uses the
// CUDA FFT library (CUFFT) to perform velocity diffusion and to
// force non-divergence in the velocity field at each time step. It uses
// CUDA-OpenGL interoperability to update the particle field directly
// instead of doing a copy to system memory before drawing. Texture is
// used for automatic bilinear interpolation at the velocity advection step.

void cleanup(void);
void reshape(int x, int y);

// CUFFT plan handle
cufftHandle planr2c;
cufftHandle planc2r;

static float2 *vxfield = NULL;
static float2 *vyfield = NULL;

float2 *hvfield = NULL;
float2 *dvfield = NULL;
static int wWidth  = std::max(512, DIM);
static int wHeight = std::max(512, DIM);

static int clicked  = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

// Particle data
static GLuint vbo = 0;                 // OpenGL vertex buffer object
static struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
static float2 *particles = NULL; // particle positions in host memory
static int lastx = 0, lasty = 0;

// Texture pitch
size_t tPitch = 0;

bool g_bExitESC = false;

void simulateFluids(void)
{
    // simulate fluid
    advectVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM, DT);
    diffuseProject(vxfield, vyfield, CPADW, DIM, DT, VIS,
                   planr2c, planc2r);
    updateVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM);
    advectParticles(cuda_vbo_resource, dvfield, DIM, DIM, DT);
}


void display(void)
{

    sdkStartTimer(&timer);
    simulateFluids();

    // render points
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(1, 1, 1, 1.0f);
    glColor4f(0, 0, 1, 0.5f);
    glPointSize(1);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, 0, NULL);
    glDrawArrays(GL_POINTS, 0, DS);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);

    // Finish timing before swap buffers to avoid refresh sync
    sdkStopTimer(&timer);
    glutSwapBuffers();

    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Cuda/GL Stable Fluids (%d x %d): %3.1f fps", DIM, DIM, ifps);
        glutSetWindowTitle(fps);
        fpsCount = 0;
        fpsLimit = (int)std::max(ifps, 1.f);
        sdkResetTimer(&timer);
    }

    glutPostRedisplay();
}


// very simple von neumann middle-square prng.  can't use rand() in -qatest
// mode because its implementation varies across platforms which makes testing
// for consistency in the important parts of this program difficult.
float myrand(void)
{
    static int seed = 72191;
    char sq[22];
    return rand()/(float)RAND_MAX;
}

void initParticles(float2 *p, int dx, int dy)
{
    int i, j;

    for (i = 0; i < dy; i++)
    {
        for (j = 0; j < dx; j++)
        {
            p[i*dx+j].x = (j+0.5f+(myrand() - 0.5f))/dx;
            p[i*dx+j].y = (i+0.5f+(myrand() - 0.5f))/dy;
        }
    }
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 27:
        g_bExitESC = true;
#if defined (__APPLE__) || defined(MACOSX)
        exit(EXIT_SUCCESS);
#else
        glutDestroyWindow(glutGetWindow());
        return;
#endif
        break;

    case 'r':
        memset(hvfield, 0, sizeof(float2) * DS);
        cudaMemcpy(dvfield, hvfield, sizeof(float2) * DS,
                   cudaMemcpyHostToDevice);

        initParticles(particles, DIM, DIM);

        cudaGraphicsUnregisterResource(cuda_vbo_resource);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * DS,
                     particles, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
        break;

    default:
        break;
    }
}

void click(int button, int updown, int x, int y)
{
    lastx = x;
    lasty = y;
    clicked = !clicked;
}

void motion(int x, int y)
{
    // Convert motion coordinates to domain
    float fx = (lastx / (float)wWidth);
    float fy = (lasty / (float)wHeight);
    int nx = (int)(fx * DIM);
    int ny = (int)(fy * DIM);

    if (clicked && nx < DIM-FR && nx > FR-1 && ny < DIM-FR && ny > FR-1)
    {
        int ddx = x - lastx;
        int ddy = y - lasty;
        fx = ddx / (float)wWidth;
        fy = ddy / (float)wHeight;
        int spy = ny-FR;
        int spx = nx-FR;
        addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
        lastx = x;
        lasty = y;
    }

    glutPostRedisplay();
}

void reshape(int x, int y)
{
    wWidth = x;
    wHeight = y;
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 1, 0, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}

void cleanup(void)
{
    cudaGraphicsUnregisterResource(cuda_vbo_resource);

    unbind_texture();
    delete_texture();

    // Free all host and device resources
    free(hvfield);
    free(particles);
    cudaFree(dvfield);
    cudaFree(vxfield);
    cudaFree(vyfield);
    cufftDestroy(planr2c);
    cufftDestroy(planc2r);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vbo);

    sdkDeleteTimer(&timer);

    if (g_bExitESC)
    {
        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
    }
}

int initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    //glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("Compute Stable Fluids");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(click);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);


    glewInit();

    if (! glewIsSupported(
                "GL_ARB_vertex_buffer_object"
                ))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    return true;
}


int main(int argc, char **argv)
{

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        exit(EXIT_SUCCESS);
    }

    // Allocate and initialize host data

    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    hvfield = (float2 *)malloc(sizeof(float2) * DS);
    memset(hvfield, 0, sizeof(float2) * DS);

    // Allocate and initialize device data
    cudaMallocPitch((void **)&dvfield, &tPitch, sizeof(float2)*DIM, DIM);

    cudaMemcpy(dvfield, hvfield, sizeof(float2) * DS,
               cudaMemcpyHostToDevice);
    // Temporary complex velocity field data
    cudaMalloc((void **)&vxfield, sizeof(float2) * PDS);
    cudaMalloc((void **)&vyfield, sizeof(float2) * PDS);

    setup_texture(DIM, DIM);
    bind_texture();

    // Create particle array
    particles = (float2 *)malloc(sizeof(float2) * DS);
    memset(particles, 0, sizeof(float2) * DS);

    initParticles(particles, DIM, DIM);

    // Create CUFFT transform plan configuration
    cufftPlan2d(&planr2c, DIM, DIM, CUFFT_R2C);
    cufftPlan2d(&planc2r, DIM, DIM, CUFFT_C2R);
    // TODO: update kernels to use the new unpadded memory layout for perf
    // rather than the old FFTW-compatible layout
    cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_FFTW_PADDING);
    cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_FFTW_PADDING);


    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * DS,
                 particles, GL_DYNAMIC_DRAW);

    GLint bsize;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);
    if (bsize != (sizeof(float2) * DS))
        goto EXTERR;

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif
    glutMainLoop();

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(EXIT_SUCCESS);

    return 0;

EXTERR:
    printf("Failed to initialize GL extensions.\n");

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(EXIT_FAILURE);
}
