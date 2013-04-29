// OpenGL Graphics includes
#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CUDA Libraries
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

#include "defines.h"
#include "fluidsGL_kernels.h"

#define MAX_EPSILON_ERROR 1.0f

const char *sSDKname = "fluidsGL";

#define getmin(a,b) (a < b ? a : b)
#define getmax(a,b) (a > b ? a : b)

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
static cData *vxfield = NULL;
static cData *vyfield = NULL;

cData *hvfield = NULL;
cData *dvfield = NULL;
static int wWidth = getmax(512,DIM);
static int wHeight = getmax(512,DIM);

static int clicked = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int timer;

// Particle data
GLuint vbo = 0;                 // OpenGL vertex buffer object
struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
static cData *particles = NULL; // particle positions in host memory
static int lastx = 0, lasty = 0;

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit

bool g_bQAReadback     = false;
bool g_bQAAddTestForce = true;
int  g_iFrameToCompare = 100;
int  g_TotalErrors     = 0;

extern "C" void addForces(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r);
extern "C" void advectVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt);
extern "C" void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt, float visc);
extern "C" void updateVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy);
extern "C" void advectParticles(GLuint vbo, cData *v, int dx, int dy, float dt);

void simulateFluids(void)
{
    // simulate fluid
    advectVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIM, RPADW, DIM, DT);
    diffuseProject(vxfield, vyfield, CPADW, DIM, DT, VIS);
    updateVelocity(dvfield, (float*)vxfield, (float*)vyfield, DIM, RPADW, DIM);
    advectParticles(vbo, dvfield, DIM, DIM, DT);
}

void display(void) {

    simulateFluids();

    // render points from vertex buffer
    glClear(GL_COLOR_BUFFER_BIT);
    glColor4f(0,1,0,0.5f); glPointSize(1);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
    glVertexPointer(2, GL_FLOAT, 0, NULL);
    glDrawArrays(GL_POINTS, 0, DS);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);

    // Finish timing before swap buffers to avoid refresh sync
    glutSwapBuffers();

    //    fpsCount++;
    //    if (fpsCount == fpsLimit) {
    //        char fps[256];
    //        float ifps = 100.0;
    //        sprintf(fps, "Cuda/GL Stable Fluids (%d x %d): %3.1f fps", DIM, DIM, ifps);
    //        glutSetWindowTitle(fps);
    //        fpsCount = 0;
    //        fpsLimit = (int)getmax(ifps, 1.f);
    //    }

    glutPostRedisplay();
}

// very simple von neumann middle-square prng.  can't use rand() in -qatest
// mode because its implementation varies across platforms which makes testing
// for consistency in the important parts of this program difficult.
float myrand(void)
{
    static int seed = 72191;
    char sq[22];

    if (g_bQAReadback) {
        seed *= seed;
        sprintf(sq, "%010d", seed);
        // pull the middle 5 digits out of sq
        sq[8] = 0;
        seed = atoi(&sq[3]);

        return seed/99999.f;
    } else {
        return rand()/(float)RAND_MAX;
    }
}

void initParticles(cData *p, int dx, int dy) {
    int i, j;
    for (i = 0; i < dy; i++) {
        for (j = 0; j < dx; j++) {
            p[i*dx+j].x = (j+0.5f+(myrand() - 0.5f))/dx;
            p[i*dx+j].y = (i+0.5f+(myrand() - 0.5f))/dy;
        }
    }
}

void keyboard( unsigned char key, int x, int y) {
    switch( key) {
    case 27:
        exit (0);
        break;
    case 'r':
        memset(hvfield, 0, sizeof(cData) * DS);
        cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS,
                   cudaMemcpyHostToDevice);

        initParticles(particles, DIM, DIM);

        cudaGraphicsUnregisterResource(cuda_vbo_resource);

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
        glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cData) * DS,
                        particles, GL_DYNAMIC_DRAW_ARB);
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

        cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);

        break;
    default: break;
    }
}

void click(int button, int updown, int x, int y) {
    lastx = x; lasty = y;
    clicked = !clicked;
}

void motion (int x, int y) {
    // Convert motion coordinates to domain
    float fx = (lastx / (float)wWidth);
    float fy = (lasty / (float)wHeight);
    int nx = (int)(fx * DIM);
    int ny = (int)(fy * DIM);

    if (clicked && nx < DIM-FR && nx > FR-1 && ny < DIM-FR && ny > FR-1) {
        int ddx = x - lastx;
        int ddy = y - lasty;
        fx = ddx / (float)wWidth;
        fy = ddy / (float)wHeight;
        int spy = ny-FR;
        int spx = nx-FR;
        addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
        lastx = x; lasty = y;
    }
    glutPostRedisplay();
}

void reshape(int x, int y) {
    wWidth = x; wHeight = y;
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 1, 0, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}

void cleanup(void) {
    cudaGraphicsUnregisterResource(cuda_vbo_resource);

    unbindTexture();
    deleteTexture();

    // Free all host and device resources
    free(hvfield); free(particles);
    cudaFree(dvfield);
    cudaFree(vxfield); cudaFree(vyfield);
    cufftDestroy(planr2c);
    cufftDestroy(planc2r);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    glDeleteBuffersARB(1, &vbo);
}

int initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("Compute Stable Fluids");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(click);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);


    return true;
}

#include "helper.h"

int main(int argc, char** argv) 
{
    int devID;
    cudaDeviceProp deviceProps;
    printf("[%s] - [OpenGL/CUDA simulation] starting...\n", sSDKname);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    initGL(&argc, argv);

    devID = cutGetMaxGflopsDeviceId();
    cudaGLSetGLDevice(devID);

    // get number of SMs on this GPU
    cudaGetDeviceProperties(&deviceProps, devID);
    printf("CUDA device [%s] has %d Multi-Processors\n",
           deviceProps.name, deviceProps.multiProcessorCount);

    // Allocate and initialize host data
    GLint bsize;
    
    hvfield = (cData*)malloc(sizeof(cData) * DS);
    memset(hvfield, 0, sizeof(cData) * DS);

    // Allocate and initialize device data
    cudaMallocPitch((void**)&dvfield, &tPitch, sizeof(cData)*DIM, DIM);
    
    cudaMemcpy(dvfield, hvfield, sizeof(cData) * DS,
               cudaMemcpyHostToDevice);
    // Temporary complex velocity field data
    cudaMalloc((void**)&vxfield, sizeof(cData) * PDS);
    cudaMalloc((void**)&vyfield, sizeof(cData) * PDS);
    
    setupTexture(DIM, DIM);
    bindTexture();
    
    // Create particle array
    particles = (cData*)malloc(sizeof(cData) * DS);
    memset(particles, 0, sizeof(cData) * DS);
    
    initParticles(particles, DIM, DIM);

    // Create CUFFT transform plan configuration
    cufftPlan2d(&planr2c, DIM, DIM, CUFFT_R2C);
    cufftPlan2d(&planc2r, DIM, DIM, CUFFT_C2R);
    // TODO: update kernels to use the new unpadded memory layout for perf
    // rather than the old FFTW-compatible layout
    cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_FFTW_PADDING);
    cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_FFTW_PADDING);

    glGenBuffersARB(1, &vbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(cData) * DS,
                    particles, GL_DYNAMIC_DRAW_ARB);

    glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bsize);
    if (bsize != (sizeof(cData) * DS))
        goto EXTERR;
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);

    atexit(cleanup);
    glutMainLoop();

    return 0;

EXTERR:
    printf("Failed to initialize GL extensions.\n");
}
