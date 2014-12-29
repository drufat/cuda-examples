#include "glfluids.h"

#include <QTimer>
#include <QKeyEvent>
#include <QMouseEvent>

// Texture pitch
size_t tPitch = 0;

void initParticles(float2 *p, int dx, int dy)
{
    auto myrand = []() -> float {
            return qrand()/(float)RAND_MAX;};

    for (int i = 0; i < dy; i++)
    {
        for (int j = 0; j < dx; j++)
        {
            p[i*dx+j].x = (j+0.5f+(myrand() - 0.5f))/dx;
            p[i*dx+j].y = (i+0.5f+(myrand() - 0.5f))/dy;
        }
    }
}

void GLFluids::simulateFluids(void)
{
    // simulate fluid
    advectVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM, DT);
    diffuseProject(vxfield, vyfield, CPADW, DIM, DT, VIS, planr2c, planc2r);
    updateVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM);
    advectParticles(cuda_vbo_resource, dvfield, DIM, DIM, DT);
}

GLFluids::GLFluids(QWidget *parent)
    : QGLWidget(parent),
      QGLFunctions()
{
    vbo = 0;

    wWidth = qMax(512, DIM);
    wHeight = qMax(512, DIM);

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

    QTimer *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, [&](){
        simulateFluids();
        updateGL();
    });
    timer->start(0);
}

GLFluids::~GLFluids(){

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

}

void GLFluids::reset()
{
    memset(hvfield, 0, sizeof(float2) * DS);
    cudaMemcpy(dvfield, hvfield, sizeof(float2) * DS,
               cudaMemcpyHostToDevice);

    initParticles(particles, DIM, DIM);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * DS,
                 particles, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);

}

void GLFluids::initializeGL()
{
    initializeGLFunctions();
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * DS,
                 particles, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);

}

void GLFluids::paintGL()
{
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
}

void GLFluids::resizeGL(int x, int y)
{
    wWidth = x;
    wHeight = y;
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 1, 0, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void GLFluids::mousePressEvent(QMouseEvent *event)
{
    auto lastPos = event->pos();
    lastx = lastPos.x();
    lasty = lastPos.y();
}

void GLFluids::mouseMoveEvent(QMouseEvent *event)
{
    int x = event->x(); int y = event->y();

    // Convert motion coordinates to domain
    float fx = (lastx / (float)wWidth);
    float fy = (lasty / (float)wHeight);
    int nx = (int)(fx * DIM);
    int ny = (int)(fy * DIM);

    if (event->buttons() & Qt::LeftButton) {
        int ddx = x - lastx;
        int ddy = y - lasty;
        fx = ddx / (float)wWidth;
        fy = ddy / (float)wHeight;
        int spy = ny-FR;
        int spx = nx-FR;
        addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);
    }

    lastx = x;
    lasty = y;
}
