#include "glwidget.h"

#include <QTimer>
#include <QMouseEvent>

#include <math.h>

GLWidget::GLWidget(QWidget *parent)
    : QGLWidget(parent)
{
    g = {0, 0, 0, 0, 0, 0, 0};

    QTimer *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, [&](){
        gears_advance(&g);
        updateGL();
    });
    timer->start(20);
}

GLWidget::~GLWidget()
{
    makeCurrent();
    glDeleteLists(g.gear1, 1);
    glDeleteLists(g.gear2, 1);
    glDeleteLists(g.gear3, 1);
}

void GLWidget::setXRotation(int angle)
{
    gears_normalize_angle(&angle);
    if (angle != g.xRot) {
        g.xRot = angle;
        emit xRotationChanged(angle);
        updateGL();
    }
}

void GLWidget::setYRotation(int angle)
{
    gears_normalize_angle(&angle);
    if (angle != g.yRot) {
        g.yRot = angle;
        emit yRotationChanged(angle);
        updateGL();
    }
}

void GLWidget::setZRotation(int angle)
{
    gears_normalize_angle(&angle);
    if (angle != g.zRot) {
        g.zRot = angle;
        emit zRotationChanged(angle);
        updateGL();
    }
}

void GLWidget::initializeGL()
{
    gears_initialize(&g);
}

void GLWidget::paintGL()
{
    gears_paint(&g);
}

void GLWidget::resizeGL(int width, int height)
{
    gears_resize(width, height);
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
        setXRotation(g.xRot + 8 * dy);
        setYRotation(g.yRot + 8 * dx);
    } else if (event->buttons() & Qt::RightButton) {
        setXRotation(g.xRot + 8 * dy);
        setZRotation(g.zRot + 8 * dx);
    }
    lastPos = event->pos();
}
