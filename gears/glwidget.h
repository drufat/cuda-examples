#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>

extern "C" {
#include "gears.h"
}

class GLWidget : public QGLWidget
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();

    int xRotation() const { return g.xRot; }
    int yRotation() const { return g.yRot; }
    int zRotation() const { return g.zRot; }

public slots:
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);

signals:
    void xRotationChanged(int angle);
    void yRotationChanged(int angle);
    void zRotationChanged(int angle);

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);

private:
    Gears g;
    QPoint lastPos;
};

#endif // GLWIDGET_H
