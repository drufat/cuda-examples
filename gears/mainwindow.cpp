#include "glwidget.h"
#include "mainwindow.h"

#include <QtWidgets>


MainWindow::MainWindow()
{

    auto glWidget = new GLWidget;

    auto glWidgetArea = new QScrollArea;
    glWidgetArea->setWidget(glWidget);
    glWidgetArea->setWidgetResizable(true);
    glWidgetArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    glWidgetArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    glWidgetArea->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    glWidgetArea->setMinimumSize(50, 50);

    auto createSlider = [&](void (GLWidget::*changedSignal)(int),
                            void (GLWidget::*setterSlot)(int)) -> QSlider* {
        QSlider *slider = new QSlider(Qt::Horizontal);
        slider->setRange(0, 360 * 16);
        slider->setSingleStep(16);
        slider->setPageStep(15 * 16);
        slider->setTickInterval(15 * 16);
        slider->setTickPosition(QSlider::TicksRight);
        connect(slider, &QSlider::valueChanged, glWidget, setterSlot);
        connect(glWidget, changedSignal, slider, &QSlider::setValue);

        return slider;
    };

    auto xSlider = createSlider(&GLWidget::xRotationChanged, &GLWidget::setXRotation);
    auto ySlider = createSlider(&GLWidget::yRotationChanged, &GLWidget::setYRotation);
    auto zSlider = createSlider(&GLWidget::zRotationChanged, &GLWidget::setZRotation);

    //Create actions and menus
    auto exitAct = new QAction(tr("E&xit"), this);
    exitAct->setShortcuts(QKeySequence::Quit);
    connect(exitAct, &QAction::triggered, this, &MainWindow::close);

    auto aboutQtAct = new QAction(tr("About &Qt"), this);
    connect(aboutQtAct, &QAction::triggered, qApp, &QApplication::aboutQt);

    auto fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addSeparator();
    fileMenu->addAction(exitAct);

    auto helpMenu = menuBar()->addMenu(tr("&Help"));
    helpMenu->addAction(aboutQtAct);

    auto centralLayout = new QVBoxLayout;
    centralLayout->addWidget(glWidgetArea);
    centralLayout->addWidget(xSlider);
    centralLayout->addWidget(ySlider);
    centralLayout->addWidget(zSlider);

    auto centralWidget = new QWidget;
    setCentralWidget(centralWidget);
    centralWidget->setLayout(centralLayout);

    xSlider->setValue(15 * 16);
    ySlider->setValue(345 * 16);
    zSlider->setValue(0 * 16);

    setWindowTitle(tr("Qt Gears"));
    resize(400, 300);
}
