#include "mainwindow.h"
#include "glfluids.h"

#include <QtWidgets>


MainWindow::MainWindow()
{

    auto glFluids = new GLFluids();

    auto glWidgetArea = new QScrollArea;
    glWidgetArea->setWidget(glFluids);
    glWidgetArea->setWidgetResizable(true);
    glWidgetArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    glWidgetArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    glWidgetArea->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    glWidgetArea->setMinimumSize(512, 512);
    setCentralWidget(glWidgetArea);

    auto fileMenu = new QMenu(tr("File"), this);
    menuBar()->addMenu(fileMenu);

    auto quitAction = fileMenu->addAction(tr("E&xit"));
    quitAction->setShortcuts(QKeySequence::Quit);
    connect(quitAction, &QAction::triggered, this, &QApplication::quit);

    auto resetAction = fileMenu->addAction(tr("&Reset"));
    resetAction->setShortcut(Qt::Key_R);
    connect(resetAction, &QAction::triggered, glFluids, &GLFluids::reset);

//    glFluids->setFocusPolicy(Qt::StrongFocus);
//    glFluids->setFocus();

    setWindowTitle(tr("Qt Fluids"));
    resize(512, 512);
}
