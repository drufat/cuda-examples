CUDA Examples
======================

Some CUDA, CUFFT and OpenGL examples.

Prerequisites
--------------

First install the prerequisites

On Ubuntu

::

    sudo apt-get install cmake nvidia-cuda-toolkit freeglut3-dev libxmu-dev libxi-dev libsdl1.2-dev

On Arch Linux

::

    sudo pacman -S cmake cuda freeglut glu sdl2

Build
------

::

    mkdir build
    cd build
    cmake ..
    make
    

Run
-------

::

    fluidsGL/fluidsGL
