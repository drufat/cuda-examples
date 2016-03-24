// OpenGL Graphics includes
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

const char* vertex_shader = R"(

    attribute float x;
    attribute float y;

    void main() {
        gl_Position = vec4(x, y, 0.0, 1.0);
        gl_PointSize = 1.0;
    }

)";

const char* fragment_shader = R"(

    void main() {
        gl_FragClor = (1.0, 0.0, 0.0, 1.0);
    }

)";


float myrand(void)
{
    return rand()/(float)RAND_MAX;
}

void initParticles(float* x, float* y, int dx, int dy)
{
    int i, j;

    for (i = 0; i < dy; i++)
    {
        for (j = 0; j < dx; j++)
        {
            x[i*dx+j] = (j+0.5f+(myrand() - 0.5f))/dx;
            y[i*dx+j] = (i+0.5f+(myrand() - 0.5f))/dy;
        }
    }
}

void init(void) {

}

int main(void){

    int width = 512;
    int height = 512;

    if (!glfwInit()) exit(EXIT_FAILURE);

    auto w = glfwCreateWindow(512, 512, "Compute Stable Fluids", NULL, NULL);
    if (!w) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    if (!glewInit()) exit(EXIT_FAILURE);

    glfwMakeContextCurrent(w);
    glfwSwapInterval(1);

    while (!glfwWindowShouldClose(w)){
        //display();
        glfwSwapBuffers(w);
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;

}
