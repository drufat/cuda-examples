#include <SDL.h>
#include <SDL_opengl.h>

#include "gears.h"

int main(int argc, char* argv[]){

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow(
                "SDL2 Gears",
                SDL_WINDOWPOS_UNDEFINED,
                SDL_WINDOWPOS_UNDEFINED,
                640, 480,
                SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE
                );

    SDL_GLContext glcontext = SDL_GL_CreateContext(window);

    Gears g = {0, 0, 0, 0, 0, 0, 0};

    gears_initialize(&g);
    int width; int height;
    SDL_GetWindowSize(window, &width, &height);
    gears_resize(width, height);

    int done = 0;
    while(!done) {
        SDL_Event e;
        while(SDL_PollEvent(&e)) {
            switch(e.type) {
                case SDL_KEYDOWN:
                    done = 1;
                    break;
                case SDL_QUIT:
                    done = 1;
                    break;
                default:
                    break;
            }
        }
        gears_paint(&g);
        SDL_GL_SwapWindow(window);
        SDL_Delay(10);

        gears_advance(&g);
    }


    SDL_GL_DeleteContext(glcontext);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
