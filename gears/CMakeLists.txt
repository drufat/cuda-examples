find_package(OpenGL REQUIRED)

find_package(GLUT REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
add_executable(GLUTgears GLUTgears.c)
target_link_libraries(GLUTgears ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} m)

find_package(SDL REQUIRED)
include_directories(${SDL_INCLUDE_DIR})
add_executable(SDLgears SDLgears.c)
target_link_libraries(SDLgears ${SDL_LIBRARY} ${OPENGL_LIBRARIES} m)

