#ifndef __STABLEFLUIDS_KERNELS_H_
#define __STABLEFLUIDS_KERNELS_H_

#include <cuda.h>
#include <cufft.h>

// Texture pitch
extern size_t tPitch;

void setup_texture(int x, int y);
void bind_texture(void);
void unbind_texture(void);
void delete_texture(void);
void update_texture(float2 *data, size_t w, size_t h, size_t pitch);

typedef void (*UpdateTextureCallback)(float2*, size_t, size_t, size_t);

void addForces(float2 *v, int dx, int dy, int spx, int spy, float fx, float fy, int r);
void advectVelocity(float2 *v, float *vx, float *vy, int dx, int pdx, int dy, float dt);
void diffuseProject(float2 *vx, float2 *vy, int dx, int dy, float dt, float visc, cufftHandle planr2c, cufftHandle planc2r);
void updateVelocity(float2 *v, float *vx, float *vy, int dx, int pdx, int dy);
void advectParticles(struct cudaGraphicsResource *cuda_vbo_resource, float2 *v, int dx, int dy, float dt);

#endif

