#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void print(char *a,int N)
{
    char p[12]="Hello CUDA\n";
    int idx=blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<N)
    {
        a[idx]=p[idx];
    }
}

int main(void)
{
    char *a_h,*a_d;
    const int N=12;
    size_t size=N*sizeof(char);
    a_h=(char *)malloc(size);
    cudaMalloc((void **)&a_d,size);
    for(int i=0;i<N;i++)
    {
        a_h[i]=0;
    }
    cudaMemcpy(a_d,a_h,size,cudaMemcpyHostToDevice);
    int blocksize=4;
    int nblock=N/blocksize+(N%blocksize==0?0:1);
    print<<<nblock,blocksize>>>(a_d,N)
                              ;
    cudaMemcpy(a_h,a_d,sizeof(char)*N,cudaMemcpyDeviceToHost)
            ;
    for(int i=0;i<N;i++)
    {
        printf("%c",a_h[i]);
    }
    free(a_h);
    cudaFree(a_d);

}
