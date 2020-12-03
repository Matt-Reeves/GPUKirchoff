#include <stdio.h>

__global__ void initializePositionsKernel(double2* r, double2* rnew, int N){

int i = threadIdx.x + blockIdx.x*blockDim.x;

if (i<N){
  rnew[i].x = r[i].x;
  rnew[i].y = r[i].y;
}
//if( i >= N){
//  rnew[i].x = 0.0;
//  rnew[i].y = 0.0;
//  r[i].x = 0.0;
//  r[i].y = 0.0;
//}

}
