#include <stdio.h>
#include <math.h>
__global__ void checkPositions(double2* rnew,int N, double L){

int tid = threadIdx.x + blockIdx.x*blockDim.x;

if (tid < N){
  if (fabs(rnew[tid].x) > L/2.0) printf("Thread %d: r.x = %lf\n",tid,rnew[tid].x);
  if (fabs(rnew[tid].y) > L/2.0) printf("Thread %d: r.y = %lf\n",tid,rnew[tid].y);
}

}



