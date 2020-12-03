#include "getCUDA.h"
#include <math.h>

__global__ void getError(int N, double2* r4, double2* r5){
  //double2* r4 = &(arr[N]);
  //double2* r5 = &(arr[4*N]);
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i<N){
//    r4[i].x = fabs(r5[i].x -r4[i].x)/fabs(r5[i].x);
//    r4[i].y = fabs(r5[i].y -r4[i].y)/fabs(r5[i].y);
    r4[i].x = fabs(r5[i].x -r4[i].x);
    r4[i].y = fabs(r5[i].y -r4[i].y);

  }
}
