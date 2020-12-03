#include <stdio.h>
__global__ void UpdatePositions( int N,double L, double2* r, double2* r5){

int i= threadIdx.x + blockIdx.x*blockDim.x;

  if (i<N){
    r[i].x = r5[i].x;
    r[i].y = r5[i].y;
    if (r[i].x*r[i].x + r[i].y*r[i].y > L*L) printf("%d is outside region. %1.4f %1.4f\n",i,r[i].x,r[i].y);
  }


}


