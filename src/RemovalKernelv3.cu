#include "getCUDA.h"
#include <stdio.h>
#include <math.h>

__global__ void boundaryLoss(int N,double L, double2* r, int* remove){
  int i;
  i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < N){
//    if (r[i].x*r[i].x + r[i].y*r[i].y > L*L){
//	printf("Warning: Vortex %d was outside domain\n",i);
//	r[i].x = 1e6; 
//	r[i].y = 1e6;
//    }
      if (sqrt( r[i].x*r[i].x + r[i].y*r[i].y) > L-10.0){
        r[i].x = 1e6;
        r[i].y = 1e6;
        printf("Vortex %d lost at boundary: (r = %1.2f)\n",i,sqrt(r[i].x*r[i].x + r[i].y*r[i].y));
      }
    
  }
}
__global__ void checkDipoles( int N, int Np, double L, double2*r, int* remove){

  int i,j;
  double rij2,dx,dy;
  double imin = 1e10;
  //Positive
  i = threadIdx.x + blockIdx.x*blockDim.x;

  if (i<Np){
    remove[i] = -1;
    for (j=Np; j < N; j++){
      dx = r[i].x - r[j].x;
      dy = r[i].y - r[j].y;
      rij2 =  dx*dx + dy*dy;
      //printf("%g\n",rij2);
      if ( rij2 <= 1.0 && rij2 < imin){ 
        remove[i] = j;
        imin = rij2;
      }
    }
  }
  else if ( (i >= Np) && (i < N) ){ //Negative
    remove[i] = -1;
    for (j=0; j < Np; j++){
      dx = r[i].x - r[j].x;
      dy = r[i].y - r[j].y;
      rij2 =  dx*dx + dy*dy;
      if ( rij2 <= 1.0 && rij2 < imin){ 
        remove[i] = j;
        imin = rij2;
      }
    }
  }
  
}

__global__ void findPairs (int N, int Np, double2*r, int* remove){

  int i; 
  i = threadIdx.x + blockIdx.x*blockDim.x; 
  int check; 
  if (i<Np){
    if (remove[i] != -1){
      check = remove[i];
      if (remove[check] == i){ 
        printf("Vortex %d and %d form a pair...\n",i,check);
        r[i].x = 1e6; r[i].y = 1e6;
        r[check].x = 1e6; r[check].y = 1e6;
        }   
    }   
  }

}

//__global__ void RemovalKernel (int nd, double L,  double2* r, int* remove ){
//  // nd is number of possible dipoles (namely, N/2)...
//  //Look for mutual nearest neighbours, return index of nearest
//  //opposite-sign neighbour..
//
//  checkPositives(nd,L,r,remove);
//  __syncthreads();
//  checkNegatives(nd,L,r,remove);
//  __syncthreads();
//  findPairs(nd,r,remove);
//
//}

