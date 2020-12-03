#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "useful.h"
#include "getCUDA.h"
#include "ark45ck.h"


//#define K1 a[i]
//#define K2 a[i+N]
//#define K3 a[i+2*N]
//#define K4 a[i+3*N]
//#define K5 a[i+4*N]
//#define K6 a[i+5*N]

#define R2 1.0
#define R1 0.1

__device__ inline void Positives(int tid,int j, int N,double L, double2 ri, double2 rj, double2* v,double* rmin) {
//Note L.x = D, L.y = q =  PI/D where D is the box size
// if ((ri.x == rj.x) && (ri.y = rj.y) ) printf("!! WARNING !!: ri (%d) == rj (%d), NaNs will follow. Aborting...\n",tid,j);
  double XIJ, YIJ;
  double k; 
  double sx, sy, cx, cy;
  XIJ = (ri.x - rj.x);
  YIJ = (ri.y - rj.y);
  if (tid < N/2){
   double rij;
    XIJ += (XIJ < -0.5*L) ? L:0.0;
    XIJ -= (XIJ >  0.5*L) ? L:0.0;
    YIJ += (YIJ < -0.5*L) ? L:0.0;
    YIJ -= (YIJ >  0.5*L) ? L:0.0;
    rij = XIJ*XIJ + YIJ*YIJ;
    if (rij < *rmin) *rmin = rij;
  }
  XIJ *= 2.0*PI/L;
  YIJ *= 2.0*PI/L;
  sincos(XIJ, &sx, &cx);
  sincos(YIJ, &sy, &cy);
  #pragma unroll
  for(k=-3.0; k<3.1; k+=1.0){ 
    (*v).x += (-sy)/(cosh(XIJ-2.0*PI*k)-cy);
    (*v).y += ( sx)/(cosh(YIJ-2.0*PI*k)-cx);
  }
}


__device__ inline void Negatives(int tid,int j,int N, double L, double2 ri, double2 rj, double2* v,double* rmin) {
 //if ((ri.x == rj.x) && (ri.y = rj.y) ) printf("!! WARNING !!: ri (%d) == rj (%d), NaNs will follow. Aborting...\n",tid,j);
  double XIJ, YIJ;
  double k; 
  double sx, sy, cx, cy;
  XIJ = (ri.x - rj.x);
  YIJ = (ri.y - rj.y);
  if (tid >= N/2 && tid < N){
    double rij;
    XIJ += (XIJ < -0.5*L) ? L:0.0;
    XIJ -= (XIJ >  0.5*L) ? L:0.0;
    YIJ += (YIJ < -0.5*L) ? L:0.0;
    YIJ -= (YIJ >  0.5*L) ? L:0.0;
    rij = XIJ*XIJ + YIJ*YIJ;
    if (rij < *rmin) *rmin = rij;
  }
  XIJ *= 2.0*PI/L;
  YIJ *= 2.0*PI/L;
  sincos(XIJ, &sx, &cx);
  sincos(YIJ, &sy, &cy);
  #pragma unroll
  for(k=-3.0; k<3.1; k+=1.0){ 
    (*v).x -=(-sy)/(cosh(XIJ-2.0*PI*k)-cy);
    (*v).y -=( sx)/(cosh(YIJ-2.0*PI*k)-cx);
  }
}


__global__ void Get_drdt(int N,double L,double dt,double gamma,double2* a,double2* rnew ){
   
  //Note that input argument "a" is now a pointer to EXACTLY the point in the array you want to dump values into,
  // no longer the first element of a. This removed the need to write a[i+(stage-1)*N].x etc....
  int i;
  double2 v;
  double local_gamma, rmin;
  extern __shared__ double2 smem[];
  double2 ri; 

  v.x = 0.0; v.y = 0.0; rmin = 1e10;
  i = threadIdx.x + blockIdx.x*blockDim.x;  
  /*Notice all threads do work here, even if they no longer have a vortex
  associated with them.  This is necessary because we still want them to load
  values into shared memory for the threads that still have work to do.*/
  ri = rnew[i];
  int p = 0; int tile; int idx;
  for (idx = threadIdx.x, tile = 0; tile < (N/2+blockDim.x-1)/blockDim.x; idx+= blockDim.x,tile++ ){
    smem[threadIdx.x] = rnew[idx];
    __syncthreads();
    for (int j=0; j<blockDim.x; j++){
      if (p == N/2) break; 
      if (i==p){ p++; continue;}
      Positives(i,p,N,L,ri, smem[j], &v,&rmin);
      p++;
    }
  __syncthreads();
  }
   if (p!= N/2) printf("!! WARNING !!: thread %d, p = %d\n",i,p);
  for (idx = threadIdx.x + N/2,tile = 0; tile < (N/2+blockDim.x-1)/blockDim.x; idx+= blockDim.x,tile++ ){
    smem[threadIdx.x] = rnew[idx];
    __syncthreads();
    for (int j=0; j<blockDim.x; j++){
      if (p == N) break; 
      if (i==p){ p++; continue; }
      Negatives(i,p,N,L,ri, smem[j], &v,&rmin);
      p++;
    }
  __syncthreads();
  }
   if (p!= N) printf("!! WARNING !!: thread %d, p = %d\n",i,p);
  //__syncthreads(); //Again, extra threads need to wait...

//  /*Failed attempt at negative viscosity .... */  
//  if (gamma < 0 ){ 
//    local_gamma = -gamma;
//    if (i < N/2 ) local_gamma = -local_gamma;
//
//    if (rmin <= R2) local_gamma *= -10;
//  }
//  else{ 

  //change the sign of gamma depending on thread index to avoid using kappai
   local_gamma =  -max(exp(log(gamma)*( (sqrt(rmin)-R1)/(R2-R1) )),gamma);
   
  if (i < N/2) local_gamma = -local_gamma;
//  }
  //only update if the thread wasn't calculating nonsense
  if (i<N){
    a[i].x = PI/L*(v.x + local_gamma*v.y)*dt;
    a[i].y = PI/L*(v.y - local_gamma*v.x)*dt;  
  }         
}


__global__ void doRungeKuttaStep1(int N,double2* r, double2* rnew, double2* k1){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i<N){
    rnew[i].x = r[i].x + b21*k1[i].x; 
    rnew[i].y = r[i].y + b21*k1[i].y;
  }
} 
 
__global__ void doRungeKuttaStep2(int N,double2* r, double2* rnew, double2* k1, double2* k2){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i<N){
    rnew[i].x = r[i].x + b31*k1[i].x + b32*k2[i].x; 
    rnew[i].y = r[i].y + b31*k1[i].y + b32*k2[i].y;
  }
}  
__global__ void doRungeKuttaStep3(int N,double2* r, double2* rnew, double2* k1, double2* k2, double2* k3){
  int i = threadIdx.x + blockIdx.x*blockDim.x;

  if (i<N){
    rnew[i].x = r[i].x + b41*k1[i].x + b42*k2[i].x + b43*k3[i].x; 
    rnew[i].y = r[i].y + b41*k1[i].y + b42*k2[i].y + b43*k3[i].y;
  }
}
__global__ void doRungeKuttaStep4(int N,double2* r, double2* rnew, double2* k1, double2* k2, double2* k3, double2* k4){
  int i = threadIdx.x + blockIdx.x*blockDim.x;

  if (i<N){
    rnew[i].x = r[i].x + b51*k1[i].x + b52*k2[i].x + b53*k3[i].x + b54*k4[i].x; 
    rnew[i].y = r[i].y + b51*k1[i].y + b52*k2[i].y + b53*k3[i].y + b54*k4[i].y;
  }
}
__global__ void doRungeKuttaStep5(int N,double2* r, double2* rnew, double2* k1, double2* k2, double2* k3, double2* k4, double2* k5){
  int i = threadIdx.x + blockIdx.x*blockDim.x;

  if (i<N){
    rnew[i].x = r[i].x + b61*k1[i].x + b62*k2[i].x + b63*k3[i].x + b64*k4[i].x + b65*k5[i].x; 
    rnew[i].y = r[i].y + b61*k1[i].y + b62*k2[i].y + b63*k3[i].y + b64*k4[i].y + b65*k5[i].y;
  }
}
__global__ void doRungeKuttaStep6(int N,double2* r, double2* rnew, double2* k1, double2* k2, double2* k3, double2* k4, double2* k5, double2* k6){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i<N){
    //4th order...
    k2[i].x = r[i].x + c41*k1[i].x + c43*k3[i].x + c44*k4[i].x +c45*k5[i].x + c46*k6[i].x;
    k2[i].y = r[i].y + c41*k1[i].y + c43*k3[i].y + c44*k4[i].y +c45*k5[i].y + c46*k6[i].y;
    //5th order...
    k5[i].x = r[i].x + c51*k1[i].x + c53*k3[i].x + c54*k4[i].x + c56*k6[i].x;
    k5[i].y = r[i].y + c51*k1[i].y + c53*k3[i].y + c54*k4[i].y + c56*k6[i].y;
  }
}

//__global__ void doRungeKuttaStep1(int N,double2* r, double2* rnew, double2* a){
//  int i = threadIdx.x + blockIdx.x*blockDim.x;
//  if (i<N){
//    rnew[i].x = r[i].x + b21*K1.x; 
//    rnew[i].y = r[i].y + b21*K1.y;
//  }
//} 
// 
//__global__ void doRungeKuttaStep2(int N,double2* r, double2* rnew, double2* a){
//  int i = threadIdx.x + blockIdx.x*blockDim.x;
//  if (i<N){
//    rnew[i].x = r[i].x + b31*K1.x + b32*K2.x; 
//    rnew[i].y = r[i].y + b31*K1.y + b32*K2.y;
//  }
//}  
//__global__ void doRungeKuttaStep3(int N,double2* r, double2* rnew, double2* a){
//  int i = threadIdx.x + blockIdx.x*blockDim.x;
//  if (i<N){
//    rnew[i].x = r[i].x + b41*K1.x + b42*K2.x + b43*K3.x; 
//    rnew[i].y = r[i].y + b41*K1.y + b42*K2.y + b43*K3.y;
//  }
//}
//__global__ void doRungeKuttaStep4(int N,double2* r, double2* rnew, double2* a){
//  int i = threadIdx.x + blockIdx.x*blockDim.x;
//  if (i<N){
//    rnew[i].x = r[i].x + b51*K1.x + b52*K2.x + b53*K3.x + b54*K4.x; 
//    rnew[i].y = r[i].y + b51*K1.y + b52*K2.y + b53*K3.y + b54*K4.y;
//  }
//}
//__global__ void doRungeKuttaStep5(int N,double2* r, double2* rnew, double2* a){
//  int i = threadIdx.x + blockIdx.x*blockDim.x;
//  if (i<N){
//    rnew[i].x = r[i].x + b61*K1.x + b62*K2.x + b63*K3.x + b64*K4.x + b65*K5.x; 
//    rnew[i].y = r[i].y + b61*K1.y + b62*K2.y + b63*K3.y + b64*K4.y + b65*K5.y;
//  }
//}
//__global__ void doRungeKuttaStep6(int N,double2* r, double2* rnew, double2* a){
//  int i = threadIdx.x + blockIdx.x*blockDim.x;
//  if (i<N){
//    //4th order...
//    K2.x = r[i].x + c41*K1.x + c43*K3.x + c44*K4.x +c45*K5.x + c46*K6.x;
//    K2.y = r[i].y + c41*K1.y + c43*K3.y + c44*K4.y +c45*K5.y + c46*K6.y;
//    //5th order...
//    K5.x = r[i].x + c51*K1.x + c53*K3.x + c54*K4.x + c56*K6.x;
//    K5.y = r[i].y + c51*K1.y + c53*K3.y + c54*K4.y + c56*K6.y;
//  }
//}

