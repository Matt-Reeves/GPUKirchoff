#ifndef HAS_DEVICE_FUNCTIONS_H
#define HAS_DEVICE_FUNCTIONS_H

#include "getCUDA.h"

__device__ inline
double __shfl_down(double var, unsigned int srcLane, int width=32) {
  int2 a = *reinterpret_cast<int2*>(&var);
  a.x = __shfl_down(a.x, srcLane, width);
  a.y = __shfl_down(a.y, srcLane, width);
  return *reinterpret_cast<double*>(&a);
}

__inline__ __device__ double warpReduceSum(double a)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
  {
     a += __shfl_down(a, offset);
  }
  return a;
}

__inline__ __device__ void blockReduceSum(double a, double* smem_addr)
{ 
  int warpId;
  // First run a warp reduction sum
  a = warpReduceSum(a);

  // Now use shared memory to accumulate the warp contributions, with guaranteed summation order
  if (threadIdx.x % warpSize == 0)
  {
    // Write result for each warp to a shared memory array
    warpId = threadIdx.x / warpSize;
    smem_addr[warpId] = a;
  }
  __syncthreads();
  // Zeroth thread does final sum
  if (threadIdx.x == 0)
  {
    for (int i = 1; i < blockDim.x/warpSize; i++)
    {
      a += smem_addr[i];
    }
    smem_addr[0] = a;
  }

__syncthreads();
}

#endif

