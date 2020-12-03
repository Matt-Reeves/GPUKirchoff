#ifndef HAS_CHECK_DEVICES_H
#define HAS_CHECK_DEVICES_H
#include "getCUDA.h"
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void checkDevices(int cardNum);
void cudasafe (cudaError_t error, const char* message);
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif

