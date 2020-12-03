#include "getCUDA.h"
#include <iostream>
#include <stdio.h>
#include "checkDevices.h"

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

void checkDevices(int cardNum)
{
    std::cout << "Checking devices ...\n";
    int numDevices;
    cudasafe( cudaGetDeviceCount(&numDevices), "cudaGetDeviceCount" );
    std::cout << "Number of devices: " << numDevices << std::endl;
    std::cout << "Setting devices ..." << std::endl;
    //cudasafe( cudaSetValidDevices(NULL, 0), "cudaSetValidDevices");
    cudasafe( cudaSetDevice(cardNum), "cudaSetDevice");
    cudasafe(cudaFree(0), "cudaFree");
    int device;
    cudasafe( cudaGetDevice(&device), "cudaGetDevice");
    std::cout << "Selected device: " << device << std::endl;

    size_t free_byte ;
    size_t total_byte ;
    cudasafe( cudaMemGetInfo( &free_byte, &total_byte ), "cudaMemGetInfo" ) ;
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    std::cout << "GPU memory usage: used = " << used_db/1024.0/1024.0 << " MB, free = " << free_db/1024.0/1024.0 << " MB, total = " << total_db/1024.0/1024.0 << " MB" << std::endl;

    cudaDeviceProp devProp;
    cudasafe( cudaGetDeviceProperties(&devProp, device), "cudaGetDeviceProperties" );
    printDevProp(devProp);

}
 
void cudasafe (cudaError_t error, const char* message)
{
    if(error!=cudaSuccess)
    {
        std::cout << "ERROR: " << message << ", " << error << std::endl;
        exit(-1);
    }
}


