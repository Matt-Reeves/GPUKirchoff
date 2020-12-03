#include "getCUDA.h"
#include <stdio.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <math.h>

ptrdiff_t RemoveVortices( int N, int Np, double2* r, int* remove, ptrdiff_t* Pnew, ptrdiff_t* Mnew){

  ptrdiff_t diff;
  thrust::device_ptr <double> start = thrust::device_pointer_cast (&(r[0].x)); 
  thrust::device_ptr <double> new_end = thrust::remove(start, start + 2*N, 1.0e6 );
  diff  = new_end - start;
  return diff; 
}

//void RemoveVortices( int N, int Np, double2* r, int* remove, ptrdiff_t* Pnew, ptrdiff_t* Mnew){
//
//
//  thrust::device_ptr <double> start2 = thrust::device_pointer_cast (&(r[Np].x)); 
//  thrust::device_ptr <double> new_end2 = thrust::remove(start2, start2 + 2*(N-Np), 1.0e6 );
//  *Mnew = new_end2 - start2;
//
//  thrust::device_ptr <double> start = thrust::device_pointer_cast (&(r[0].x)); 
//  thrust::device_ptr <double> new_end = thrust::remove(start, start + 2*Np, 1.0e6 );
//  *Pnew = new_end - start;
//
//}


