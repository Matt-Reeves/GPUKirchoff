#ifndef HAS_EVOLVE

#define HAS_EVOLVE
__global__ void  UpdatePositions (int N,double L, double2* r, double2* r4);
__global__ void  RemovalKernel (int N,double L,  double2* r, int * remove );
__global__ void  Get_drdt(int N, int Np,  double L, double dt, double gamma, double2* a, double2* rnew);
__global__ void getError(int N, double2* r4, double2* r5);
__global__ void initializePositionsKernel( double2* r, double2* rnew, int N); 
__global__ void doRungeKuttaStep1(int N,double2* r, double2* rnew, double2* k1);
__global__ void doRungeKuttaStep2(int N,double2* r, double2* rnew, double2* k1, double2* k2);
__global__ void doRungeKuttaStep3(int N,double2* r, double2* rnew, double2* k1, double2* k2, double2* k3);
__global__ void doRungeKuttaStep4(int N,double2* r, double2* rnew, double2* k1, double2* k2, double2* k3, double2* k4);
__global__ void doRungeKuttaStep5(int N,double2* r, double2* rnew, double2* k1, double2* k2, double2* k3, double2* k4, double2* k5);
__global__ void doRungeKuttaStep6(int N,double2* r, double2* rnew, double2* k1, double2* k2, double2* k3, double2* k4, double2* k5, double2* k6);
__global__ void checkPositions(double2* rnew, int N, double L); 
__global__ void checkDipoles( int N, int Np, double L, double2* r, int* remove); 
__global__ void findPairs ( int N, int Np, double2* r, int* remove);
__global__ void boundaryLoss (int N, double L, double2* r, int* remove);
ptrdiff_t RemoveVortices(int N,int Np, double2* r, int* remove, ptrdiff_t* Pnew, ptrdiff_t* Mnew);


#endif
