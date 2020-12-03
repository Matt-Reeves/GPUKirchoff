#include <stdio.h>
#include <stdlib.h>
#include "getCUDA.h" //cuda runtime 
//#include "devicefunctions.cuh" //Checking errors, card selection
#include "checkDevices.h"
#include <thrust/extrema.h>  //Error checking for ark45
#include <thrust/device_ptr.h>
#include <omp.h>
#include "useful.h"    
#include "ark45ck.h" // Contains aj and bij coefficients
#include "wmcw_energy.h" //Weiss-Mcwilliams energy to check forcing
#include "evolve.cuh"

__constant__ double3 kf[2048]; // Be careful here, this needs to be big enough

int main(int argc, char* argv[]){

  if (argc < 8){printf("Usage is: ./prog.out cardNum N L tf loadfile threadsPerBlock Np \n");return -1;}

  //Select a card
  int cardNum = atoi(argv[1]);  printf("Using card # %d\n",cardNum);
  checkDevices(cardNum);
  cudaDeviceSynchronize();
  //Generic  parameters ...
  int N = atoi(argv[2]);
  int Np = atoi(argv[7]);
  double2 L;
  L.x = atof(argv[3]);
  L.y = PI/L.x;
  int threadNum = atoi(argv[6]);
  int blockNum =(N+threadNum-1)/threadNum;
  printf("Running %d blocks of %d threads, %d total threads\n",blockNum,threadNum,blockNum*threadNum); 
// if (threadNum*blockNum > N){
//    printf("!! ERROR !! : Launching program with more threads than initial number of vortices is forbidden\n");
//   return -1;
//  }
  if (threadNum > N/2){
    threadNum /= 2;
    blockNum = (N+threadNum-1)/threadNum;
    printf("!! Warning !!: Number of threads per block cannot exceed  N/2: continuing with %d blocks of %d threads\n",blockNum,threadNum);
  }
  //Allocate memory 
  double2 *r, *k1,*k2,*k3,*k4,*k5,*k6, *rnew;
  int *remove;
  int * remove_host = (int*) malloc(N*sizeof(int));
  gpuErrchk( cudaMalloc( (void**) &r, sizeof(double2)*N ) );
  gpuErrchk( cudaMalloc( (void**) &k1, sizeof(double2)*N) );
  gpuErrchk( cudaMalloc( (void**) &k2, sizeof(double2)*N) );
  gpuErrchk( cudaMalloc( (void**) &k3, sizeof(double2)*N) );
  gpuErrchk( cudaMalloc( (void**) &k4, sizeof(double2)*N) );
  gpuErrchk( cudaMalloc( (void**) &k5, sizeof(double2)*N) );
  gpuErrchk( cudaMalloc( (void**) &k6, sizeof(double2)*N) );
  gpuErrchk( cudaMalloc( (void**) &remove, sizeof(int)*N ) ) ;
  gpuErrchk( cudaMalloc( (void**) &rnew, sizeof(double2)*N ) );
  double2* r_host = (double2*) malloc(N*sizeof(double2));


/******************** Load Initial Condition ******************/
 int MAX_FILE_COLUMNS = N;
 char* loadfile = argv[5];
 FILE *file = fopen(loadfile, "r");
 int temp;
 for (temp = 0; temp < MAX_FILE_COLUMNS; temp++){
  if (feof(file)) break;
  fscanf(file, "%lf %lf ", &(r_host[temp].x), &(r_host[temp].y));
  if (r_host[temp].x*r_host[temp].x + r_host[temp].y*r_host[temp].y > L.x*L.x){
    printf("Error: vortex in initial condition is outside the domain!\n");
   // return -1;
  }
 }
 fclose(file);
 if (temp != N){
  printf("temp = %d\n",temp); 
  printf("Error!: number of vortices in input file is not the same as N\n"); 
   return -1;
 }
  
 gpuErrchk( cudaMemcpy(r,r_host,N*sizeof(double2),cudaMemcpyHostToDevice) );
 
/* ********** Evolve... this REALLY should  be a .c function **************/
  double gamma = 0.05;
  double tol = 1e-6;  
  double t = 0.0;
  double dt = 1e-3;
  double DT= 1000;
  double tf = atof(argv[4]);
  int Nt = tf/DT;
  int filenum = 1; 
  

  double current_energy;
  int * kappa = (int*) malloc(N*sizeof(double));
  for (int z=0;z<N/2;z++){
    kappa[z] = 1;
    kappa[z+N/2] = -1;
  }
  double tStart, tEnd;
  double t1, t2;
  
    FILE *eFile;
    char efilename[64];
    sprintf(efilename,"energy.txt");
    eFile = fopen(efilename,"w");
  //double t3, t4;
  double pass, fail;
  double fail_rate;
  pass = 1; fail=0;
  tStart = omp_get_wtime();
  for (int ii=1;ii<=Nt;ii++){
    t1 = omp_get_wtime();
    while (t < ii*DT ){  
      
      if (t + dt > ii*DT)  dt = ii*DT - t;
       
      int success = 0;
      //Evolve with step dt
      while( success == 0){    
//       //Get k1,...,k6
       //printf("Line 119: N = %d Np = %d\n",N,Np);
        initializePositionsKernel <<< blockNum, threadNum >>> (r,rnew,N); //Sets rnew = r (rnew is positions at trial steps)
        Get_drdt <<< blockNum, threadNum, threadNum*sizeof(double2)  >>> (N,Np,L.x,dt,gamma,k1,rnew); //Gets k1*dt
        gpuErrchk( cudaPeekAtLastError() );
        doRungeKuttaStep1<<< blockNum, threadNum >>>(N,r,rnew,k1);
        gpuErrchk( cudaPeekAtLastError() );
/*******************************************************************************************************/
        Get_drdt <<< blockNum, threadNum, threadNum*sizeof(double2)  >>> (N,Np,L.x,dt,gamma,k2,rnew); //Gets k2*dt from rnew = r + b21*k1        
        gpuErrchk( cudaPeekAtLastError() );
        doRungeKuttaStep2<<< blockNum, threadNum >>>(N,r,rnew,k1,k2);
       gpuErrchk( cudaPeekAtLastError() );
/*******************************************************************************************************/
        Get_drdt <<< blockNum, threadNum, threadNum*sizeof(double2)  >>> (N,Np,L.x,dt,gamma,k3,rnew);        
        gpuErrchk( cudaPeekAtLastError() );
        doRungeKuttaStep3<<< blockNum, threadNum  >>>(N,r,rnew,k1,k2,k3);
        gpuErrchk( cudaPeekAtLastError() );
/*******************************************************************************************************/
        Get_drdt <<< blockNum, threadNum, threadNum*sizeof(double2)  >>> (N,Np,L.x,dt,gamma,k4,rnew);        
        gpuErrchk( cudaPeekAtLastError() );
        doRungeKuttaStep4<<< blockNum, threadNum >>>(N,r,rnew,k1,k2,k3,k4);
        gpuErrchk( cudaPeekAtLastError() );
/*******************************************************************************************************/
        Get_drdt <<< blockNum, threadNum, threadNum*sizeof(double2)  >>> (N,Np,L.x,dt,gamma,k5,rnew);        
        gpuErrchk( cudaPeekAtLastError() ); 
        doRungeKuttaStep5<<< blockNum, threadNum  >>>(N,r,rnew,k1,k2,k3,k4,k5);
        gpuErrchk( cudaPeekAtLastError() );
/*******************************************************************************************************/
        Get_drdt <<< blockNum, threadNum, threadNum*sizeof(double2)  >>> (N,Np,L.x,dt,gamma,k6,rnew);        
        gpuErrchk( cudaPeekAtLastError( ));
        doRungeKuttaStep6<<< blockNum, threadNum >>>(N,r,rnew,k1,k2,k3,k4,k5,k6);
        gpuErrchk( cudaPeekAtLastError() );
        
        getError <<< blockNum, threadNum  >>> (N,k2,k5);
        gpuErrchk( cudaPeekAtLastError() );
        cudaDeviceSynchronize();
        //Use thrust to check the error...
        double* r4 = &(k2[0].x);
        thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(r4);
        thrust::device_ptr<double> maxval = thrust::max_element(dev_ptr,dev_ptr+N);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk(cudaDeviceSynchronize () );
        double result = maxval[0];
        if (result < tol){
      //    printf("Success: Max error = %g, dt = %g\n",result,dt);
          t+=dt;
          dt *= 0.87*pow(tol/result,0.2);
          success = 1;
          pass++;
          fail_rate = fail/(pass+fail);
          
        }
        else{
          fail++;    
          fail_rate = fail/(pass+fail);
          printf("Fail: Max error = %g, dt = %g, t = %g, fail_rate = %g\n",result,dt,t,fail_rate);	
          dt*= 0.87*pow(tol/result,0.25);
          if (isnan(dt)){ printf("dt is NaN, aborting ...\n"); return -1;}
        }
       // t4 = omp_get_wtime();
      }
      UpdatePositions <<< blockNum ,  threadNum >>> (N,L.x,r,k5);
      gpuErrchk( cudaPeekAtLastError() );
      checkDipoles <<< blockNum, threadNum>>> (N,Np,L.x,r,remove);
      findPairs <<< blockNum , threadNum >>> (N,Np,r,remove);
      cudaDeviceSynchronize();
      ptrdiff_t Pnew, Mnew, Nnew;
	Pnew = 0; Mnew = 0; Nnew = 0;
      int Nold = N;
      Nnew = RemoveVortices(N,Np,r,remove,&Pnew,&Mnew);
      cudaDeviceSynchronize(); 
      int numlost = (N - (int) Nnew/2);
      Np -= numlost/2;
      N = (int) Nnew/2; 
      boundaryLoss <<< blockNum, threadNum >>> (Np,L.x,r,remove);
      cudaDeviceSynchronize();
      Nnew = RemoveVortices(N,Np,r,remove,&Pnew,&Mnew);
      if ((int)Nnew/2<N )
	printf("%d positive vortices removed at boundary\n",N-(int)Nnew/2);
      Np -= (N- ((int) (Nnew/2)));
      N = (int)Nnew/2;
      boundaryLoss <<< blockNum, threadNum >>> (N,L.x,r,remove);
      cudaDeviceSynchronize();
      Nnew = RemoveVortices(N,Np,r,remove,&Pnew,&Mnew);
      if ((int) Nnew/2 <N)
	printf("%d negative vortices removed at boundary\n",N-(int)Nnew/2);
      N =((int) (Nnew/2));

      //Nnew = Pnew + Mnew;
	//printf("%d %d %d\n",N,Np,N-Np);
      //error checking...
      if (Nnew == 0){
        printf("Terminating simulation: all vortices are gone...\n");
        return 0;
      }
      //if (Nnew%2 !=0){ printf("!!ERROR!! : Number of vortices is no longer even!\n"); return -1;}
      if (N <  Nold){ 
      //  printf("N = %d, Np = %d, Nm = %d \n",(int) Nnew/2, (int) Pnew/2, (int) ((Nnew-Pnew)/2) ); 
        //N = (int) Nnew/2;
	//Np = N/2;
        printf("N = %d, Np = %d, Nm = %d \n",N, Np, N-Np ); 
        //Np = N/2;
        int blockNum_check =(N+threadNum-1)/threadNum;
        if (blockNum_check < blockNum){
          blockNum = blockNum_check;
          printf("New number of blocks: %d\n",blockNum);
        }
        while (threadNum > N/2){
         printf("!! Warning !!: threadNum > N/2: reducing threads per block to %d\n",threadNum/2);
         threadNum /= 2;
         blockNum = (N+threadNum-1)/threadNum;
         printf("Continuing with %d blocks of %d threads, %d threads in total...\n",blockNum,threadNum,blockNum*threadNum);
        }
        if (threadNum*blockNum < N){
          printf("!! ERROR !!:  Total number of theads is less than N\n");
          return -1;
        }
      } 

    }
 
    //Bring back to host to save ... 
    gpuErrchk( cudaMemcpy(r_host,r,sizeof(double2)*N,cudaMemcpyDeviceToHost) );
    cudaDeviceSynchronize();
//    current_energy = wmcw_energy(N,r_host,kappa,L.x);
 //   printf("Current energy is: %g\n",current_energy);
 //   fprintf(eFile,"%g\n",current_energy);  
    //Write to file 
    FILE *pFile;
    char filenamep[256];
    sprintf(filenamep,"%dPositivePositions.adat",filenum);
    pFile = fopen(filenamep,"w");
    cudaDeviceSynchronize();  //wait for device to finish copying before writing...
    for (int i=0;i<Np;i++)  fprintf(pFile," %g %g\n",r_host[i].x,r_host[i].y); 
    fclose(pFile);
    char filenamem[256];
    sprintf(filenamem,"%dNegativePositions.adat",filenum);
    pFile = fopen(filenamem,"w");
    for (int i = Np; i <N; i++) fprintf(pFile," %g %g\n",r_host[i].x,r_host[i].y);
    fclose(pFile);
    printf("Generated output: %d\n",filenum);
    filenum++;
    t2 = omp_get_wtime();
    printf("Elapsed time for step: %g s\n",(t2-t1));
  }
  tEnd = omp_get_wtime();
  printf("Simulation took %g s\n",(tEnd-tStart) );
  
  //Cleanup
  free(remove_host);
  gpuErrchk( cudaFree(r) );
  //gpuErrchk( cudaFree(arr) );
  gpuErrchk( cudaFree(remove) ) ;
  free(r_host);
  fclose(eFile);
  return 0;
}
