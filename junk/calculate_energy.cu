
#include <stdio.h>
#include <stdlib.h>
#include "wmcw_energy.h"

int main(int argc, char* argv[]){
  if (argc < 4)
  {
    printf("Usage is: ./energy.out N L numFiles\n");
    return -1;
  }

  int N = atoi(argv[1]);
  double2* r = (double2*) malloc(N*sizeof(double2));
  double L = atof(argv[2]);
  int numFiles = atoi(argv[3]);
    
   FILE* efile = fopen("energy.txt","w");
   
   int* kappa = (int*) malloc(N*sizeof(int));
   for (int j=0;j<N/2;j++){
     kappa[j] = 1;
     kappa[j+N/2] = -1;
   }
  
  
   for (int i=1;i<= numFiles; i++){
     char filename[64];
     sprintf(filename,"%d.txt",i);
     printf("%s\n",filename);
     FILE* pfile = fopen(filename,"r");
     int temp;  
     for (temp =0; temp < N; temp++){
       if (feof(pfile)) break;
       fscanf(pfile,"%lf %lf ", &(r[temp].x),&(r[temp].y));
     }
     fclose(pfile);
     if (temp != N) {
      N = temp; 
     }
  
     for (int j=0;j<N/2;j++){
       kappa[j] = 1;
       kappa[j+N/2] = -1;
     }
     double energy = wmcw_energy(N,r,kappa,L);
     fprintf(efile,"%g\n",energy);
   }
  
 fclose(efile);

return 0;}
