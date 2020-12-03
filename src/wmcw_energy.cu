#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double wmcw_energy(int n,double2* r,int* sig, double l){

int  i, j, k;
double x, y, inverselx, inversely, h, wmcw_energy;
double pi = 4.0*atan(1.0);

inverselx = 2.0*pi/l;
inversely = 2.0*pi/l;

wmcw_energy = 0.0;
for (i = 0;i<=n-2;i++){
  for (j=i+1;j<=n-1;j++){
    if (i != j){ 
      x = fabs(r[i].x-r[j].x) * inverselx;
      y = fabs(r[i].y-r[j].y) * inversely;
      h = 0.0;
      for (k = -10;k<=10;k++){
        h = h + log((cosh(x-2.0*pi*(double)k)-cos(y))/cosh(2.0*pi*(double)k));
      }
      h = h - x*x/(2.0*pi);
      wmcw_energy = wmcw_energy - h* ((double)sig[i]) *( (double)sig[j]);
    }
  }
}
//Difference between Weiss & McWilliams and (?)Montgomery & Joyce
wmcw_energy = wmcw_energy/(double)n-0.177021697969890;

return wmcw_energy;
 }
