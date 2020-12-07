# GPU Kirchoff

<img align="right" src="vortfluid.gif" width="100" height="100">

The code simulates damped point vortex dynamics in a disk. Annihilations 
and extra repulsion of same-sign vortices are also included at small 
scales. 

The N-body problem is accelerated using a shared memory N-body algorithm,
adapted from H. Nguyen, "GPU Gems 3". 

This allows to solve for N ~ 10^4 - 10^5 vortices.



M. T. Reeves et. al, Physical Review Letters 119 (18), 184502

T. P. Billam et. al, Physical Review A 91 (2), 023615

********************************** COMPILATION ******************************

Compilation *should* only require the following:

module load cuda
cd ./src
make

Producing the executable prog.out. The compiler will throw a lot of warnings
about string to char* conversion, but this is fine. 

However, if the above fails try adding the following to ~/.bash_profile, and
then type "source ~/.bash_profile":

PATH=$PATH:$HOME/bin
PATH=/usr/include:/usr/local/cuda-7.5/bin:${PATH}
LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:${LD_LIBRARY_PATH}
LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH}
export PATH
export LD_LIBRARY_PATH

and then try calling 'make' again.

*********************************** RUNNING *******************************
Examples are provided in the folder ./run 

To run an example simply type: qsub ./PVrun.sh 

PVrun.sh sets up the slurm job and thenit calls prog.out with the format:

    ./prog.out cardNum N L tf loadfile threadsPerBLock Np 

 where 
	cardNum = number of GPU (always 0 unless the node has multiple GPUs)
	N = total number of vortices
	L = Radius of Disk
	tf = integration time 
	loadfile = initial condition containing vortex positions
	Np = number of positive vortices

The loadfile needs to be a (space-delimited) .txt file with the positions in
the following format:

xp_1 yp_1
xp_2 yp_2
  ...
xp_Np xp_NP
xm_1 ym_1 
  .. 
xm_Nm ym_Nm

i.e., all the positive vortex positions listed first, then all the negative
vortex positions.

****************************** NOTES **************************************

Note that the disk radius (L) needs to be >> xi = 1. Vortex-antivortex 
annihilations, boundary losses, and the phenomenological small-scale 
damping for same-sign vortices all kick in at distance = 1. To convert to
the unit disk simply rescale x -> x/L, and t -> t/L^2. (We can modify the 
code if this becomes annoying). 

Simulation output step size (DT) and dissipation (gamma) currently aren't
specified at run time. This may be worth changing so that the code doesn't
need to be recompiled every time these are changed.

Setting gamma=0 exactly causes some issues due to the way the 
additional small-scale damping is implemented. Again, this can be fixed if 
necessary. 

