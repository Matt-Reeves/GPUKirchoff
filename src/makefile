src = $(wildcard *.cu)
obj = $(src:.cu=.o)
LDFLAGS = -lcuda -lcudart -lgomp 
CFLAGS= -I. -I../include -arch=sm_20 -rdc=true -Xcompiler -fopenmp -O3

prog.out: $(obj)
	nvcc -o $@ $(CFLAGS) $^  $(LDFLAGS)

%.o : %.cu
	nvcc -c $(CFLAGS) $^ -o $@
.PHONY: clean
clean:
	rm -f $(obj) prog.out
